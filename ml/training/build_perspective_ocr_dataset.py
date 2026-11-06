"""Build deterministic perspective augmentations for plate OCR fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/processed/plate_ocr"
SOURCE_MANIFEST = DATA_ROOT / "manifests/train.txt"
TARGET_MANIFEST = DATA_ROOT / "manifests/train_perspective_v3.txt"
TARGET_IMAGE_ROOT = DATA_ROOT / "images/train_perspective_v3"
V2_CONFIG = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer/plate_recognizer_en_v2.yml"
V3_CONFIG = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer/plate_recognizer_perspective_v3.yml"
V2_CHECKPOINT = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer/output_en_v2/best_model/model"
V3_OUTPUT = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer/output_perspective_v3_balanced"


def deterministic_rng(key: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(seed)


def perspective_warp(image: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Apply camera-like horizontal perspective while retaining all characters."""
    height, width = image.shape[:2]
    inset = max(1.0, width * strength)
    vertical = max(1.0, height * strength * 0.55)
    direction = -1 if rng.random() < 0.5 else 1
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    if direction < 0:
        target = np.float32(
            [[inset, vertical], [width - 1, 0], [width - 1, height - 1], [inset, height - 1 - vertical]]
        )
    else:
        target = np.float32(
            [[0, 0], [width - 1 - inset, vertical], [width - 1 - inset, height - 1 - vertical], [0, height - 1]]
        )
    matrix = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def degrade_image(image: np.ndarray, rng: np.random.Generator, variant: int) -> np.ndarray:
    strength = (0.08, 0.13, 0.18, 0.22)[variant]
    result = perspective_warp(image, rng, strength)

    angle = float(rng.uniform(-3.0, 3.0))
    height, width = result.shape[:2]
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    result = cv2.warpAffine(result, rotation, (width, height), borderMode=cv2.BORDER_REPLICATE)

    alpha = float(rng.uniform(0.75, 1.25))
    beta = float(rng.uniform(-24, 24))
    result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
    if variant >= 1:
        kernel = 3 if variant < 3 else 5
        result = cv2.GaussianBlur(result, (kernel, kernel), sigmaX=0)
    if variant >= 2:
        glare = np.zeros_like(result)
        center = (int(rng.uniform(0.2, 0.8) * width), int(rng.uniform(0.2, 0.8) * height))
        axes = (max(4, width // 6), max(2, height // 4))
        cv2.ellipse(glare, center, axes, float(rng.uniform(-20, 20)), 0, 360, (80, 80, 80), -1)
        glare = cv2.GaussianBlur(glare, (0, 0), sigmaX=max(1.0, height / 12))
        result = cv2.add(result, glare)

    quality = int(rng.integers(45 if variant >= 2 else 65, 88))
    ok, encoded = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Failed to encode augmented image.")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Failed to decode augmented image.")
    return decoded


def read_manifest(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid manifest row {path}:{line_number}")
        rows.append((parts[0], parts[1]))
    return rows


def build_dataset(variants_per_image: int, original_repeats: int) -> int:
    rows = read_manifest(SOURCE_MANIFEST)
    TARGET_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    output_rows = rows * original_repeats
    for index, (relative_path, label) in enumerate(rows, start=1):
        source = DATA_ROOT / relative_path
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Unreadable source image: {source}")
        stem = Path(relative_path).stem
        for variant in range(variants_per_image):
            rng = deterministic_rng(f"{relative_path}:{variant}")
            augmented = degrade_image(image, rng, variant)
            target_name = f"{stem}__perspective_{variant + 1:02d}.jpg"
            target = TARGET_IMAGE_ROOT / target_name
            if not cv2.imwrite(str(target), augmented):
                raise ValueError(f"Failed to write augmented image: {target}")
            output_rows.append((target.relative_to(DATA_ROOT).as_posix(), label))
        if index % 100 == 0:
            print(f"Augmented {index}/{len(rows)} source images")

    TARGET_MANIFEST.write_text(
        "".join(f"{relative_path}\t{label}\n" for relative_path, label in output_rows),
        encoding="utf-8",
    )
    return len(output_rows)


def write_v3_config() -> None:
    config = yaml.safe_load(V2_CONFIG.read_text(encoding="utf-8"))
    config["Global"].update(
        {
            "epoch_num": 12,
            "save_model_dir": str(V3_OUTPUT),
            "pretrained_model": str(V2_CHECKPOINT.with_suffix(".pdparams")),
            "checkpoints": None,
            "save_epoch_step": 2,
            "eval_batch_step": [0, 100],
        }
    )
    config["Optimizer"]["lr"].update({"learning_rate": 0.00001, "warmup_epoch": 1})
    config["Train"]["dataset"]["label_file_list"] = [str(TARGET_MANIFEST)]
    config["Train"]["dataset"]["transforms"][1]["RecConAug"]["prob"] = 0.10
    V3_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", type=int, default=2, choices=range(1, 5))
    parser.add_argument("--original-repeats", type=int, default=2, choices=range(1, 5))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [SOURCE_MANIFEST, V2_CONFIG, V2_CHECKPOINT.with_suffix(".pdparams")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(map(str, missing)))
    total = build_dataset(args.variants, args.original_repeats)
    write_v3_config()
    print(f"Training rows: {total}")
    print(f"Manifest: {TARGET_MANIFEST}")
    print(f"Config: {V3_CONFIG}")


if __name__ == "__main__":
    main()
