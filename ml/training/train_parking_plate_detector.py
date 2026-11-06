"""Fine-tune the license-plate detector for the parking CCTV domain."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = PROJECT_ROOT / "TEMPORARY/Test V2"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "ml/experiments/yolo_license_plate_detector/yolo11n_plate_v1/weights/best.pt"
)
RUNS_DIR = PROJECT_ROOT / "ml/experiments/yolo_license_plate_detector"
RUN_NAME = "yolo11n_parking_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def validate_dataset(dataset: Path) -> None:
    required = [
        dataset / split / kind
        for split in ("train", "valid", "test")
        for kind in ("images", "labels")
    ]
    missing = [path for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Missing dataset directories:\n" + "\n".join(map(str, missing)))


def write_dataset_config(dataset: Path) -> Path:
    config_path = RUNS_DIR / "parking_v2_dataset.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": dataset.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "license"},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def main() -> None:
    args = parse_args()
    validate_dataset(args.dataset)
    if not args.model.is_file():
        raise FileNotFoundError(f"Base detector not found: {args.model}")

    dataset_config = write_dataset_config(args.dataset)
    model = YOLO(str(args.model))
    model.train(
        data=str(dataset_config),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=10,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        cache=False,
        close_mosaic=5,
        degrees=3.0,
        perspective=0.0005,
    )

    best_model = RUNS_DIR / RUN_NAME / "weights/best.pt"
    evaluator = YOLO(str(best_model))
    evaluator.val(data=str(dataset_config), split="test", imgsz=args.image_size, device=args.device)
    print(f"Detector ready: {best_model}")


if __name__ == "__main__":
    main()
