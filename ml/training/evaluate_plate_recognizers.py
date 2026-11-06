"""Compare packaged PaddleOCR recognizers on the fixed OCR test split."""

from __future__ import annotations

import csv
import re
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import torch  # noqa: F401 - preload Torch DLLs before Paddle on Windows.
import paddle
from paddleocr import TextRecognition


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data/processed/plate_ocr"
TEST_MANIFEST = DATA_ROOT / "manifests/test.txt"
EXPERIMENT_ROOT = PROJECT_ROOT / "ml/experiments/paddleocr_plate_recognizer"
MODELS = {
    "v2": EXPERIMENT_ROOT / "inference_en_v2",
    "perspective_v3": EXPERIMENT_ROOT / "inference_perspective_v3",
}
RESULT_PATH = EXPERIMENT_ROOT / "perspective_v3_test_comparison.csv"


def normalize_text(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def similarity(prediction: str, ground_truth: str) -> float:
    return SequenceMatcher(None, prediction, ground_truth).ratio()


def read_manifest() -> list[tuple[str, str]]:
    return [tuple(line.split("\t", maxsplit=1)) for line in TEST_MANIFEST.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    device = "gpu:0" if paddle.device.is_compiled_with_cuda() else "cpu"
    rows: list[dict[str, object]] = []
    manifest = read_manifest()
    for model_name, model_dir in MODELS.items():
        reader = TextRecognition(
            model_name="en_PP-OCRv5_mobile_rec",
            model_dir=str(model_dir),
            device=device,
        )
        for index, (relative_path, ground_truth) in enumerate(manifest, start=1):
            image = cv2.imread(str(DATA_ROOT / relative_path))
            if image is None:
                raise ValueError(f"Unreadable test image: {relative_path}")
            outputs = list(reader.predict(input=image, batch_size=1))
            payload = outputs[0].json.get("res", outputs[0].json) if outputs else {}
            prediction = normalize_text(str(payload.get("rec_text", "")))
            rows.append(
                {
                    "model": model_name,
                    "file": relative_path,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "confidence": float(payload.get("rec_score", 0.0)),
                    "exact_match": prediction == ground_truth,
                    "character_similarity": similarity(prediction, ground_truth),
                }
            )
            if index % 50 == 0:
                print(f"{model_name}: {index}/{len(manifest)}")
        del reader

    with RESULT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for model_name in MODELS:
        model_rows = [row for row in rows if row["model"] == model_name]
        exact = sum(bool(row["exact_match"]) for row in model_rows) / len(model_rows)
        similarity_mean = sum(float(row["character_similarity"]) for row in model_rows) / len(model_rows)
        print(f"{model_name}: exact={exact:.2%}, character_similarity={similarity_mean:.2%}")
    print(f"Results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
