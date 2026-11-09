"""Evaluate the candidate detector with the packaged OCR recognizer."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from apps.vision.src.recognition.anpr_service import AnprService, AnprSettings

MODEL_PATH = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation/yolo11n_plate_seg_v2_camera/weights/best.pt"
RECOGNIZER_DIR = PROJECT_ROOT / "ml/weights/anpr/recognizer"
CAMERA_ROOT = PROJECT_ROOT / "data/raw/camera_plate_v1"
SPLIT_MANIFEST = PROJECT_ROOT / "data/processed/plate_segmentation_v2/camera_split.csv"
OUTPUT_PATH = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation/comparison_v1_v2/camera_anpr.csv"
GROUND_TRUTH = "D4574ACH"


def test_images() -> list[Path]:
    with SPLIT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        names = [row["file"] for row in csv.DictReader(handle) if row["split"] == "test"]
    return [CAMERA_ROOT / "images/unlabeled" / name for name in names]


def main() -> None:
    service = AnprService(AnprSettings(detector_path=MODEL_PATH, recognizer_dir=RECOGNIZER_DIR))
    rows: list[dict[str, object]] = []
    for image_path in test_images():
        result = service.analyze_bytes(image_path.read_bytes())
        detections = result["detections"]
        texts = [str(item["plate_text"]) for item in detections]
        rows.append(
            {
                "file": image_path.name,
                "detection_count": len(detections),
                "texts": "|".join(texts),
                "exact_match": GROUND_TRUTH in texts,
                "rectified": any(item["geometry_method"] == "segmentation_rectified" for item in detections),
                "processing_ms": result["processing_ms"],
            }
        )
        print(f"{image_path.name}: {texts or ['NO DETECTION']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    exact = sum(bool(row["exact_match"]) for row in rows)
    detected = sum(int(row["detection_count"]) > 0 for row in rows)
    rectified = sum(bool(row["rectified"]) for row in rows)
    average_ms = sum(float(row["processing_ms"]) for row in rows) / len(rows)
    print(f"\nDetected: {detected}/{len(rows)}")
    print(f"Rectified: {rectified}/{len(rows)}")
    print(f"Exact OCR: {exact}/{len(rows)} ({exact / len(rows):.1%})")
    print(f"Average pipeline latency: {average_ms:.1f} ms/image")
    print(f"Results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
