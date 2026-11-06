"""Train and evaluate a YOLO11 plate-segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = PROJECT_ROOT / "data/processed/plate_segmentation_v1/data.yaml"
RUNS_ROOT = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation"
RUN_NAME = "yolo11n_plate_seg_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not DATASET_CONFIG.is_file():
        raise FileNotFoundError(
            f"Segmentation dataset not found: {DATASET_CONFIG}. "
            "Run build_plate_segmentation_dataset.py first."
        )
    model = YOLO(args.model)
    model.train(
        data=str(DATASET_CONFIG),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=5,
        project=str(RUNS_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        cache=False,
        close_mosaic=5,
        degrees=3.0,
        perspective=0.0005,
    )
    best_model = RUNS_ROOT / RUN_NAME / "weights/best.pt"
    YOLO(str(best_model)).val(
        data=str(DATASET_CONFIG),
        split="test",
        imgsz=args.image_size,
        device=args.device,
    )
    print(f"Segmentation model ready: {best_model}")


if __name__ == "__main__":
    main()
