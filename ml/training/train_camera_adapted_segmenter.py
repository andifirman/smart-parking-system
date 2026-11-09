"""Fine-tune the plate segmenter on legacy and camera-domain data."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/processed/plate_segmentation_v2"
DATASET_CONFIG = DATASET_ROOT / "data.yaml"
BASE_CHECKPOINT = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation/yolo11n_plate_seg_v1/weights/best.pt"
RUNS_ROOT = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation"
RUN_NAME = "yolo11n_plate_seg_v2_camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def evaluation_config(name: str, image_list: Path) -> Path:
    config = {
        "path": DATASET_ROOT.resolve().as_posix(),
        "train": image_list.resolve().as_posix(),
        "val": image_list.resolve().as_posix(),
        "test": image_list.resolve().as_posix(),
        "names": {0: "license"},
    }
    path = DATASET_ROOT / f"eval_{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    for required in (DATASET_CONFIG, BASE_CHECKPOINT):
        if not required.is_file():
            raise FileNotFoundError(f"Required artifact not found: {required}")

    model = YOLO(str(BASE_CHECKPOINT))
    model.train(
        data=str(DATASET_CONFIG),
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=0,
        patience=3,
        project=str(RUNS_ROOT),
        name=RUN_NAME,
        exist_ok=True,
        lr0=0.001,
        lrf=0.01,
        close_mosaic=3,
        degrees=4.0,
        perspective=0.0008,
    )

    best_model = RUNS_ROOT / RUN_NAME / "weights/best.pt"
    evaluator = YOLO(str(best_model))
    for domain in ("legacy", "camera"):
        config = evaluation_config(domain, DATASET_ROOT / f"{domain}_test.txt")
        metrics = evaluator.val(data=str(config), split="test", imgsz=args.image_size, device=args.device)
        print(f"{domain} mask mAP50: {metrics.seg.map50:.4f}; mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"Candidate model ready: {best_model}")


if __name__ == "__main__":
    main()
