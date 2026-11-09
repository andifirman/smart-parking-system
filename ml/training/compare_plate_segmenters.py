"""Compare legacy and camera-adapted segmenters on both test domains."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/processed/plate_segmentation_v2"
OUTPUT_ROOT = PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation/comparison_v1_v2"
MODELS = {
    "v1": PROJECT_ROOT / "ml/experiments/yolo_plate_segmentation/yolo11n_plate_seg_v1/weights/best.pt",
    "v2_camera": PROJECT_ROOT
    / "ml/experiments/yolo_plate_segmentation/yolo11n_plate_seg_v2_camera/weights/best.pt",
}


def domain_config(domain: str) -> Path:
    image_list = DATASET_ROOT / f"{domain}_test.txt"
    config = {
        "path": DATASET_ROOT.resolve().as_posix(),
        "train": image_list.resolve().as_posix(),
        "val": image_list.resolve().as_posix(),
        "test": image_list.resolve().as_posix(),
        "names": {0: "license"},
    }
    path = DATASET_ROOT / f"compare_{domain}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_path in MODELS.items():
        if not model_path.is_file():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = YOLO(str(model_path))
        results[model_name] = {}
        for domain in ("legacy", "camera"):
            metrics = model.val(
                data=str(domain_config(domain)),
                split="test",
                imgsz=960,
                device=0,
                project=str(OUTPUT_ROOT),
                name=f"{model_name}_{domain}",
                exist_ok=True,
            )
            results[model_name][domain] = {
                "box_precision": float(metrics.box.mp),
                "box_recall": float(metrics.box.mr),
                "box_map50": float(metrics.box.map50),
                "box_map50_95": float(metrics.box.map),
                "mask_precision": float(metrics.seg.mp),
                "mask_recall": float(metrics.seg.mr),
                "mask_map50": float(metrics.seg.map50),
                "mask_map50_95": float(metrics.seg.map),
            }

    output_path = OUTPUT_ROOT / "metrics.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for domain in ("legacy", "camera"):
        print(f"\n{domain.upper()} TEST")
        for model_name in MODELS:
            values = results[model_name][domain]
            print(
                f"{model_name:10} mask P={values['mask_precision']:.4f} "
                f"R={values['mask_recall']:.4f} mAP50={values['mask_map50']:.4f} "
                f"mAP50-95={values['mask_map50_95']:.4f}"
            )
    print(f"\nComparison saved to: {output_path}")


if __name__ == "__main__":
    main()
