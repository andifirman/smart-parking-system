"""Create a polygon-only YOLO dataset from the mixed Test V2 annotations."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "TEMPORARY/Test V2"
TARGET_ROOT = PROJECT_ROOT / "data/processed/plate_segmentation_v1"
DATASET_CONFIG = TARGET_ROOT / "data.yaml"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def is_polygon_label(parts: list[str]) -> bool:
    """Accept class plus at least three normalized x/y coordinate pairs."""
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        return False
    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except ValueError:
        return False
    return class_id == 0 and all(0.0 <= value <= 1.0 for value in coordinates)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_split(split: str) -> tuple[int, int, int]:
    source_images = SOURCE_ROOT / split / "images"
    source_labels = SOURCE_ROOT / split / "labels"
    target_images = TARGET_ROOT / split / "images"
    target_labels = TARGET_ROOT / split / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    polygon_count = 0
    missing_image_count = 0
    for label_path in sorted(source_labels.glob("*.txt")):
        polygons = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and is_polygon_label(line.split())
        ]
        if not polygons:
            continue
        image_path = next(
            (source_images / f"{label_path.stem}{extension}" for extension in IMAGE_EXTENSIONS if (source_images / f"{label_path.stem}{extension}").is_file()),
            None,
        )
        if image_path is None:
            missing_image_count += 1
            continue
        link_or_copy(image_path, target_images / image_path.name)
        (target_labels / label_path.name).write_text("\n".join(polygons) + "\n", encoding="utf-8")
        image_count += 1
        polygon_count += len(polygons)
    return image_count, polygon_count, missing_image_count


def main() -> None:
    statistics = {split: build_split(split) for split in ("train", "valid", "test")}
    config = {
        "path": TARGET_ROOT.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "license"},
    }
    DATASET_CONFIG.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for split, (images, polygons, missing_images) in statistics.items():
        print(f"{split}: {images} images, {polygons} polygons, {missing_images} orphan labels skipped")
    print(f"Dataset config: {DATASET_CONFIG}")


if __name__ == "__main__":
    main()
