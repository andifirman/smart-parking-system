"""Build a leakage-aware mixed dataset for camera-domain adaptation."""

from __future__ import annotations

import csv
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "data/processed/plate_segmentation_v1"
CAMERA_ROOT = PROJECT_ROOT / "data/raw/camera_plate_v1"
TARGET_ROOT = PROJECT_ROOT / "data/processed/plate_segmentation_v2"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_TARGETS = {"train": 80, "valid": 20, "test": 18}
CAMERA_TRAIN_REPEATS = 8
RANDOM_SEED = 42


def valid_polygon_label(label_path: Path) -> bool:
    lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return False
    for parts in lines:
        if len(parts) < 7 or (len(parts) - 1) % 2:
            return False
        try:
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            return False
        if parts[0] != "0" or not all(0.0 <= value <= 1.0 for value in coordinates):
            return False
    return True


def capture_time(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def capture_groups(paths: list[Path], maximum_gap_seconds: int = 5) -> list[list[Path]]:
    groups: list[list[Path]] = []
    current: list[Path] = []
    previous_time: datetime | None = None
    for path in sorted(paths):
        timestamp = capture_time(path)
        starts_new_group = (
            current
            and (timestamp is None or previous_time is None or (timestamp - previous_time).total_seconds() > maximum_gap_seconds)
        )
        if starts_new_group:
            groups.append(current)
            current = []
        current.append(path)
        previous_time = timestamp
    if current:
        groups.append(current)
    return groups


def assign_groups(groups: list[list[Path]]) -> dict[str, list[Path]]:
    shuffled = groups.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)
    assignments = {split: [] for split in SPLIT_TARGETS}
    for group in sorted(shuffled, key=len, reverse=True):
        split = max(
            SPLIT_TARGETS,
            key=lambda name: (SPLIT_TARGETS[name] - len(assignments[name])) / SPLIT_TARGETS[name],
        )
        assignments[split].extend(group)
    return assignments


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def add_pair(image: Path, label: Path, split: str, prefix: str) -> Path:
    target_image = TARGET_ROOT / split / "images" / f"{prefix}{image.name}"
    target_label = TARGET_ROOT / split / "labels" / f"{prefix}{label.name}"
    link_or_copy(image, target_image)
    link_or_copy(label, target_label)
    return target_image


def main() -> None:
    if not (LEGACY_ROOT / "data.yaml").is_file():
        raise FileNotFoundError("Build plate_segmentation_v1 before creating the adaptation dataset.")
    camera_images = sorted(
        path for path in (CAMERA_ROOT / "images/unlabeled").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    invalid_or_missing = [
        path.name
        for path in camera_images
        if not (CAMERA_ROOT / "labels" / f"{path.stem}.txt").is_file()
        or not valid_polygon_label(CAMERA_ROOT / "labels" / f"{path.stem}.txt")
    ]
    if invalid_or_missing:
        raise ValueError(f"Invalid or missing camera labels: {invalid_or_missing[:10]}")

    assignments = assign_groups(capture_groups(camera_images))
    camera_lists: dict[str, list[Path]] = {split: [] for split in SPLIT_TARGETS}
    legacy_lists: dict[str, list[Path]] = {split: [] for split in SPLIT_TARGETS}

    for split in SPLIT_TARGETS:
        legacy_split = "valid" if split == "valid" else split
        for image in sorted((LEGACY_ROOT / legacy_split / "images").iterdir()):
            label = LEGACY_ROOT / legacy_split / "labels" / f"{image.stem}.txt"
            if label.is_file():
                legacy_lists[split].append(add_pair(image, label, split, "legacy__"))

        repeats = CAMERA_TRAIN_REPEATS if split == "train" else 1
        for repeat in range(repeats):
            for image in assignments[split]:
                label = CAMERA_ROOT / "labels" / f"{image.stem}.txt"
                target = add_pair(image, label, split, f"camera_r{repeat}__")
                if repeat == 0:
                    camera_lists[split].append(target)

    config = {
        "path": TARGET_ROOT.resolve().as_posix(),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {0: "license"},
    }
    (TARGET_ROOT / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    for domain, lists in (("camera", camera_lists), ("legacy", legacy_lists)):
        for split, paths in lists.items():
            list_path = TARGET_ROOT / f"{domain}_{split}.txt"
            list_path.write_text("\n".join(path.resolve().as_posix() for path in paths) + "\n", encoding="utf-8")

    manifest_path = TARGET_ROOT / "camera_split.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "split"])
        for split, paths in assignments.items():
            writer.writerows((path.name, split) for path in sorted(paths))

    print(f"Capture groups: {len(capture_groups(camera_images))}")
    for split in SPLIT_TARGETS:
        print(
            f"{split}: {len(legacy_lists[split])} legacy + "
            f"{len(camera_lists[split])} unique camera images"
        )
    print(f"Dataset config: {TARGET_ROOT / 'data.yaml'}")
    print(f"Camera split manifest: {manifest_path}")


if __name__ == "__main__":
    main()
