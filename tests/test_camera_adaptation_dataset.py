from datetime import datetime, timedelta
from pathlib import Path

from ml.training.build_camera_adaptation_dataset import capture_groups, valid_polygon_label


def test_capture_groups_keeps_burst_images_together() -> None:
    start = datetime(2026, 11, 9, 13, 0, 0)
    paths = [Path((start + timedelta(seconds=offset)).strftime("%Y%m%d_%H%M%S") + ".jpg") for offset in (0, 2, 4, 20)]

    groups = capture_groups(paths)

    assert [len(group) for group in groups] == [3, 1]


def test_valid_polygon_label_accepts_normalized_quad(tmp_path: Path) -> None:
    label = tmp_path / "plate.txt"
    label.write_text("0 0.1 0.2 0.8 0.2 0.8 0.7 0.1 0.7\n", encoding="utf-8")

    assert valid_polygon_label(label)


def test_valid_polygon_label_rejects_box_annotation(tmp_path: Path) -> None:
    label = tmp_path / "plate.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")

    assert not valid_polygon_label(label)
