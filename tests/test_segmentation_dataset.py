from ml.training.build_plate_segmentation_dataset import is_polygon_label


def test_polygon_label_accepts_three_or_more_coordinate_pairs() -> None:
    assert is_polygon_label("0 0.1 0.2 0.8 0.2 0.8 0.7 0.1 0.7".split())


def test_polygon_label_rejects_detection_box() -> None:
    assert not is_polygon_label("0 0.5 0.5 0.4 0.2".split())


def test_polygon_label_rejects_out_of_range_coordinate() -> None:
    assert not is_polygon_label("0 0.1 0.2 1.2 0.2 0.8 0.7".split())
