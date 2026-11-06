import cv2
import numpy as np

from apps.vision.src.recognition.anpr_service import (
    candidate_score,
    decision_status,
    normalize_plate_text,
    plate_preprocessing_variants,
    rectify_plate_crop,
)


def test_normalize_plate_text_removes_spacing_and_punctuation() -> None:
    assert normalize_plate_text(" b 1497-lk ") == "B1497LK"


def test_decision_status_accepts_valid_high_confidence_plate() -> None:
    assert decision_status("B1497LK", 0.95, accept_threshold=0.90, review_threshold=0.75) == "ACCEPT"


def test_decision_status_reviews_uncertain_readable_plate() -> None:
    assert decision_status("B1497ZK", 0.82, accept_threshold=0.90, review_threshold=0.75) == "REVIEW"


def test_decision_status_rejects_invalid_low_confidence_text() -> None:
    assert decision_status("X", 0.40, accept_threshold=0.90, review_threshold=0.75) == "REJECT"


def test_candidate_score_rewards_repeated_valid_plate() -> None:
    assert candidate_score("F1506YD", 0.80, votes=2) > candidate_score("F199TH", 0.80)


def test_plate_preprocessing_variants_upscale_small_crop() -> None:
    crop = np.zeros((40, 120, 3), dtype=np.uint8)
    variants = plate_preprocessing_variants(crop)

    assert [name for name, _ in variants] == ["original", "clahe", "otsu", "adaptive"]
    assert all(image.shape[0] == 112 for _, image in variants)


def test_rectify_plate_crop_warps_large_quadrilateral() -> None:
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    polygon = np.array([[35, 30], [220, 10], [205, 95], [20, 110]], dtype=np.int32)
    cv2.polylines(image, [polygon], True, (255, 255, 255), 4)

    rectified = rectify_plate_crop(image)

    assert rectified is not None
    assert rectified.shape[1] > rectified.shape[0]
