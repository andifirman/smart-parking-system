import numpy as np

from ml.training.build_perspective_ocr_dataset import deterministic_rng, perspective_warp


def test_perspective_warp_preserves_image_dimensions() -> None:
    image = np.full((48, 160, 3), 127, dtype=np.uint8)

    augmented = perspective_warp(image, deterministic_rng("plate-1"), strength=0.18)

    assert augmented.shape == image.shape


def test_deterministic_rng_repeats_same_augmentation_parameters() -> None:
    first = deterministic_rng("plate-1").uniform(size=5)
    second = deterministic_rng("plate-1").uniform(size=5)

    assert np.array_equal(first, second)
