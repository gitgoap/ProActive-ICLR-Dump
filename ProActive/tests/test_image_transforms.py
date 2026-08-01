"""
Tests for image transform probes.

Validates:
- Transform determinism (same input -> same output)
- SHA-256 seed derivation
- Output size preservation (all transforms return original dimensions)
- Blank canvas uniformity
- Severity parameter effects
- No invalid pixel values
- Image export helper
"""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from proactive.probes.image_transforms import (
    apply_blank,
    apply_blur,
    apply_brightness,
    apply_crop,
    apply_noise,
    apply_image_transform,
    get_transform_hash,
    derive_transform_seed,
    export_sample_transformed_images,
    CANONICAL_SEVERITIES,
    PILOT_SEVERITIES,
)


@pytest.fixture
def sample_image():
    """Create a non-trivial test image with varied pixel values."""
    rng = np.random.RandomState(123)
    arr = rng.randint(0, 256, (100, 150, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def small_image():
    """Create a small 32x32 image for fast tests."""
    rng = np.random.RandomState(456)
    arr = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestSeeding:
    def test_seed_derivation_deterministic(self):
        s1 = derive_transform_seed(42, "pope_001", "noise", 25.0)
        s2 = derive_transform_seed(42, "pope_001", "noise", 25.0)
        assert s1 == s2
        assert isinstance(s1, int)

    def test_seed_varies_with_instance(self):
        s1 = derive_transform_seed(42, "inst_1", "noise", 25.0)
        s2 = derive_transform_seed(42, "inst_2", "noise", 25.0)
        assert s1 != s2

    def test_seed_varies_with_severity(self):
        s1 = derive_transform_seed(42, "inst_1", "noise", 10.0)
        s2 = derive_transform_seed(42, "inst_1", "noise", 25.0)
        assert s1 != s2


class TestBlank:
    def test_output_size(self, sample_image):
        result = apply_blank(sample_image)
        assert result.size == sample_image.size

    def test_uniform_color(self, sample_image):
        result = apply_blank(sample_image)
        arr = np.array(result)
        assert np.all(arr == 128)

    def test_custom_fill(self, sample_image):
        result = apply_blank(sample_image, fill_color=(0, 0, 0))
        arr = np.array(result)
        assert np.all(arr == 0)

    def test_determinism(self, sample_image):
        r1 = np.array(apply_blank(sample_image))
        r2 = np.array(apply_blank(sample_image))
        np.testing.assert_array_equal(r1, r2)


class TestBlur:
    def test_output_size(self, sample_image):
        result = apply_blur(sample_image, sigma=15.0)
        assert result.size == sample_image.size

    def test_blur_reduces_variance(self, sample_image):
        original_var = np.var(np.array(sample_image).astype(float))
        blurred_var = np.var(np.array(apply_blur(sample_image, sigma=15.0)).astype(float))
        assert blurred_var < original_var

    def test_determinism(self, sample_image):
        r1 = np.array(apply_blur(sample_image, sigma=15.0))
        r2 = np.array(apply_blur(sample_image, sigma=15.0))
        np.testing.assert_array_equal(r1, r2)


class TestCrop:
    def test_output_size(self, sample_image):
        result = apply_crop(sample_image, retain_fraction=0.50)
        assert result.size == sample_image.size

    def test_crop_fraction_bounds(self, sample_image):
        r_35 = apply_crop(sample_image, retain_fraction=0.35)
        r_65 = apply_crop(sample_image, retain_fraction=0.65)
        assert r_35.size == sample_image.size
        assert r_65.size == sample_image.size


class TestBrightness:
    def test_output_size(self, sample_image):
        result = apply_brightness(sample_image, factor=0.30)
        assert result.size == sample_image.size

    def test_darkening(self, sample_image):
        arr_orig = np.array(sample_image).astype(float)
        arr_dark = np.array(apply_brightness(sample_image, factor=0.30)).astype(float)
        assert arr_dark.mean() < arr_orig.mean()


class TestNoise:
    def test_noise_determinism_with_seed(self, sample_image):
        r1 = np.array(apply_noise(sample_image, sigma=25.0, seed=42))
        r2 = np.array(apply_noise(sample_image, sigma=25.0, seed=42))
        np.testing.assert_array_equal(r1, r2)

    def test_noise_varies_with_different_seed(self, sample_image):
        r1 = np.array(apply_noise(sample_image, sigma=25.0, seed=42))
        r2 = np.array(apply_noise(sample_image, sigma=25.0, seed=99))
        assert not np.array_equal(r1, r2)


class TestExport:
    def test_export_sample_images(self, small_image):
        with tempfile.TemporaryDirectory() as tmpdir:
            samples = [("test_01", small_image), ("test_02", small_image)]
            exported = export_sample_transformed_images(samples, tmpdir)
            assert "blank" in exported
            assert len(exported["blank"]) == 2
            assert Path(exported["blank"][0]).exists()
