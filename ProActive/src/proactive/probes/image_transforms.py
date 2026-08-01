"""
Image transformation probes for the ProActive diagnostic system.

Each transform takes a PIL Image and returns a deterministically
transformed PIL Image. Every probe is applied INDEPENDENTLY to the
original image — never chained on top of another probe's output.
(Plan §2.5, §13.1)

Transforms:
- blank:      Uniform canvas at the same resolution
- blur:       Gaussian blur at configurable sigma
- crop:       Center crop to a fraction, resize back to original
- brightness: Multiply brightness by a factor
- noise:      Additive Gaussian noise at configurable sigma with SHA-256 derived seed
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter

from proactive.utils.hashing import hash_transform


# ---------------------------------------------------------------------------
# Deterministic Seed Derivation (Contract 3)
# ---------------------------------------------------------------------------

def derive_transform_seed(
    global_seed: int = 42,
    instance_id: str = "",
    probe_name: str = "noise",
    severity: float = 25.0,
) -> int:
    """Derive deterministic transform seed from global_seed | instance_id | probe_name | severity.

    Uses SHA-256 to ensure cross-process and cross-platform reproducibility.
    """
    key = f"{global_seed}|{instance_id}|{probe_name}|{severity}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Blank probe  (Plan §13.1)
# ---------------------------------------------------------------------------

def apply_blank(
    image: Image.Image,
    fill_color: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """Replace image with a uniform blank canvas at the same resolution."""
    return Image.new("RGB", image.size, fill_color)


def blank_transform_hash(
    fill_color: Tuple[int, int, int] = (128, 128, 128),
) -> str:
    """Hash the blank transform specification."""
    return hash_transform("blank", {"fill_color": list(fill_color)})


# ---------------------------------------------------------------------------
# Blur probe  (Plan §13.1, §13.2)
# ---------------------------------------------------------------------------

def apply_blur(
    image: Image.Image,
    sigma: float = 15.0,
) -> Image.Image:
    """Apply Gaussian blur to the image."""
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def blur_transform_hash(sigma: float = 15.0) -> str:
    """Hash the blur transform specification."""
    return hash_transform("blur", {"sigma": sigma})


# ---------------------------------------------------------------------------
# Crop probe  (Plan §13.1, §13.2)
# ---------------------------------------------------------------------------

def apply_crop(
    image: Image.Image,
    retain_fraction: float = 0.50,
) -> Image.Image:
    """Center-crop to retain_fraction of width and height, resize back."""
    w, h = image.size
    new_w = max(1, int(w * retain_fraction))
    new_h = max(1, int(h * retain_fraction))

    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h

    cropped = image.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


def crop_transform_hash(retain_fraction: float = 0.50) -> str:
    """Hash the crop transform specification."""
    return hash_transform("crop", {"retain_fraction": retain_fraction})


# ---------------------------------------------------------------------------
# Brightness probe  (Plan §13.1, §13.2)
# ---------------------------------------------------------------------------

def apply_brightness(
    image: Image.Image,
    factor: float = 0.30,
) -> Image.Image:
    """Multiply pixel brightness by a factor."""
    arr = np.array(image, dtype=np.float32)
    arr = arr * factor
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def brightness_transform_hash(factor: float = 0.30) -> str:
    """Hash the brightness transform specification."""
    return hash_transform("brightness", {"factor": factor})


# ---------------------------------------------------------------------------
# Noise probe  (Plan §13.1, §13.2)
# ---------------------------------------------------------------------------

def apply_noise(
    image: Image.Image,
    sigma: float = 25.0,
    seed: int = 42,
) -> Image.Image:
    """Add Gaussian noise to the image using a deterministic seed."""
    rng = np.random.RandomState(seed)
    arr = np.array(image, dtype=np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def noise_transform_hash(sigma: float = 25.0, seed: int = 42) -> str:
    """Hash the noise transform specification."""
    return hash_transform("noise", {"sigma": sigma, "seed": seed})


# ---------------------------------------------------------------------------
# Dispatcher & Constants
# ---------------------------------------------------------------------------

CANONICAL_SEVERITIES: Dict[str, float] = {
    "blur": 15.0,
    "crop": 0.50,
    "brightness": 0.30,
    "noise": 25.0,
}

PILOT_SEVERITIES: Dict[str, List[float]] = {
    "blur": [8.0, 15.0, 22.0],
    "crop": [0.35, 0.50, 0.65],
    "brightness": [0.15, 0.30, 0.50],
    "noise": [10.0, 25.0, 50.0],
}


def apply_image_transform(
    image: Image.Image,
    probe_name: str,
    severity: float | None = None,
    noise_seed: int = 42,
) -> Image.Image:
    """Apply a named image transform with optional severity override."""
    if probe_name == "blank":
        return apply_blank(image)
    elif probe_name == "blur":
        s = severity if severity is not None else CANONICAL_SEVERITIES["blur"]
        return apply_blur(image, sigma=s)
    elif probe_name == "crop":
        s = severity if severity is not None else CANONICAL_SEVERITIES["crop"]
        return apply_crop(image, retain_fraction=s)
    elif probe_name == "brightness":
        s = severity if severity is not None else CANONICAL_SEVERITIES["brightness"]
        return apply_brightness(image, factor=s)
    elif probe_name == "noise":
        s = severity if severity is not None else CANONICAL_SEVERITIES["noise"]
        return apply_noise(image, sigma=s, seed=noise_seed)
    else:
        raise ValueError(
            f"Unknown image transform: '{probe_name}'. "
            f"Valid: blank, blur, crop, brightness, noise."
        )


def get_transform_hash(
    probe_name: str,
    severity: float | None = None,
    noise_seed: int = 42,
) -> str:
    """Get the hash for a transform specification."""
    if probe_name == "blank":
        return blank_transform_hash()
    elif probe_name == "blur":
        s = severity if severity is not None else CANONICAL_SEVERITIES["blur"]
        return blur_transform_hash(s)
    elif probe_name == "crop":
        s = severity if severity is not None else CANONICAL_SEVERITIES["crop"]
        return crop_transform_hash(s)
    elif probe_name == "brightness":
        s = severity if severity is not None else CANONICAL_SEVERITIES["brightness"]
        return brightness_transform_hash(s)
    elif probe_name == "noise":
        s = severity if severity is not None else CANONICAL_SEVERITIES["noise"]
        return noise_transform_hash(s, noise_seed)
    else:
        raise ValueError(f"Unknown image transform: '{probe_name}'.")


def export_sample_transformed_images(
    sample_images: Sequence[Tuple[str, Image.Image]],
    output_dir: Union[str, Path],
    probes: Optional[List[str]] = None,
    global_seed: int = 42,
) -> Dict[str, List[Path]]:
    """Export transformed sample images for manual inspection (§25.5 gate).

    Args:
        sample_images: List of (instance_id, PIL.Image) tuples.
        output_dir: Directory where inspected images will be saved.
        probes: List of probe names (defaults to all visual probes).
        global_seed: Global seed for deterministic noise generation.

    Returns:
        Dict mapping probe name to list of saved image paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if probes is None:
        probes = ["blank", "blur", "crop", "brightness", "noise"]

    exported: Dict[str, List[Path]] = {p: [] for p in probes}

    for inst_id, img in sample_images:
        for probe_name in probes:
            probe_dir = out_path / probe_name
            probe_dir.mkdir(exist_ok=True)

            sev = CANONICAL_SEVERITIES.get(probe_name)
            seed = derive_transform_seed(global_seed, inst_id, probe_name, sev or 0.0)

            t_img = apply_image_transform(img, probe_name, severity=sev, noise_seed=seed)
            save_file = probe_dir / f"{inst_id}_{probe_name}.png"
            t_img.save(save_file)
            exported[probe_name].append(save_file)

    return exported
