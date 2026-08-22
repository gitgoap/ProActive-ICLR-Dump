from __future__ import annotations

from PIL import Image
import pytest

from proactive.models.internvl_adapter import (
    InternVLAdapter,
    _candidate_ratios,
    dynamic_preprocess,
    validate_internvl_transformers_version,
)


def test_internvl_accepts_pinned_transformers_version():
    validate_internvl_transformers_version("4.37.2")


@pytest.mark.parametrize("version", ["4.36.2", "4.44.0", "5.0.0", "5.5.4"])
def test_internvl_rejects_incompatible_transformers_versions(version):
    with pytest.raises(RuntimeError, match="isolated proactive-internvl"):
        validate_internvl_transformers_version(version)


def test_internvl_candidate_ratios_are_unique_and_deterministic():
    first = _candidate_ratios(1, 12)
    second = _candidate_ratios(1, 12)
    assert first == second
    assert len(first) == len(set(first))
    assert all(1 <= width * height <= 12 for width, height in first)


def test_internvl_dynamic_preprocess_tiles_wide_image_and_adds_thumbnail():
    image = Image.new("RGB", (896, 448), color="blue")
    patches = dynamic_preprocess(image)
    assert len(patches) == 3
    assert all(patch.size == (448, 448) for patch in patches)


def test_internvl_dynamic_preprocess_is_deterministic():
    image = Image.new("RGB", (731, 419), color=(10, 20, 30))
    first = dynamic_preprocess(image)
    second = dynamic_preprocess(image)
    assert len(first) == len(second)
    assert [patch.tobytes() for patch in first] == [patch.tobytes() for patch in second]


def test_internvl_generation_kwargs_are_deterministic_and_drop_none():
    values = InternVLAdapter._generation_kwargs(
        {
            "do_sample": False,
            "temperature": None,
            "max_new_tokens": 16,
            "num_beams": 1,
        },
        eos_token_id=7,
    )
    assert values == {
        "do_sample": False,
        "max_new_tokens": 16,
        "num_beams": 1,
        "eos_token_id": 7,
        "return_dict_in_generate": True,
        "output_scores": True,
        "use_cache": True,
    }


@pytest.mark.parametrize(
    "config, message",
    [
        ({"do_sample": True, "num_beams": 1}, "deterministic"),
        ({"do_sample": False, "num_beams": 2}, "num_beams=1"),
    ],
)
def test_internvl_generation_kwargs_reject_nondeterministic_or_beam_modes(
    config, message
):
    with pytest.raises(ValueError, match=message):
        InternVLAdapter._generation_kwargs(config, eos_token_id=7)
