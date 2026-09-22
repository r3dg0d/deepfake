import pytest

from deepfake.presets import PRESETS, get_preset


def test_presets():
    assert set(PRESETS) == {"low-latency", "balanced", "high-quality"}
    assert get_preset("balanced").fps > 0


def test_unknown_preset():
    with pytest.raises(KeyError):
        get_preset("ultra")
