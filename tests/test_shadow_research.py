import numpy as np
import pytest

from deepfake.experimental.shadow import apply_linear_gain, shadow_gain


def test_research_operator_bounds_chromaticity_and_foreground():
    shading = np.tile(np.linspace(0.05, 1, 64), (64, 1))
    skin = np.zeros((64, 64), bool)
    skin[8:56, 8:56] = True
    gain = shadow_gain(shading, skin)
    assert gain.min() >= 0.82 and gain.max() <= 1
    image = np.full((64, 64, 3), (0.3, 0.5, 0.8), np.float32)
    output = apply_linear_gain(image, gain, skin.astype(np.float32))
    assert np.array_equal(output[~skin], image[~skin])
    assert np.all(output <= image + 1e-6)

    def linear(a):
        return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)

    before, after = linear(image), linear(output)
    assert np.allclose(before[..., 0] / before[..., 2], after[..., 0] / after[..., 2], atol=1e-6)
    assert np.any(output[skin] < image[skin])


def test_missing_geometry_is_noop_and_invalid_geometry_rejected():
    assert np.all(shadow_gain(np.zeros((8, 8)), np.ones((8, 8), bool)) == 1)
    with pytest.raises(ValueError):
        shadow_gain(np.full((8, 8), np.nan), np.ones((8, 8), bool))
