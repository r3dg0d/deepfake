import numpy as np
import pytest

from deepfake.framegen.nvof import NvidiaOpticalFlowBackend

pytestmark = pytest.mark.gpu


def backend():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    result = NvidiaOpticalFlowBackend("quality")
    result.initialize(128, 128)
    return result


def test_translation_moves_midpoint_in_correct_direction():
    fg = backend()
    try:
        image = np.repeat(np.tile(np.arange(128, dtype=np.uint8)[None, :, None], (128, 1, 1)), 3, 2)
        fg.push(image)
        fg.push(np.roll(image, 12, axis=1))
        forward = np.zeros((128, 128, 2), np.float32)
        forward[:, :, 0] = 12
        generated = fg._warp_blend(0.5, forward, -forward)
        expected = np.roll(image, 6, axis=1)
        assert np.mean(np.abs(generated[16:-16, 20:-20].astype(float) - expected[16:-16, 20:-20])) < 2
    finally:
        fg.shutdown()


def test_generated_foreground_uses_target_pixels_after_sharpening():
    fg = backend()
    try:
        original = np.full((128, 128, 3), 40, np.uint8)
        swapped = np.full_like(original, 200)
        visible = np.ones((128, 128), np.float32)
        visible[32:96, 32:96] = 0
        context = (original, visible, np.ones_like(visible))
        for _ in range(2):
            fg.set_composite_context(context)
            fg.push(swapped)
        flow = np.zeros((128, 128, 2), np.float32)
        generated = fg._warp_blend(0.5, flow, flow)
        assert np.array_equal(generated[40:88, 40:88], original[40:88, 40:88])
        assert np.all(generated[10:20, 10:20] == 200)
    finally:
        fg.shutdown()
