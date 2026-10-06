import cv2
import numpy as np
import pytest

from deepfake.detect import arcface_transform

REFERENCE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], np.float32
)


def test_similarity_preserves_geometry_and_inverts_to_native_roi():
    angle = 0.12
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = REFERENCE @ rotation.T * 3 + [150, 70]
    matrix = arcface_transform(points)
    target = (REFERENCE + [8, 0]) * 2
    assert np.allclose(points @ matrix[:, :2].T + matrix[:, 2], target, atol=1e-3)
    inverse = cv2.invertAffineTransform(matrix)
    assert np.allclose(target @ inverse[:, :2].T + inverse[:, 2], points, atol=1e-3)
    # Two eyes occupy ~70 native model pixels rather than a padded square crop.
    assert np.linalg.norm(target[1] - target[0]) > 70
    source = arcface_transform(points, 112, source=True)
    assert np.allclose(points @ source[:, :2].T + source[:, 2], REFERENCE, atol=1e-3)


@pytest.mark.parametrize("points", [None, np.zeros((5, 2)), np.full((5, 2), np.nan), REFERENCE[::-1]])
def test_invalid_landmarks_cannot_create_an_alignment(points):
    assert arcface_transform(points) is None


def test_unwarp_coverage_excludes_unmapped_pixels():
    from deepfake.swap.alphaface import AlphaFaceSwapper
    from deepfake.swap.base import SwapResult

    swapper = object.__new__(AlphaFaceSwapper)
    swapper.swap = lambda frame: SwapResult(np.full_like(frame, 200), 1, "test")
    native = np.zeros((512, 512, 3), np.uint8)
    result = swapper.swap_aligned(native, REFERENCE * 2 + [140, 80])
    assert result.face_bgr.shape == native.shape
    assert result.coverage_mask.shape == native.shape[:2]
    assert result.coverage_mask.any() and not result.coverage_mask[0].any()
    assert not np.any(result.face_bgr[result.coverage_mask == 0] == 255)


def test_unmapped_aligned_pixels_are_hidden_from_composite_and_framegen(monkeypatch):
    from deepfake import pipeline
    from deepfake.detect import FaceBox
    from deepfake.occlusion import OcclusionEngine
    from deepfake.pipeline import FaceSwapPipeline, build_config
    from deepfake.swap.base import SwapResult

    class Detector:
        def detect(self, frame):
            return [FaceBox(16, 16, 64, 64, landmarks=REFERENCE * 0.4 + [20, 10])]

    class Masks:
        def parse(self, crop):
            return np.ones(crop.shape[:2], np.float32), 0.95

        def visible(self, crop):
            return np.ones(crop.shape[:2], np.float32)

    class Aligned:
        name = "test"

        def swap_aligned(self, crop, landmarks):
            coverage = np.zeros(crop.shape[:2], np.float32)
            coverage[:, : crop.shape[1] // 2] = 1
            return SwapResult(np.full_like(crop, 220), 1, self.name, coverage)

    monkeypatch.setattr(pipeline, "create_detector", lambda _: Detector())
    monkeypatch.setattr(pipeline, "create_swapper", lambda *args, **kwargs: Aligned())
    monkeypatch.setattr(pipeline, "OcclusionEngine", lambda *a, **k: OcclusionEngine(models=Masks()))
    p = FaceSwapPipeline(build_config("realtime"))
    original = np.full((96, 96, 3), 100, np.uint8)
    output, _ = p.swap_frame(original)
    _, alpha, _ = p.framegen_context(original)
    assert not alpha[:, 48:].any()
    assert np.array_equal(output[:, 48:], original[:, 48:])
    assert np.any(output[:, :48] != original[:, :48])
