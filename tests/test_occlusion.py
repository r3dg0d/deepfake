import cv2
import numpy as np
import pytest

from deepfake.composite import paste_face
from deepfake.detect import FaceBox
from deepfake.occlusion import OcclusionEngine


class ColorOccluder:
    """Deterministic model double: the foreground is bright red, face gray."""

    backend = "test-color-segmenter"

    def parse(self, crop):
        return np.ones(crop.shape[:2], np.float32), 0.95

    def visible(self, crop):
        return (crop[:, :, 2] < 200).astype(np.float32)


@pytest.mark.parametrize("kind", ["hand", "hair", "glasses", "microphone", "cup", "phone", "full"])
def test_foreground_remains_target_immediately(kind):
    frame = np.full((128, 128, 3), 100, np.uint8)
    gt = np.zeros((128, 128), np.uint8)
    if kind == "glasses":
        cv2.rectangle(gt, (24, 38), (60, 64), 1, 4)
        cv2.rectangle(gt, (68, 38), (104, 64), 1, 4)
    elif kind == "hair":
        cv2.line(gt, (18, 12), (100, 56), 1, 14)
    elif kind == "microphone":
        cv2.circle(gt, (64, 95), 20, 1, -1)
    elif kind == "full":
        gt[:] = 1
    else:
        cv2.rectangle(gt, (35, 20), (90, 112), 1, -1)
    engine = OcclusionEngine(models=ColorOccluder())
    state = engine.estimate(frame).temporal_state
    target = frame.copy()
    target[gt > 0] = (20, 20, 240)
    estimate = engine.estimate(target, temporal_state=state)
    swapped = np.full_like(frame, (200, 180, 60))
    out = paste_face(target, swapped, FaceBox(0, 0, 128, 128), color_match=False, visible_mask=estimate.visible_mask)
    assert np.array_equal(out[gt > 0], target[gt > 0])
    if kind == "full":
        assert np.array_equal(out, target)
    else:
        assert np.any(out[gt == 0] != target[gt == 0])
    recovered = engine.estimate(frame, temporal_state=estimate.temporal_state)
    assert recovered.visible_mask[64, 64] >= 0.85


def test_missing_models_preserves_original(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_MASK_MODELS", "/does-not-exist")
    result = OcclusionEngine().estimate(np.zeros((64, 64, 3), np.uint8))
    assert not result.visible_mask.any()
    assert result.confidence == 0
    assert "preserve-original" in result.status


def test_paste_clipped_face_keeps_coordinate_geometry():
    target = np.zeros((20, 20, 3), np.uint8)
    source = np.zeros((20, 20, 3), np.uint8)
    source[:, 10:] = 255
    out = paste_face(
        target, source, FaceBox(-10, 0, 20, 20), color_match=False, visible_mask=np.ones((20, 20), np.float32)
    )
    assert np.all(out[:, :10] == 255)
    assert np.all(out[:, 10:] == 0)


@pytest.mark.parametrize("failure", ["raise", "nan", "shape"])
def test_invalid_inference_preserves_original(failure):
    class Broken(ColorOccluder):
        def visible(self, crop):
            if failure == "raise":
                raise RuntimeError("provider failed")
            if failure == "nan":
                return np.full(crop.shape[:2], np.nan)
            return np.ones((1, 1))

    result = OcclusionEngine(models=Broken()).estimate(np.zeros((64, 64, 3), np.uint8))
    assert not result.visible_mask.any()
    assert result.confidence == 0
    assert "mask inference failed" in result.status


def test_uncertain_track_blocks_swap_and_generated_face_visibility(monkeypatch):
    from deepfake import pipeline
    from deepfake.pipeline import FaceSwapPipeline, build_config

    class Detector:
        def detect(self, frame):
            return [FaceBox(16, 16, 32, 32, score=0.30)]

    monkeypatch.setattr(pipeline, "create_detector", lambda _kind: Detector())
    monkeypatch.setattr(pipeline, "OcclusionEngine", lambda *args, **kwargs: OcclusionEngine(models=ColorOccluder()))
    pipe = FaceSwapPipeline(build_config("realtime", backend="passthrough", allow_passthrough=True))

    def forbidden_swap(crop):
        pytest.fail("uncertain track should never invoke swap")

    pipe.swapper.swap = forbidden_swap
    original = np.full((64, 64, 3), 100, np.uint8)
    output, _ = pipe.swap_frame(original)
    assert np.array_equal(output, original)
    context = pipe.framegen_context(original)
    assert not context[1].any()
    assert context[2].any()
    assert "uncertain track" in pipe.stage_stats["occlusion_status"]
