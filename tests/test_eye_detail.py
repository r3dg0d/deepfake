import cv2
import numpy as np

from deepfake.composite import enhance_eye_detail, paste_face
from deepfake.detect import FaceBox
from deepfake.occlusion import OcclusionEngine


def test_generated_eye_detail_improves_edges_without_overshoot():
    face = np.full((64, 64, 3), 150, np.uint8)
    cv2.ellipse(face, (32, 32), (14, 6), 0, 0, 360, (50, 50, 50), -1)
    face = cv2.GaussianBlur(face, (0, 0), 1.1)
    mask = np.zeros((64, 64), np.float32)
    mask[20:44, 12:52] = 1
    sharpened = enhance_eye_detail(face, mask)
    before = np.abs(np.diff(face[32, :, 0].astype(float))).max()
    after = np.abs(np.diff(sharpened[32, :, 0].astype(float))).max()
    assert after > before
    assert sharpened.min() >= face.min() and sharpened.max() <= face.max()
    assert np.array_equal(sharpened[:12], face[:12])


def test_flat_or_noisy_eye_and_missing_mask_are_unchanged():
    rng = np.random.default_rng(2)
    gray = rng.integers(149, 152, size=(32, 32), dtype=np.uint8)
    image = np.repeat(gray[..., None], 3, axis=2)
    assert np.array_equal(enhance_eye_detail(image, np.ones((32, 32), np.float32)), image)
    assert enhance_eye_detail(image, None) is image


def test_eye_detail_never_overwrites_occluded_pixels_after_upscaling():
    target = np.full((96, 96, 3), (30, 100, 200), np.uint8)
    donor = np.full((64, 64, 3), 180, np.uint8)
    donor[20:40, 20:40] = 40
    visible = np.ones((64, 64), np.float32)
    visible[20:40, 20:40] = 0
    output = paste_face(
        target,
        donor,
        FaceBox(0, 0, 96, 96),
        color_match=False,
        visible_mask=visible,
        eye_mask=np.ones((64, 64), np.float32),
    )
    hidden = cv2.resize(visible, (96, 96), interpolation=cv2.INTER_NEAREST) == 0
    assert np.array_equal(output[hidden], target[hidden])


def test_eye_regions_are_current_and_intersect_visible_alpha():
    class Regions:
        def parse_regions(self, frame):
            face = np.ones(frame.shape[:2], np.float32)
            return face, 0.95, np.zeros_like(face), face

        def visible(self, frame):
            visible = np.ones(frame.shape[:2], np.float32)
            visible[20:40, 20:40] = 0
            return visible

    engine = OcclusionEngine(models=Regions())
    result = engine.estimate(np.full((64, 64, 3), 100, np.uint8))
    assert not result.eye_mask[20:40, 20:40].any()
    assert np.array_equal(result.eye_mask, result.visible_mask)
