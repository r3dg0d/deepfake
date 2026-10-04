"""CPU occlusion mask math. No GPU, no weights, no temporal filter."""

import numpy as np

from deepfake.composite import paste_face
from deepfake.detect import FaceBox
from deepfake.occlusion import ColorKeyOccluder, OcclusionEngine, combine_visible_mask
from deepfake.paths import models_dir

HAND_BGR = (30, 40, 220)  # synthetic hand swatch, not a skin tone
HAIR_BGR = (20, 20, 20)


def _frame_and_box(h=200, w=220):
    frame = np.full((h, w, 3), 80, dtype=np.uint8)
    box = FaceBox(30, 24, 160, 160)
    return frame, box


def test_hand_rectangle_excluded_from_visible_mask():
    frame, box = _frame_and_box()
    frame[90:130, 80:140] = HAND_BGR
    engine = OcclusionEngine(parser=ColorKeyOccluder(HAND_BGR))
    est = engine.estimate(frame, box, landmarks=None, face_mask=None, temporal_state=None)
    vis = est.visible_face_mask
    assert vis[90:130, 80:140].max() == 0.0
    # Interior of the ellipse, away from the swatch, stays paintable.
    assert vis[50:70, 100:120].mean() > 0.5
    assert est.occluder_mask[100, 100] == 1.0
    assert "unavailable" in est.detail


def test_hair_occluder_pixels_stay_out_of_swap_mask():
    frame, box = _frame_and_box()
    hair = (slice(36, 58), slice(70, 150))
    frame[hair] = HAIR_BGR
    original = frame.copy()
    swapped = np.full((box.h, box.w, 3), 240, dtype=np.uint8)
    engine = OcclusionEngine(parser=ColorKeyOccluder(HAIR_BGR, tolerance=0))
    est = engine.estimate(frame, box, None, None, None)
    assert est.visible_face_mask[hair].max() == 0.0
    assert est.face_mask[hair].max() > 0.0  # hair sat inside the ellipse
    out = paste_face(
        frame,
        swapped,
        box,
        feather=1,
        color_match=False,
        visible_mask=est.visible_face_mask,
    )
    assert np.array_equal(out[hair], original[hair])
    assert out[120, 110].mean() > original[120, 110].mean()


def test_full_occlusion_yields_empty_visible_mask():
    frame, box = _frame_and_box()
    frame[:, :] = HAND_BGR
    est = OcclusionEngine(parser=ColorKeyOccluder(HAND_BGR)).estimate(
        frame, box, None, None, temporal_state=None
    )
    assert est.visible_face_mask.sum() == 0.0
    swapped = np.full((box.h, box.w, 3), 255, dtype=np.uint8)
    out = paste_face(frame, swapped, box, feather=4, color_match=False, visible_mask=est.visible_face_mask)
    assert np.array_equal(out, frame)


def test_temporal_state_not_required_and_parser_unavailable():
    frame, box = _frame_and_box()
    engine = OcclusionEngine()
    est = engine.estimate(frame, box, landmarks=None, face_mask=None, temporal_state=None)
    assert est.temporal_state is None
    assert est.parser_available is False
    assert est.detail == "parser unavailable: no face-parsing weights installed"
    assert est.occluder_mask.sum() == 0.0
    assert est.confidence == 0.0
    # Returned visible mask is the current ellipse (no extra hole).
    assert np.allclose(est.visible_face_mask, est.face_mask)
    assert est.face_mask[110, 110] > 0.8
    state = {"frame": 3}
    again = engine.estimate(frame, box, None, None, state)
    assert again.temporal_state is state
    assert again.detail == est.detail  # probe is not repeated into a different answer


def test_unreadable_weight_stays_on_the_ellipse():
    # Isolated XDG from conftest. A junk file must not raise or download.
    dest = models_dir() / "vision" / "bisenet_resnet_18.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"not-an-onnx-model")
    frame, box = _frame_and_box()
    est = OcclusionEngine().estimate(frame, box, None, None, None)
    assert est.parser_available is False
    assert "parser unavailable" in est.detail
    assert "bisenet_resnet_18.onnx" in est.detail
    assert "no download" in est.detail
    assert np.allclose(est.visible_face_mask, est.face_mask)
    assert dest.is_file()


def test_paste_without_visible_mask_unchanged_by_ones_mask():
    frame = np.zeros((180, 200, 3), dtype=np.uint8)
    face = np.full((100, 100, 3), 200, dtype=np.uint8)
    box = FaceBox(40, 30, 100, 100)
    plain = paste_face(frame, face, box, feather=6, color_match=False)
    ones = paste_face(
        frame,
        face,
        box,
        feather=6,
        color_match=False,
        visible_mask=np.ones(frame.shape[:2], dtype=np.float32),
    )
    assert np.array_equal(plain, ones)


def test_combine_visible_mask_subtracts_occluder():
    face = np.ones((4, 4), dtype=np.float32)
    occ = np.zeros((4, 4), dtype=np.float32)
    occ[0, :] = 1
    vis = combine_visible_mask(face, occ)
    assert vis[0].sum() == 0
    assert vis[1:].min() == 1
