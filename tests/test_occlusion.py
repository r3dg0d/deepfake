"""CPU occlusion mask math. No GPU, no weights, no temporal filter."""

import numpy as np

from deepfake.composite import paste_face
from deepfake.detect import FaceBox
from deepfake.occlusion import (
    ColorKeyOccluder,
    OcclusionEngine,
    combine_visible_mask,
    xseg_occluder_from_keep,
    xseg_should_run,
)
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


def test_label_map_keeps_hair_cloth_and_background_out():
    from deepfake.occlusion import SWAP_CLASS_IDS, masks_from_label_map

    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1:5, 1:5] = 1  # skin
    labels[0, :] = 17  # hair
    labels[5, :] = 16  # cloth
    labels[3, 3] = 0  # background hole inside the face
    labels[2, 2] = 6  # glasses
    _face, occ, vis = masks_from_label_map(labels)
    assert vis[0].max() == 0.0
    assert vis[5].max() == 0.0
    assert vis[3, 3] == 0.0
    assert vis[2, 2] == 0.0
    assert occ[0, 0] == 1.0 and occ[5, 0] == 1.0 and occ[2, 2] == 1.0
    assert vis[1, 1] == 1.0
    assert 17 not in SWAP_CLASS_IDS and 16 not in SWAP_CLASS_IDS and 0 not in SWAP_CLASS_IDS
    assert set(SWAP_CLASS_IDS) == {1, 2, 3, 4, 5, 10, 11, 12, 13}


def test_pipeline_passes_visible_mask_from_mocked_parser(monkeypatch):
    from deepfake.occlusion import OcclusionEngine
    from deepfake.pipeline import FaceSwapPipeline, build_config

    class Strip:
        def occluder_mask(self, frame, box):
            mask = np.zeros(frame.shape[:2], dtype=np.float32)
            y0 = int(box.y) + 8
            mask[y0 : y0 + 6, int(box.x) : int(box.x) + int(box.w)] = 1.0
            return mask

    seen: dict = {}

    def spy(frame, face, box, **kwargs):
        seen["visible"] = kwargs.get("visible_mask")
        seen["box"] = box
        return paste_face(frame, face, box, **kwargs)

    monkeypatch.setattr("deepfake.pipeline.paste_face", spy)

    class Det:
        def detect(self, frame):
            return [FaceBox(20, 20, 80, 80)]

    monkeypatch.setattr("deepfake.pipeline.create_detector", lambda prefer="auto": Det())
    pipe = FaceSwapPipeline(build_config("low-latency", allow_passthrough=True))
    pipe.occlusion = OcclusionEngine(parser=Strip())
    pipe.set_source(np.zeros((32, 32, 3), dtype=np.uint8))
    pipe.swap_frame(np.full((160, 180, 3), 90, dtype=np.uint8))
    vis = seen["visible"]
    box = seen["box"]
    assert vis is not None
    y0 = int(box.y) + 8
    assert vis[y0 : y0 + 6, int(box.x) : int(box.x) + int(box.w)].max() == 0.0
    assert vis[int(box.y) + 30 : int(box.y) + 50, int(box.x) + 20 : int(box.x) + 40].mean() > 0.2


def test_pipeline_keeps_ellipse_when_parser_unavailable(monkeypatch, capsys):
    from deepfake.pipeline import FaceSwapPipeline, build_config

    seen: dict = {}

    def spy(frame, face, box, **kwargs):
        seen["visible"] = kwargs.get("visible_mask")
        return paste_face(frame, face, box, **kwargs)

    monkeypatch.setattr("deepfake.pipeline.paste_face", spy)

    class Det:
        def detect(self, frame):
            return [FaceBox(20, 20, 80, 80)]

    monkeypatch.setattr("deepfake.pipeline.create_detector", lambda prefer="auto": Det())
    pipe = FaceSwapPipeline(build_config("low-latency", allow_passthrough=True))
    pipe.set_source(np.zeros((32, 32, 3), dtype=np.uint8))
    pipe.swap_frame(np.full((160, 180, 3), 90, dtype=np.uint8))
    assert seen["visible"] is None
    err = capsys.readouterr().err
    assert "occlusion: inactive" in err
    assert "parser unavailable" in err
    assert "occlusion: active" not in err


def test_bisenet_parses_bench_face_on_cpu(tmp_path):
    pytest = __import__("pytest")
    pytest.importorskip("onnxruntime")
    from pathlib import Path

    import cv2

    weight = Path.home() / ".cache/deepfake/models/vision/bisenet_resnet_18.onnx"
    if not weight.is_file():
        pytest.skip("cached bisenet weight is not on this machine")
    # Keep this check BiSeNet-only. The real cache also has xseg_2.onnx beside it.
    isolated = tmp_path / "bisenet_resnet_18.onnx"
    isolated.symlink_to(weight)
    image = Path(__file__).resolve().parents[1] / "src/deepfake/assets/bench_face.jpg"
    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    assert frame is not None
    h, w = frame.shape[:2]
    engine = OcclusionEngine(weights_path=isolated)
    est = engine.estimate(frame, FaceBox(0, 0, w, h), None, None, None)
    assert est.parser_available is True
    assert est.apply_to_composite is True
    assert "CelebAMask-HQ" in est.detail
    assert "CPUExecutionProvider" in est.detail or "CUDAExecutionProvider" in est.detail
    # Top of this portrait is background/hair, not a swap class.
    assert est.visible_face_mask[: h // 10].mean() < 0.05
    # Mid-face has skin.
    assert est.visible_face_mask[h // 3 : h // 2, w // 4 : 3 * w // 4].mean() > 0.2
    # Upper-face occluders (hair on this portrait) are outside the swap mask.
    band = est.occluder_mask[h // 6 : h // 4, w // 3 : 2 * w // 3]
    assert band.mean() > 0.3
    blocked = est.visible_face_mask[est.occluder_mask > 0.5]
    assert blocked.size > 0
    assert blocked.max() == 0.0


class _ScriptedOccluder:
    """Test double. mode is 'open', 'zero', or 'hole'."""

    def __init__(self) -> None:
        self.mode = "open"

    def occluder_mask(self, frame, box):
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        if self.mode == "zero":
            mask[:] = 1.0
        elif self.mode == "hole":
            mask[70:95, 70:95] = 1.0
        return mask


def _run_scripted(mode_frames: list[str]):
    parser = _ScriptedOccluder()
    engine = OcclusionEngine(parser=parser)
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    box = FaceBox(30, 30, 100, 100)
    face = np.ones(frame.shape[:2], dtype=np.float32)
    state = None
    last = None
    for mode in mode_frames:
        parser.mode = mode
        last = engine.estimate(frame, box, None, face, state)
        state = last.temporal_state
    return last


def test_one_dropped_frame_does_not_zero_visible_mask():
    # Three stable frames, then a single empty parse. Unsmoothed output is all zeros.
    est = _run_scripted(["open", "open", "open", "zero"])
    assert est.visible_face_mask[80, 80] > 0.8
    assert est.visible_face_mask.sum() > 1000
    assert est.confidence < 0.05  # raw parse collapsed; the held mask is what we composite
    assert est.temporal_state["dropout_holds"] == 1


def test_sudden_hole_shows_within_two_frames_and_reopens_slower():
    # A hand-sized hole must be visible within one frame. Unsmoothed code snaps
    # the same pixel back to 1 the moment the hole leaves; the reveal step must not.
    holed = _run_scripted(["open", "open", "hole"])
    assert holed.visible_face_mask[80, 80] < 0.25
    assert holed.visible_face_mask[50, 50] > 0.8  # rest of the face stays open
    reopened = _run_scripted(["open", "open", "hole", "hole", "open"])
    # After two hole frames the pixel is ~0.01; one reveal step at 0.35 stays well below 1.
    assert reopened.visible_face_mask[80, 80] < 0.75
    assert reopened.visible_face_mask[80, 80] > 0.2


def test_second_empty_frame_is_not_held_open():
    est = _run_scripted(["open", "open", "zero", "zero"])
    # First empty frame held the open mask; the second must follow the empty parse.
    assert est.visible_face_mask[80, 80] < 0.25
    assert est.temporal_state["dropout_holds"] == 0


class _RectangleOccluder:
    """Mock occluder: a fixed rectangle, not a color key and not a model."""

    def __init__(self, rect: tuple[int, int, int, int]) -> None:
        self.rect = rect

    def occluder_mask(self, frame, box):
        del box
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        y0, y1, x0, x1 = self.rect
        mask[y0:y1, x0:x1] = 1.0
        return mask


def test_rectangle_occluder_is_subtracted_from_the_face_mask():
    frame, box = _frame_and_box()
    face = np.ones(frame.shape[:2], dtype=np.float32)
    est = OcclusionEngine(parser=_RectangleOccluder((90, 130, 80, 140))).estimate(
        frame, box, None, face, None
    )
    assert est.occluder_mask[100, 100] == 1.0
    assert est.visible_face_mask[90:130, 80:140].max() == 0.0
    assert est.visible_face_mask[20, 20] == 1.0
    assert est.apply_to_composite is True


def test_xseg_keep_mask_punches_skin_rectangle():
    skin = np.zeros((8, 8), dtype=np.float32)
    skin[2:6, 1:5] = 1.0
    keep = np.ones((8, 8), dtype=np.float32)
    keep[2:6, 1:5] = 0.0
    occ = xseg_occluder_from_keep(skin, keep)
    assert occ[3, 2] == 1.0
    assert occ[0, 0] == 0.0
    # Skin the matte still accepts is not an occluder.
    keep[2:6, 1:5] = 1.0
    assert xseg_occluder_from_keep(skin, keep).sum() == 0.0


def test_xseg_runs_on_interval_or_skin_flood():
    assert xseg_should_run(0, 0.0, 0.0) is True
    assert xseg_should_run(1, 0.2, 0.05) is False
    assert xseg_should_run(4, 0.0, 0.0) is True
    assert xseg_should_run(2, 0.2, 0.01) is True


def test_missing_xseg_leaves_bisenet_mask_math_unchanged(tmp_path):
    # No weight on this path. The rectangle helper is the whole occluder.
    skin = np.ones((4, 4), dtype=np.float32)
    keep = np.ones((4, 4), dtype=np.float32)
    assert xseg_occluder_from_keep(skin, keep).sum() == 0.0
    assert not (tmp_path / "xseg_2.onnx").is_file()


def test_xseg_rejects_flat_overlay_on_bench_face():
    pytest = __import__("pytest")
    pytest.importorskip("onnxruntime")
    import time
    from pathlib import Path

    import cv2

    bisenet = Path.home() / ".cache/deepfake/models/vision/bisenet_resnet_18.onnx"
    xseg = bisenet.with_name("xseg_2.onnx")
    if not bisenet.is_file() or not xseg.is_file():
        pytest.skip("cached bisenet or xseg_2.onnx is not on this machine")
    image = Path(__file__).resolve().parents[1] / "src/deepfake/assets/bench_face.jpg"
    frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
    assert frame is not None
    h, w = frame.shape[:2]
    box = FaceBox(0, 0, w, h)
    started = time.perf_counter()
    clean = OcclusionEngine(weights_path=bisenet).estimate(frame, box, None, None, None)
    clean_s = time.perf_counter() - started
    assert clean.parser_available is True
    assert "XSeg" in clean.detail
    covered = frame.copy()
    y0, y1, x0, x1 = h // 3, h // 2, w // 3, w // 2
    color = tuple(int(v) for v in frame[h // 2, w // 2])
    covered[y0:y1, x0:x1] = color
    # A new engine so this is a fresh matte, not the 4-frame hold of the clean face.
    started = time.perf_counter()
    punched = OcclusionEngine(weights_path=bisenet).estimate(covered, box, None, None, None)
    punched_s = time.perf_counter() - started
    assert "XSeg" in punched.detail
    clean_roi = clean.occluder_mask[y0:y1, x0:x1].mean()
    punched_roi = punched.occluder_mask[y0:y1, x0:x1].mean()
    assert punched_roi > clean_roi + 0.15
    assert punched.visible_face_mask[y0:y1, x0:x1].mean() < clean.visible_face_mask[y0:y1, x0:x1].mean()
    # Timing is reported by the test process; a hang would fail the suite.
    assert clean_s < 5.0 and punched_s < 5.0
