"""Geometry transforms: crop <-> frame round trips, proportions, masks, aspect, smoothing."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from deepfake.detect import FaceBox  # noqa: E402
from deepfake.geometry import (  # noqa: E402
    OneEuroFilter,
    align_face,
    anisotropy,
    estimate_similarity,
    eye_metrics,
    fit_frame,
    invert_affine,
    landmark_mask,
    paste_aligned,
    similarity_params,
    template_points,
    transform_points,
)
from deepfake.pipeline import FaceSwapPipeline, build_config  # noqa: E402


def _landmarks(cx=300.0, cy=220.0, eye=80.0, roll_deg=0.0):
    base = (template_points("ffhq", 1) - [0.5, 0.5]) * (eye / 0.25)
    a = np.radians(roll_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return base @ rot.T + [cx, cy]


def _frame(h=480, w=640, seed=0):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


@pytest.mark.parametrize("roll", [-35.0, 0.0, 20.0])
@pytest.mark.parametrize("template", ["ffhq", "arcface"])
def test_similarity_recovers_known_transform(roll, template):
    lm = _landmarks(roll_deg=roll)
    m = estimate_similarity(lm, template_points(template, 256))
    assert np.allclose(transform_points(m, lm)[:2], template_points(template, 256)[:2], atol=6.0)
    scale, angle = similarity_params(m)
    assert angle == pytest.approx(-roll, abs=0.5)
    assert anisotropy(m) == pytest.approx(1.0, abs=1e-9)  # uniform scale: proportions cannot change
    assert np.allclose(transform_points(invert_affine(m), transform_points(m, lm)), lm, atol=1e-6)


def test_alignment_removes_roll_and_fixes_scale():
    for roll, eye in [(-25.0, 60.0), (0.0, 110.0), (30.0, 85.0)]:
        lm = _landmarks(eye=eye, roll_deg=roll)
        aligned = align_face(_frame(), lm, size=256, template="ffhq")
        in_crop = eye_metrics(transform_points(aligned.matrix, lm))
        assert abs(in_crop["roll_deg"]) < 0.5
        assert in_crop["eye_distance"] / 256 == pytest.approx(0.25, abs=0.01)


def test_crop_then_paste_is_the_identity_inside_the_mask():
    frame = _frame()
    aligned = align_face(frame, _landmarks(roll_deg=17.0), size=256)
    out = paste_aligned(frame, aligned.crop, aligned, color_match=False)
    assert out.shape == frame.shape
    assert np.abs(out.astype(int) - frame.astype(int)).mean() < 2.0  # resampling only, no shift


def test_paste_lands_on_the_face_not_the_crop_box():
    frame = np.full((480, 640, 3), 40, np.uint8)
    lm = _landmarks(roll_deg=-20.0)
    aligned = align_face(frame, lm, size=256)
    out = paste_aligned(frame, np.full((256, 256, 3), 240, np.uint8), aligned, color_match=False)
    changed = np.abs(out.astype(int) - frame.astype(int)).max(axis=2) > 8
    ys, xs = np.nonzero(changed)
    centre = np.array([xs.mean(), ys.mean()])
    expected = (lm[0] + lm[1]) / 2 + ((lm[3] + lm[4]) / 2 - (lm[0] + lm[1]) / 2) * 0.45
    assert np.linalg.norm(centre - expected) < 4.0
    # the changed area stays well inside the warped crop square (no rectangular edge)
    corners = transform_points(aligned.inverse, np.array([[0, 0], [256, 0], [256, 256], [0, 256]], float))
    crop_area = cv2.contourArea(corners.astype(np.float32))
    assert changed.sum() < 0.55 * crop_area


def test_face_partly_outside_the_frame_keeps_proportions():
    frame = _frame()
    lm = _landmarks(cx=20.0, cy=30.0, eye=120.0)  # most of the face is off the top-left corner
    aligned = align_face(frame, lm, size=256)
    assert anisotropy(aligned.matrix) == pytest.approx(1.0, abs=1e-9)
    out = paste_aligned(frame, aligned.crop, aligned, color_match=False)
    assert out.shape == frame.shape


def test_landmark_mask_is_zero_at_the_crop_border_and_covers_the_features():
    for template in ("ffhq", "arcface"):
        mask = landmark_mask(256, template, feather=24)
        assert mask.max() == pytest.approx(1.0)
        assert mask[0].max() == 0 and mask[-1].max() == 0 and mask[:, 0].max() == 0 and mask[:, -1].max() == 0
        for x, y in template_points(template, 256):
            assert mask[int(y), int(x)] > 0.6


def test_fit_frame_cover_and_contain_preserve_proportions_stretch_does_not():
    frame = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(frame, (320, 240), 100, (255, 255, 255), -1)

    def aspect(img):
        ys, xs = np.nonzero(img[:, :, 0] > 127)
        return (xs.max() - xs.min()) / (ys.max() - ys.min())

    assert aspect(fit_frame(frame, (960, 540), "cover")) == pytest.approx(1.0, abs=0.02)
    assert aspect(fit_frame(frame, (960, 540), "contain")) == pytest.approx(1.0, abs=0.02)
    assert aspect(fit_frame(frame, (960, 540), "stretch")) == pytest.approx(4 / 3, abs=0.03)
    for mode in ("cover", "contain", "stretch"):
        assert fit_frame(frame, (960, 540), mode).shape == (540, 960, 3)
    assert fit_frame(frame, (640, 480), "cover") is frame
    with pytest.raises(ValueError):
        fit_frame(frame, (10, 10), "squash")


def test_one_euro_filter_reduces_jitter_and_follows_motion():
    rng = np.random.default_rng(3)
    base = _landmarks()
    flt = OneEuroFilter()
    still = [flt(base + rng.normal(0, 1.0, base.shape), 1 / 24, scale=80.0) for _ in range(60)]
    assert np.stack(still[10:]).std(axis=0).mean() < 0.7  # input jitter is 1.0 px
    flt = OneEuroFilter()
    last = None
    for i in range(48):
        truth = base + [i * 12.0, 0.0]  # 3.6 eye distances per second
        last = flt(truth, 1 / 24, scale=80.0)
    assert np.abs(last - truth).max() < 4.0


class _Detector:
    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, frame):
        return self.boxes


def _pipeline(monkeypatch, boxes, **settings):
    import deepfake.pipeline as pipeline

    monkeypatch.setattr(pipeline, "create_detector", lambda *a, **k: _Detector(boxes))
    return FaceSwapPipeline(build_config("high-quality", backend="passthrough", allow_passthrough=True,
                                         watermark=False, color_match=False, **settings))


def test_pipeline_landmark_path_keeps_the_frame_and_legacy_path_is_still_reachable(monkeypatch):
    frame = _frame()
    lm = _landmarks(roll_deg=12.0)
    box = FaceBox(200, 120, 200, 240, landmarks=lm.astype(np.float32))
    seen = {}
    for name, settings in {"landmark": {}, "bbox": {"alignment": "bbox", "mask_mode": "oval"}}.items():
        pipe = _pipeline(monkeypatch, [box], **settings)
        crops = []
        swap = pipe.swapper.swap
        pipe.swapper.swap = lambda crop, swap=swap, crops=crops: (crops.append(crop.copy()), swap(crop))[1]
        out, _ = pipe.swap_frame(frame.copy())
        assert out.shape == frame.shape
        assert np.abs(out.astype(int) - frame.astype(int)).mean() < 3.0  # passthrough: output is the input
        seen[name] = crops[0]
    assert seen["landmark"].shape == seen["bbox"].shape == (256, 256, 3)
    assert np.abs(seen["landmark"].astype(int) - seen["bbox"].astype(int)).mean() > 5.0  # different crops


def test_pipeline_falls_back_to_the_box_path_without_landmarks(monkeypatch):
    pipe = _pipeline(monkeypatch, [FaceBox(200, 120, 200, 240)])
    out, _ = pipe.swap_frame(_frame())
    assert out.shape == (480, 640, 3)


def test_geometry_settings_are_recorded_in_provenance(monkeypatch, tmp_path):
    from deepfake.provenance import build_provenance, write_sidecar

    cfg = build_config("balanced", backend="passthrough", alignment="bbox", fit="stretch")
    record = build_provenance(cfg)
    assert record["geometry"]["alignment"] == "bbox" and record["geometry"]["fit"] == "stretch"
    assert record["backend"] == "passthrough" and "version" in record
    out = tmp_path / "clip.mp4"
    out.write_bytes(b"x")
    side = write_sidecar(out, cfg)
    assert side.name == "clip.mp4.deepfake.json" and '"alignment": "bbox"' in side.read_text()


def test_compare_variant_parsing():
    from deepfake.geometry_report import CURRENT, LEGACY, parse_variant

    assert parse_variant("legacy") == ("legacy", LEGACY)
    name, settings = parse_variant("mine:align_template=arcface,landmark_smooth=off")
    assert name == "mine" and settings["align_template"] == "arcface" and settings["landmark_smooth"] is False
    assert settings["alignment"] == CURRENT["alignment"]
    with pytest.raises(ValueError):
        parse_variant("nope")
    with pytest.raises(ValueError):
        parse_variant("mine:bogus=1")


def test_desktop_preferences_carry_geometry_flags():
    from deepfake.desktop import DEFAULT_PREFERENCES, build_session_command, validate_preferences

    args = build_session_command(dict(DEFAULT_PREFERENCES))
    assert args[args.index("--align") + 1] == "landmark" and "--landmark-smooth" in args
    legacy = {**DEFAULT_PREFERENCES, "alignment": "bbox", "mask_mode": "oval", "fit": "stretch", "landmark_smooth": False}
    args = build_session_command(legacy)
    assert args[args.index("--fit") + 1] == "stretch" and "--no-landmark-smooth" in args
    with pytest.raises(ValueError):
        validate_preferences({"alignment": "sideways"})


@pytest.mark.parametrize("error", [OSError("missing EGL"), ImportError("missing dependency")])
def test_optional_landmarker_load_failure_falls_back(monkeypatch, error):
    from deepfake import detect

    detector = object()
    monkeypatch.setattr(detect, "_create_box_detector", lambda prefer: detector)

    def unavailable():
        raise error

    monkeypatch.setattr(detect, "MediaPipeLandmarker", unavailable)
    assert detect.create_detector(landmarker="auto") is detector
    with pytest.raises(type(error)):
        detect.create_detector(landmarker="mediapipe")
