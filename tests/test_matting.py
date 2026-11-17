"""Rotoscoping / background replacement for the LivePortrait head engine.

The compositor logic is tested with a deterministic segmenter. The tests that need the real
PP-HumanSeg network run when its weights are installed (or ``DEEPFAKE_TEST_PPHUMANSEG`` points at
the .onnx file) and are skipped otherwise.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from deepfake.cli import _apply_config_defaults, _cfg_from_kwargs, main
from deepfake.detect import FaceBox
from deepfake.matting import (
    MODEL_FILE,
    BackgroundSpec,
    HeadMatte,
    MatteUnavailable,
    PersonSegmenter,
    TemporalMatte,
    guided_refine,
    model_path,
    parse_background,
    resolve_rotoscope,
    sharpen,
)

GREEN = (0, 255, 0)
RED_GENERATED = (0, 0, 255)


class ColorSegmenter:
    """Person = pixels that are clearly brighter than the dark test background."""

    def alpha(self, bgr):
        import cv2

        small = cv2.resize(bgr, (192, 192), interpolation=cv2.INTER_AREA)
        return (small.max(axis=2) > 100).astype(np.float32)


def camera_frame(head_x=110):
    """Dark textured room with a grey 'person': head box plus shoulders."""
    yy, xx = np.mgrid[0:240, 0:320]
    frame = np.dstack([(xx * 3 + yy) % 60, (yy * 2) % 60, (xx + yy * 5) % 60]).astype(np.uint8)
    frame[60:140, head_x:head_x + 80] = 200  # head
    frame[140:240, head_x - 30:head_x + 110] = 180  # shoulders
    return frame


def generated_crop():
    """Reference character (red disc) on a bright reference background that must NOT reach the output."""
    import cv2

    crop = np.full((256, 256, 3), 90, np.uint8)  # reference background: below the fake segmenter's threshold
    cv2.circle(crop, (128, 128), 70, RED_GENERATED, -1)
    return crop


FACE = FaceBox(110, 60, 80, 80)
PASTE = FaceBox(70, 20, 160, 160)


def test_parse_background(tmp_path):
    assert parse_background(None).kind == "keep" and not parse_background("keep").replaces
    assert parse_background("green").color == GREEN
    assert parse_background("#FF8000").color == (0, 128, 255)  # BGR
    assert parse_background("remove").color == (0, 0, 0)
    assert parse_background("blur").kind == "blur"
    import cv2

    image = tmp_path / "room.png"
    cv2.imwrite(str(image), np.zeros((8, 8, 3), np.uint8))
    spec = parse_background(str(image))
    assert spec.kind == "image" and spec.path == image and spec.describe() == str(image)
    with pytest.raises(ValueError, match="Unknown background"):
        parse_background("/nonexistent/room.png")


def test_resolve_rotoscope_follows_installed_weights():
    keep, green = BackgroundSpec(), parse_background("green")
    enabled, note = resolve_rotoscope(None, keep)  # auto, not installed
    assert enabled is False and "models install pphumanseg" in note
    assert resolve_rotoscope(False, keep) == (False, None)
    with pytest.raises(MatteUnavailable, match="models install pphumanseg"):
        resolve_rotoscope(True, keep)
    with pytest.raises(MatteUnavailable):  # asked to remove the background: never silently show it
        resolve_rotoscope(None, green)
    with pytest.raises(ValueError, match="needs the matte"):
        resolve_rotoscope(False, green)
    model_path().parent.mkdir(parents=True)
    model_path().write_bytes(b"\x08")
    assert resolve_rotoscope(None, keep) == (True, None)
    assert resolve_rotoscope(None, green) == (True, None)


def test_matte_shaping():
    alpha = np.linspace(0, 1, 11, dtype=np.float32)[None, :].repeat(4, axis=0)
    shaped = sharpen(alpha)
    assert shaped[0, 0] == 0 and shaped[0, -1] == 1 and shaped[0, 1] == 0 and np.all(np.diff(shaped[0]) >= 0)
    # the guided filter snaps a blurry matte to the image edge
    guide = np.zeros((64, 64, 3), np.uint8)
    guide[:, 32:] = 255
    blurry = np.clip((np.arange(64, dtype=np.float32) - 16) / 32, 0, 1)[None, :].repeat(64, axis=0)
    refined = guided_refine(blurry, guide, radius=6)
    step, before = float(refined[32, 32] - refined[32, 31]), float(blurry[32, 32] - blurry[32, 31])
    assert before < 0.05 and step > 0.15  # the matte now jumps where the image does


def test_temporal_matte_damps_flicker_but_follows_motion():
    smooth = TemporalMatte(keep=0.5, motion=0.35)
    a = np.full((4, 4), 0.5, np.float32)
    assert smooth(a) is a
    flicker = smooth(np.full((4, 4), 0.6, np.float32))
    assert 0.5 < float(flicker[0, 0]) < 0.6  # small change: averaged
    jump = smooth(np.ones((4, 4), np.float32))
    assert float(jump[0, 0]) == 1.0  # large change: followed immediately, no trail


def test_rotoscope_keeps_camera_background_and_drops_reference_background():
    matte = HeadMatte(segmenter_factory=ColorSegmenter)
    frame = camera_frame()
    matte.begin(frame, frame[PASTE.y:PASTE.y + PASTE.h, PASTE.x:PASTE.x + PASTE.w])
    out = matte.composite(frame, generated_crop(), PASTE, FACE)
    matte.close()
    assert out is not frame and out.shape == frame.shape
    # centre of the head: the generated character
    np.testing.assert_allclose(out[100, 150], RED_GENERATED, atol=12)
    # a corner of the paste box: reference background (90) is not pasted, the room texture survives
    np.testing.assert_array_equal(out[30, 80], frame[30, 80])
    # outside the paste box nothing changes
    np.testing.assert_array_equal(out[:, :60], frame[:, :60])
    np.testing.assert_array_equal(out[200:], frame[200:])
    assert matte.matte_ms is not None and matte.matte_ms >= 0


def test_clean_plate_hides_the_real_head_where_the_character_is_smaller():
    """The real head (80 px wide) sticks out behind a narrower generated head once the plate was seen."""
    import cv2

    narrow = np.full((256, 256, 3), 90, np.uint8)
    cv2.rectangle(narrow, (110, 60), (146, 200), RED_GENERATED, -1)  # ~22 px wide in the frame
    room = camera_frame(head_x=400)  # person outside the view: the plate learns the empty room
    matte = HeadMatte(segmenter_factory=ColorSegmenter)
    crop = lambda f: f[PASTE.y:PASTE.y + PASTE.h, PASTE.x:PASTE.x + PASTE.w]  # noqa: E731
    for _ in range(3):
        matte.begin(room, crop(room))
        matte.composite(room, narrow, PASTE, FACE)
    frame = camera_frame()
    matte.begin(frame, crop(frame))
    out = matte.composite(frame, narrow, PASTE, FACE)
    matte.close()
    # real head pixel left of the narrow character: replaced by the learned room, not left grey (200)
    assert frame[100, 118, 0] == 200
    np.testing.assert_allclose(out[100, 118], room[100, 118], atol=10)
    # below the chin the real shoulders stay
    np.testing.assert_array_equal(out[170, 118], frame[170, 118])


@pytest.mark.parametrize("scope", ["person", "head"])
def test_background_replacement_scopes(scope):
    matte = HeadMatte(parse_background("green"), scope, segmenter_factory=ColorSegmenter)
    frame = camera_frame()
    matte.begin(frame, None)
    out = matte.composite(frame, generated_crop(), PASTE, FACE)
    np.testing.assert_array_equal(out[5, 5], GREEN)  # room replaced
    np.testing.assert_allclose(out[100, 150], RED_GENERATED, atol=12)  # character kept
    np.testing.assert_array_equal(out[70, 112], GREEN)  # real head corner behind the disc: gone in both scopes
    if scope == "person":
        np.testing.assert_array_equal(out[220, 150], frame[220, 150])  # real shoulders kept
    else:
        np.testing.assert_array_equal(out[220, 150], GREEN)  # only the generated head remains
    # a frame without a face still gets its background treated
    matte.begin(frame, None)
    lost = matte.passthrough(frame)
    matte.close()
    np.testing.assert_array_equal(lost[5, 5], GREEN)
    assert (lost[220, 150] == frame[220, 150]).all() == (scope == "person")


def test_background_image_and_blur(tmp_path):
    import cv2

    image = tmp_path / "bg.png"
    cv2.imwrite(str(image), np.full((30, 90, 3), (10, 200, 30), np.uint8))  # other aspect ratio: cover-cropped
    frame = camera_frame()
    matte = HeadMatte(parse_background(str(image)), segmenter_factory=ColorSegmenter)
    matte.begin(frame, None)
    out = matte.composite(frame, generated_crop(), PASTE, FACE)
    matte.close()
    np.testing.assert_array_equal(out[5, 5], (10, 200, 30))
    blur = HeadMatte(parse_background("blur"), segmenter_factory=ColorSegmenter)
    blur.begin(frame, None)
    out = blur.composite(frame, generated_crop(), PASTE, FACE)
    blur.close()
    patch, original = out[:40, :40].astype(float), frame[:40, :40].astype(float)
    assert patch.std() < original.std() / 2  # texture smoothed away
    with pytest.raises(ValueError, match="scope"):
        HeadMatte(scope="body", segmenter_factory=ColorSegmenter)


def test_paste_box_clipped_at_frame_border():
    matte = HeadMatte(parse_background("black"), segmenter_factory=ColorSegmenter)
    frame = camera_frame()
    matte.begin(frame, None)
    out = matte.composite(frame, generated_crop(), FaceBox(250, -40, 160, 160), FaceBox(280, 0, 60, 60))
    matte.close()
    assert out.shape == frame.shape


def test_engine_paste_uses_matte_and_reports_metrics(monkeypatch):
    """The engine's compositing path without the neural weights (they need CUDA and a checkout)."""
    import deepfake.matting as matting
    from deepfake.engines.experimental.liveportrait import LivePortraitEngine
    from deepfake.pipeline import build_config

    class Detector:
        def detect(self, frame):
            return [FACE]

    monkeypatch.setattr("deepfake.engines.experimental.liveportrait.create_detector", lambda *a, **k: Detector())
    monkeypatch.setattr(matting, "PersonSegmenter", ColorSegmenter)
    monkeypatch.setattr(matting.HeadMatte.__init__, "__kwdefaults__", {"segmenter_factory": ColorSegmenter})
    monkeypatch.setattr(matting, "model_installed", lambda: True)

    engine = LivePortraitEngine()
    engine._init_runtime(build_config(engine="experimental", mode="head", model_backend="liveportrait",
                                      background="green"))
    assert engine.matte is not None
    frame = camera_frame()
    engine.face_box = FACE
    engine.matte.begin(frame, None)
    out = engine._paste(frame, generated_crop(), PASTE)
    np.testing.assert_array_equal(out[5, 5], GREEN)
    metrics = engine.live_metrics()
    assert metrics["rotoscope"] is True and metrics["background"] == "#00ff00" and metrics["matte_ms"] is not None
    engine.unload()

    plain = LivePortraitEngine()
    plain._init_runtime(build_config(engine="experimental", mode="head", model_backend="liveportrait", rotoscope=False))
    assert plain.matte is None and plain.live_metrics() == {"rotoscope": False, "matte_ms": None, "background": "keep"}
    boxed = plain._paste(frame, generated_crop(), PASTE)
    np.testing.assert_allclose(boxed[100, 150], RED_GENERATED, atol=3)
    assert abs(int(boxed[40, 90, 0]) - 90) < 12  # legacy rectangle: the reference background is pasted too


def test_cli_matte_flags():
    for command in ("webcam", "video", "virtualcam"):
        help_text = CliRunner().invoke(main, [command, "--help"]).output
        for flag in ("--rotoscope", "--background", "--matte-scope"):
            assert flag in help_text
    head = {"engine": "experimental", "mode": "head"}
    cfg = _cfg_from_kwargs(_apply_config_defaults({**head, "background": "#102030", "rotoscope": True}))
    assert (cfg.rotoscope, cfg.background, cfg.matte_scope) == (True, "#102030", "person")
    assert _cfg_from_kwargs(_apply_config_defaults(dict(head))).rotoscope is None  # auto
    import click

    for bad in ({"background": "green"},  # AlphaFace has no head crop to matte
                {**head, "background": "green", "rotoscope": False},
                {**head, "matte_scope": "head"},
                {"engine": "experimental", "mode": "fullbody", "model_backend": "editalive", "rotoscope": True}):
        with pytest.raises(click.UsageError):
            _cfg_from_kwargs(_apply_config_defaults(dict(bad)))
    with pytest.raises(click.BadParameter):
        _cfg_from_kwargs(_apply_config_defaults({**head, "background": "plaid"}))


def _real_weights() -> Path | None:
    env = os.environ.get("DEEPFAKE_TEST_PPHUMANSEG")
    for candidate in (Path(env) if env else None, Path.home() / ".cache/deepfake/models/pphumanseg" / MODEL_FILE):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


@pytest.mark.skipif(_real_weights() is None, reason="PP-HumanSeg weights not installed")
def test_real_segmenter_is_registered_to_the_image():
    """Guards the OpenCV 5 graph-engine regression: the matte must sit on the person and mirror with the image."""
    from importlib.resources import files

    import cv2

    img = cv2.imdecode(np.frombuffer((files("deepfake") / "assets" / "bench_face.jpg").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    seg = PersonSegmenter(_real_weights())
    alpha = seg.alpha(img)
    assert alpha.shape == (192, 192) and alpha.dtype == np.float32 and 0 <= alpha.min() and alpha.max() <= 1
    mirrored = seg.alpha(np.ascontiguousarray(img[:, ::-1]))[:, ::-1]
    assert float(np.abs(alpha - mirrored).mean()) < 0.05
    assert alpha[48:96, 72:120].mean() > 0.9 and alpha[:24, :24].mean() < 0.2
    # end to end with the real network: the portrait's studio background does not reach the frame
    matte = HeadMatte(parse_background("green"), "head", segmenter_factory=lambda: PersonSegmenter(_real_weights()))
    frame = camera_frame()
    matte.begin(frame, None)
    out = matte.composite(frame, cv2.resize(img[40:424], (256, 256)), PASTE, FACE)
    matte.close()
    np.testing.assert_array_equal(out[5, 5], GREEN)
    assert not np.array_equal(out[100, 150], GREEN)
