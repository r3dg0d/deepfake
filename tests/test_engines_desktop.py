import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from deepfake.cli import _apply_config_defaults, _cfg_from_kwargs, main
from deepfake.desktop import DEFAULT_PREFERENCES, build_session_command, request, validate_preferences
from deepfake.engines import EngineUnavailable, create_engine
from deepfake.engines.experimental.adapters import worker_command
from deepfake.experimental_models import manifests, validate_manifest, verify
from deepfake.ipc import SessionBus, read_json, send_control, session_path, state_dir, write_json_atomic
from deepfake.paths import models_dir
from deepfake.pipeline import build_config


def test_default_engine_and_cli_options():
    assert _cfg_from_kwargs(_apply_config_defaults({})).engine == "alphaface"
    for command in ("webcam", "video", "virtualcam"):
        result = CliRunner().invoke(main, [command, "--help"])
        assert result.exit_code == 0
        for flag in ("--engine", "--mode", "--model-backend", "--reference", "--frame-gen", "--backend"):
            assert flag in result.output
    assert _cfg_from_kwargs(_apply_config_defaults({"engine": "experimental", "mode": "head"})).model_backend == "liveportrait"


def test_offline_engine_never_opens_camera():
    cfg = build_config(engine="experimental", mode="fullbody", model_backend="editalive")
    with pytest.raises(EngineUnavailable, match="offline"):
        create_engine(cfg)


def test_disabled_worker_never_verifies_or_starts_models(monkeypatch):
    from deepfake.engines.experimental import adapters

    monkeypatch.setattr(adapters, "backend_status", lambda name: {
        "available": False, "disabled_reason": "Disabled after host memory-pressure incident"})
    with pytest.raises(EngineUnavailable, match="Disabled after host memory-pressure"):
        adapters.render("editalive", None, None, None, None)


def test_session_lock_does_not_overwrite_active_state():
    bus = SessionBus("first")
    try:
        bus.update(state="running", engine="alphaface")
        with pytest.raises(RuntimeError, match="Another Deepfake session"):
            SessionBus("second")
        assert read_json(session_path())["session_id"] == "first"
    finally:
        bus.close()
    other = SessionBus("second")
    other.close()


def test_scoped_versioned_controls_and_legacy_stop():
    bus = SessionBus("ours")
    try:
        send_control("stale", "stop")
        send_control("ours", "stop")
        send_control("ours", "stop")
        write_json_atomic(state_dir() / "control.json", {"cmd": "stop"})
        commands = bus.drain_controls()
        assert len(commands) == 3
        assert not bus.drain_controls()
        assert not bus._valid_control({"cmd": "stop", "protocol_version": 999})
        assert not bus._valid_control({"cmd": "exec", "session_id": "ours"})
    finally:
        bus.close()


def test_private_atomic_state():
    write_json_atomic(state_dir() / "private.json", {"x": "y"})
    assert (state_dir() / "private.json").stat().st_mode & 0o777 == 0o600
    assert state_dir().stat().st_mode & 0o777 == 0o700


def test_preview_png_without_pillow():
    import cv2

    from deepfake.outputs.sinks import PreviewSink

    source = np.full((80, 120, 3), 173, dtype=np.uint8)
    sink = PreviewSink(filename="original.png", publish_stats=False)
    sink.write(source)
    decoded = cv2.imread(str(state_dir() / "original.png"))
    np.testing.assert_array_equal(decoded, source)
    assert not (state_dir() / "preview.json").exists()
    sink.close()


def test_manifest_validation_and_tampering(tmp_path, monkeypatch):
    import deepfake.experimental_models as module

    manifest = manifests()["liveportrait"]
    validate_manifest(manifest)
    for changes in ({"revision": "main"}, {"files": [{**manifest["files"][0], "path": "../escape"}]},
                    {"files": [{**manifest["files"][0], "sha256": "unknown"}]}):
        with pytest.raises(ValueError):
            validate_manifest({**manifest, **changes})
    import hashlib

    content = b"original"
    small = {**manifest, "files": [{**manifest["files"][0], "path": "weights.pth", "size": len(content),
                                    "sha256": hashlib.sha256(content).hexdigest()}]}
    monkeypatch.setattr(module, "manifests", lambda: {"liveportrait": small})
    target = models_dir() / "liveportrait" / "weights.pth"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    assert verify("liveportrait")["ok"]
    target.write_bytes(b"tampered")
    assert verify("liveportrait")["corrupt"] == ["weights.pth"]


def test_worker_command_argv_paths_not_shell():
    command = worker_command("editalive", Path("/a path"), Path("/venv/python"), Path("/ref ; $(echo no).png"),
                             Path("/video with space.mp4"), Path("/output"), prompt="a `command` ; text")
    assert "/ref ; $(echo no).png" in command
    assert "a `command` ; text" in command
    assert "--low_vram" in command and "weights+kv" in command


def test_desktop_validation_no_camera_without_consent():
    assert request({"protocol_version": 1, "action": "snapshot"})["session"]["active"] is False
    with pytest.raises(ValueError, match="consent"):
        request({"protocol_version": 1, "action": "start"})
    for values in ({"engine": "shell"}, {"width": -1}, {"fps": True}, {"unknown": "anything"}):
        with pytest.raises(ValueError):
            validate_preferences(values)
    with pytest.raises(ValueError, match="protocol"):
        request({"protocol_version": 0, "action": "snapshot"})
    argv = build_session_command({**DEFAULT_PREFERENCES, "reference": "/face with space.png"})
    assert "/face with space.png" in argv and "--no-widget" in argv


def test_character_library_content_addressed(tmp_path):
    import cv2

    from deepfake.characters import import_character, list_characters, remove_character

    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 200, np.uint8))
    row = import_character(path, "Synthetic", "Synthetic fixture")
    assert list_characters()[0]["id"] == row["id"]
    assert Path(row["path"]).stat().st_mode & 0o777 == 0o600
    remove_character(row["id"])
    assert not list_characters()


def test_offline_order_and_source_timeline(tmp_path, monkeypatch):
    import cv2

    import deepfake.realtime as module
    from deepfake.outputs.sinks import NullSink

    image = tmp_path / "source.png"
    cv2.imwrite(str(image), np.zeros((64, 64, 3), np.uint8))
    frames = [np.full((64, 64, 3), i, np.uint8) for i in range(7)]

    class Capture:
        def __init__(self):
            self.i = 0
        def get(self, prop):
            return {cv2.CAP_PROP_FPS: 17, cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 64}.get(prop, 0)
        def read(self):
            self.i += 1
            return (True, frames[self.i - 1]) if self.i <= len(frames) else (False, None)
        def release(self):
            pass

    class Engine:
        closed = False
        def set_source(self, frame):
            pass
        def swap_frame(self, frame):
            return frame, None
        def close(self):
            self.closed = True

    class Sink(NullSink):
        def write(self, frame):
            seen.append(int(frame[0, 0, 0]))

    engine, seen, rates = Engine(), [], []
    monkeypatch.setattr(module, "create_engine", lambda cfg: engine)
    monkeypatch.setattr(module, "open_capture", lambda source: Capture())

    def factory(w, h, fps):
        rates.append((w, h, fps))
        return Sink()

    result = module.run_offline("fixture", image, factory, replace(build_config(), watermark=False), None)
    assert seen == list(range(7))
    assert rates == [(64, 64, 17)]
    assert result["source_frames"] == 7 and engine.closed


def test_desktop_request_cli_json():
    result = CliRunner().invoke(main, ["desktop", "request", '{"protocol_version":1,"action":"snapshot"}'])
    assert result.exit_code == 0
    assert json.loads(result.output)["ok"]
