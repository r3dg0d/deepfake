import numpy as np
import pytest

from deepfake.framegen.settings import FrameGenSettings
from deepfake.pipeline import build_config


def test_startup_failure_releases_sink_and_engine(monkeypatch):
    import deepfake.realtime as module

    class Engine:
        closed = False
        def set_source(self, frame):
            raise RuntimeError("synthetic GPU initialization failure")
        def close(self):
            self.closed = True

    class Sink:
        closed = False
        def close(self):
            self.closed = True

    engine, sink = Engine(), Sink()
    monkeypatch.setattr(module, "create_engine", lambda cfg: engine)
    with pytest.raises(RuntimeError, match="GPU initialization"):
        module.run_realtime(None, None, sink, build_config(), FrameGenSettings(enabled=False),
                            source_fps=30, source_bgr=np.zeros((64, 64, 3), np.uint8))
    assert engine.closed and sink.closed


def test_live_metrics_percentiles_separate_generated_frames():
    from deepfake.realtime import LiveStats

    stats = LiveStats()
    stats.direct_latency_ms.extend(range(1, 101))
    snapshot = stats.snapshot(None)
    assert snapshot["generated_frames"] == 0
    assert snapshot["queue_depth"] == 0
    assert snapshot["latency_p95_ms"] == pytest.approx(95.0, abs=0.1)


def test_camera_disconnect_unblocks_consumer():
    from deepfake.realtime import CaptureThread, LiveStats

    class Camera:
        def read(self):
            raise OSError("camera disconnected")

    worker = CaptureThread(Camera(), (64, 64), LiveStats())
    worker.start()
    worker.join(timeout=1)
    assert worker.eof and worker.take(timeout=0.01) is None


def test_existing_launcher_identity_and_backup(tmp_path):
    from deepfake.desktop import install_desktop
    from deepfake.paths import config_home, data_home

    target = config_home().parent / "quickshell/deepfake"
    target.mkdir(parents=True)
    (target / "shell.qml").write_text("existing configuration")
    executable = tmp_path / "deepfake with spaces"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    result = install_desktop(executable)
    entry = data_home().parent / "applications/galaxy-deepfake.desktop"
    assert entry.is_file() and '" app' in entry.read_text()
    assert "Terminal=false" in entry.read_text()
    from pathlib import Path

    assert (Path(result["backup"]) / "quickshell/shell.qml").read_text() == "existing configuration"
    assert (target / "ControlCenter.qml").is_file()


def test_widget_cleanup_does_not_kill_launcher(monkeypatch):
    import deepfake.widget as module

    def unexpected(*args, **kwargs):
        raise AssertionError("A persistent frontend must not be killed by session cleanup")

    monkeypatch.setattr(module.subprocess, "run", unexpected)
    module.cleanup_orphans()
    module.WidgetHandle().stop()


def test_live_recording_duration_when_inference_drops_frames():
    from deepfake.outputs.sinks import LiveRecordingSink

    seen, now = [], [0.0]

    class Writer:
        def write(self, frame):
            seen.append(frame)
        def close(self):
            pass

    sink = LiveRecordingSink(Writer(), 10, clock=lambda: now[0])
    sink.write("first")
    now[0] = 0.4
    sink.write("second")
    assert seen == ["first", "first", "first", "first", "second"]
    assert sink.held_frames == 3
    assert len(seen) / 10 == pytest.approx(0.5)
