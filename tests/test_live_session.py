from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepfake import cli, realtime
from deepfake.framegen.settings import FrameGenSettings
from deepfake.watermark import apply_watermark


@pytest.fixture
def live_stub(monkeypatch):
    events = []

    class Session:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            events.append("start")

        def apply_frame_gen(self, *a, **kw):
            pass

        def mark_running(self):
            events.append("running")

        def publish_stats(self, stats):
            events.append(("stats", stats))

        def close(self):
            events.append("close")

    monkeypatch.setattr(cli, "DeepfakeSession", Session)
    monkeypatch.setattr(cli, "_fg_from_kwargs", lambda *a: FrameGenSettings(enabled=False))
    monkeypatch.setattr(cli, "_make_sink", lambda *a, **kw: object())
    cfg = SimpleNamespace(preset=SimpleNamespace(width=640, height=480, fps=30), swap_precision="bf16", watermark=True)
    return cfg, events


@pytest.mark.parametrize(
    ("mode", "visible", "expected"), [("virtualcam", False, False), ("virtualcam", True, True), ("webcam", False, True)]
)
def test_live_output_label_and_final_metrics(monkeypatch, live_stub, mode, visible, expected):
    cfg, events = live_stub
    frame = np.full((480, 640, 3), 100, np.uint8)
    final = {"output_fps": 60.0, "keyframes": 3}
    output = []

    def run(*a, **kw):
        output.append(apply_watermark(frame, enabled=a[3].watermark))
        return final

    monkeypatch.setattr(realtime, "run_realtime", run)
    cli._run_live({"visible_watermark": visible, "output_device": "/dev/test"}, Path("face.png"), cfg, mode=mode)
    assert cfg.watermark is expected
    assert np.array_equal(output[0], frame) is (not expected)
    assert events == ["start", "running", ("stats", final), "close"]


def test_failed_sink_start_closes_live_session(monkeypatch, live_stub):
    cfg, events = live_stub

    def broken(*a, **kw):
        raise RuntimeError("virtual camera unavailable")

    monkeypatch.setattr(cli, "_make_sink", broken)
    with pytest.raises(RuntimeError, match="virtual camera unavailable"):
        cli._run_live({}, Path("face.png"), cfg, mode="virtualcam")
    assert events == ["start", "close"]
