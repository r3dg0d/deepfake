import io
import subprocess
from types import SimpleNamespace

import pytest

from deepfake.outputs.sinks import V4L2LoopbackSink


@pytest.mark.parametrize("code", [0, 1])
def test_virtualcam_writer_failure_reported(code):
    sink = V4L2LoopbackSink.__new__(V4L2LoopbackSink)
    sink._device = "/dev/test"
    sink._proc = SimpleNamespace(stdin=io.BytesIO(), wait=lambda **kw: code)
    if code:
        with pytest.raises(RuntimeError, match="ffmpeg exit 1"):
            sink.close()
    else:
        sink.close()
    assert sink._proc.stdin.closed


def test_virtualcam_timeout_kills_and_reaps_writer():
    sink = V4L2LoopbackSink.__new__(V4L2LoopbackSink)
    sink._device = "/dev/test"
    events = []

    def wait(timeout):
        events.append(("wait", timeout))
        if timeout == 10:
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        return -9

    sink._proc = SimpleNamespace(stdin=io.BytesIO(), wait=wait, kill=lambda: events.append("kill"))
    with pytest.raises(RuntimeError, match="timed out"):
        sink.close()
    assert events == [("wait", 10), "kill", ("wait", 5)]
