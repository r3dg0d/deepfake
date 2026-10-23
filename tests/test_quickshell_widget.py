"""Render the Galaxy control center offscreen with a fake backend; fail on QML warnings.

Skipped when Quickshell (`qs`) is not installed. The fake CLI answers `desktop request` with a fixed
snapshot, so this checks QML loading, imports, the Galaxy singleton and every page without a camera.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

QS = shutil.which("qs") or shutil.which("quickshell")
SHELL_DIR = Path(__file__).resolve().parents[1] / "overlays/deepfake-preview/quickshell"

SNAPSHOT = {
    "ok": True, "protocol_version": 1, "state_dir": "/nonexistent",
    "session": {"active": True, "state": "running", "ts": 0, "source_fps": 30.0, "native_inference_fps": 31.2,
                "output_fps": 60.0, "latency_ms": 41.5, "latency_p95_ms": 55.0, "dropped_source_frames": 0,
                "backend": "alphaface", "resolution": "1280x720", "framegen_enabled": True, "framegen_multiplier": 2},
    "preferences": {"theme": "galaxy", "comparison": "processed", "engine": "alphaface", "preset": "balanced",
                    "frame_gen": "auto", "camera": "0", "width": 1280, "height": 720, "fps": 30},
    "models": [{"name": "alphaface", "installed": True, "integrity": "verified", "load_check_ok": True,
                "code_license": "MIT", "weights_license": "UNDOCUMENTED", "size_hint": "2.1 GB"},
               {"name": "liveportrait", "installed": False, "experimental": True, "code_license": "MIT",
                "weights_license": "MIT", "size_hint": "0.49 GiB"}],
    "characters": [], "devices": [{"path": "/dev/video10", "name": "Galaxy Deepfake", "kind": "loopback"}],
    "engines": [{"name": "liveportrait", "available": False, "status": "not installed"}],
    "consent": True, "consent_text": "notice", "model_job": {}, "gpu": {"name": "Test GPU", "util_pct": 10,
                                                                       "mem_used_mib": 1000, "mem_total_mib": 24000,
                                                                       "temp_c": 40, "power_w": 60},
}


@pytest.mark.skipif(QS is None, reason="quickshell not installed")
@pytest.mark.parametrize("page,size", [("Dashboard", "1180x820"), ("Models", "1180x820"), ("Performance", "760x560"),
                                       ("Engines", "760x560"), ("Settings", "1600x900")])
def test_control_center_renders_without_warnings(tmp_path, page, size):
    fake = tmp_path / "fake-deepfake"
    state = tmp_path / "state"
    state.mkdir()
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    yy, xx = np.mgrid[0:720, 0:1280]
    cv2.imwrite(str(state / "preview.png"), np.dstack([xx % 256, yy % 256, (xx + yy) % 256]).astype(np.uint8))
    (tmp_path / "snap.json").write_text(json.dumps({**SNAPSHOT, "state_dir": str(state)}))
    fake.write_text(f"#!/bin/sh\ncat {tmp_path / 'snap.json'}\n")
    fake.chmod(0o755)
    shot = tmp_path / "shot.png"
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "DEEPFAKE_CLI": str(fake), "DEEPFAKE_UI_PAGE": page,
           "DEEPFAKE_UI_SIZE": size, "DEEPFAKE_UI_SCREENSHOT": str(shot), "DEEPFAKE_UI_SCREENSHOT_QUIT": "1"}
    r = subprocess.run([QS, "-p", str(SHELL_DIR)], env=env, capture_output=True, text=True, timeout=60)
    log = r.stdout + r.stderr
    warnings = [line for line in log.splitlines() if ("WARN" in line or "ERROR" in line) and "window masks" not in line]
    assert not warnings, "\n".join(warnings)
    assert shot.is_file() and shot.stat().st_size > 10_000


@pytest.mark.skipif(QS is None, reason="quickshell not installed")
def test_browse_dialog_opens(tmp_path):
    """Browse… must open a file dialog: declared on ShellRoot (no window) it never became visible."""
    fake = tmp_path / "fake-deepfake"
    (tmp_path / "snap.json").write_text(json.dumps({**SNAPSHOT, "state_dir": str(tmp_path)}))
    fake.write_text(f"#!/bin/sh\ncat {tmp_path / 'snap.json'}\n")
    fake.chmod(0o755)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "DEEPFAKE_CLI": str(fake), "DEEPFAKE_UI_TEST_DIALOG": "1",
           "DEEPFAKE_UI_SCREENSHOT": str(tmp_path / "s.png"), "DEEPFAKE_UI_SCREENSHOT_QUIT": "1"}
    r = subprocess.run([QS, "-p", str(SHELL_DIR)], env=env, capture_output=True, text=True, timeout=60)
    assert "DIALOG visible=true" in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
