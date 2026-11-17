"""Render the Galaxy control center offscreen with a fake backend; fail on QML warnings.

The fake CLI answers `desktop request` with a fixed snapshot, so this checks QML loading, imports,
the Galaxy singleton and every page without a camera. The pages render under Quickshell (`qs`) when
it is installed, and under plain Qt with the stand-in types of tests/quickshell_shim when PySide6
is; with neither only the static checks run.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

QS = shutil.which("qs") or shutil.which("quickshell")
SHELL_DIR = Path(__file__).resolve().parents[1] / "overlays/deepfake-preview/quickshell"
SHIM = Path(__file__).resolve().parent / "quickshell_shim" / "run.py"
HAVE_QT = importlib.util.find_spec("PySide6") is not None
PAGES = ["Studio", "Characters", "Models", "Render", "Performance", "Settings"]

SNAPSHOT = {
    "ok": True, "protocol_version": 1, "state_dir": "/nonexistent",
    "session": {"active": True, "state": "running", "ts": 0, "source_fps": 30.0, "native_inference_fps": 31.2,
                "output_fps": 60.0, "latency_ms": 41.5, "latency_p95_ms": 55.0, "dropped_source_frames": 0,
                "backend": "alphaface", "resolution": "1280x720", "framegen_enabled": True, "framegen_multiplier": 2},
    "preferences": {"theme": "galaxy", "comparison": "processed", "engine": "experimental", "mode": "head",
                    "backend": "liveportrait", "preset": "balanced", "frame_gen": "auto", "camera": "0", "width": 1280,
                    "height": 720, "fps": 30, "rotoscope": "auto", "background": "green", "matte_scope": "person",
                    "reference": "", "input_video": "", "recording_path": ""},
    "models": [{"name": "alphaface", "installed": True, "integrity": "verified", "load_check_ok": True,
                "code_license": "MIT", "weights_license": "UNDOCUMENTED", "size_hint": "2.1 GB"},
               {"name": "liveportrait", "installed": False, "experimental": True, "code_license": "MIT",
                "weights_license": "MIT", "size_hint": "0.49 GiB", "file_count": 5, "present_bytes": 0},
               {"name": "editalive", "installed": True, "experimental": True, "integrity": "verified", "code_license": "Apache-2.0",
                "weights_license": "Apache-2.0", "size_hint": "52.71 GiB", "file_count": 412, "present_bytes": 1}],
    "characters": [], "devices": [{"path": "/dev/video10", "name": "Galaxy Deepfake", "kind": "loopback"}],
    "engines": [{"name": "alphaface", "engine": "alphaface"},
                {"name": "liveportrait", "engine": "experimental", "available": False, "installed": False, "live": True,
                 "status": "weights not installed", "source": "https://github.com/KlingAIResearch/LivePortrait",
                 "revision": "9b294b3d0536135442ea73cb01e6cb3ca7029dd3", "suggested": {"checkout": "", "python": ""}},
                {"name": "editalive", "engine": "experimental", "available": False, "installed": True, "configured": True,
                 "live": False, "disabled_reason": "Disabled after host memory-pressure incident.",
                 "status": "Disabled after host memory-pressure incident.", "checkout": "/src/EditaLive",
                 "python": "/venv/bin/python", "checkout_ok": True, "python_ok": True,
                 "memory": {"limit_gb": 54.4, "total_gb": 64.0, "available_gb": 40.0, "method": "watchdog"}}],
    "version": "test", "matte_installed": False,
    "consent": True, "consent_text": "notice",
    "model_job": {"state": "running", "action": "install_model", "model": "liveportrait", "stage": "Downloading",
                  "bytes": 100, "total_bytes": 400}, "gpu": {"name": "Test GPU", "util_pct": 10,
                                                                       "mem_used_mib": 1000, "mem_total_mib": 24000,
                                                                       "temp_c": 40, "power_w": 60},
}


def _fake_backend(tmp_path, snapshot=None):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    import numpy as np

    cv2 = pytest.importorskip("cv2")
    yy, xx = np.mgrid[0:720, 0:1280]
    cv2.imwrite(str(state / "preview.png"), np.dstack([xx % 256, yy % 256, (xx + yy) % 256]).astype(np.uint8))
    (tmp_path / "snap.json").write_text(json.dumps({**(snapshot or SNAPSHOT), "state_dir": str(state)}))
    fake = tmp_path / "fake-deepfake"
    fake.write_text(f"#!/bin/sh\ncat {tmp_path / 'snap.json'}\n")
    fake.chmod(0o755)
    return fake


def _env(tmp_path, fake, page, size, shot):
    return {**os.environ, "QT_QPA_PLATFORM": "offscreen", "DEEPFAKE_CLI": str(fake), "DEEPFAKE_UI_PAGE": page,
            "DEEPFAKE_UI_SIZE": size, "DEEPFAKE_UI_SCREENSHOT": str(shot), "DEEPFAKE_UI_SCREENSHOT_QUIT": "1"}


CASES = [("Studio", "1240x840"), ("Characters", "1240x840"), ("Models", "1240x1400"), ("Render", "1240x840"),
         ("Performance", "760x560"), ("Settings", "1600x900"), ("Studio", "760x560"),
         ("Dashboard", "1180x820")]  # the pre-0.5 page name still opens the Studio


@pytest.mark.skipif(QS is None, reason="quickshell not installed")
@pytest.mark.parametrize("page,size", CASES)
def test_control_center_renders_without_warnings(tmp_path, page, size):
    fake = _fake_backend(tmp_path)
    shot = tmp_path / "shot.png"
    r = subprocess.run([QS, "-p", str(SHELL_DIR)], env=_env(tmp_path, fake, page, size, shot), capture_output=True, text=True, timeout=60)
    log = r.stdout + r.stderr
    warnings = [line for line in log.splitlines() if ("WARN" in line or "ERROR" in line) and "window masks" not in line]
    assert not warnings, "\n".join(warnings)
    assert shot.is_file() and shot.stat().st_size > 10_000


@pytest.mark.skipif(not HAVE_QT, reason="PySide6 not installed")
@pytest.mark.parametrize("page,size", CASES)
def test_control_center_renders_under_plain_qt(tmp_path, page, size):
    """The same pages through the Quickshell stand-in: QML errors and unresolved bindings fail here without `qs`."""
    idle = {**SNAPSHOT, "session": {"active": False, "state": "error", "mode": "video", "backend": "editalive",
                                    "session_id": "x", "error": "worker stopped by the memory guard"}}
    fake = _fake_backend(tmp_path, SNAPSHOT if page in ("Studio", "Performance") else idle)
    shot = tmp_path / "shot.png"
    env = {**_env(tmp_path, fake, page, size, shot), "QT_QUICK_BACKEND": "software"}
    r = subprocess.run([sys.executable, str(SHIM), str(SHELL_DIR)], env=env, capture_output=True, text=True, timeout=90)
    log = r.stdout + r.stderr
    if "could not be loaded" in log or "libEGL" in log:
        pytest.skip("Qt cannot start on this machine: " + log.strip().splitlines()[-1])
    # Qt 6.10+ notes that the shared GalaxySpark's `data` property shadows Item.data; it is not an error
    warnings = [line for line in log.splitlines() if ("WARN" in line or "ERROR" in line) and "overrides a member" not in line]
    assert not warnings, "\n".join(warnings)
    assert shot.is_file() and shot.stat().st_size > 10_000


def test_every_qml_type_used_is_shipped_and_installed(tmp_path, monkeypatch):
    """Static: components referenced by the shell exist beside it, and `desktop install` copies all of them."""
    sources = {p.stem: p.read_text() for p in SHELL_DIR.glob("*.qml")}
    for page in PAGES:
        assert f"{page}Page" in sources and f"{page}Page {{" in sources["ControlCenter"]
        assert f'"{page}"' in sources["ControlCenter"]
    local = set(sources)
    used = {name for text in sources.values() for name in re.findall(r"^\s*([A-Z][A-Za-z]+) \{", text, re.M)}
    ours = {n for n in used if n.startswith(("Galaxy", "Text", "Card", "Check", "Choice", "Key", "Log", "Path", "Preview", "Segmented", "Starfield")) or n.endswith("Page")}
    missing = ours - local - {"Text", "TextEdit", "TextField", "TextInput", "TextMetrics"}
    assert not missing, f"QML types used but not shipped: {sorted(missing)}"
    # every action the QML sends is one the backend implements
    import inspect

    from deepfake import desktop

    backend = inspect.getsource(desktop.request) + " ".join(desktop.MODEL_ACTIONS)
    sent = {a for text in sources.values() for a in re.findall(r'enqueue\("([a-z_]+)"', text)}
    sent |= {a for text in sources.values() for a in re.findall(r"\b([a-z]+_model)\b", text) if a != "experimental_model"}
    assert sent and all(action in backend for action in sent), sorted(a for a in sent if a not in backend)

    from deepfake.desktop import install_desktop

    launcher = tmp_path / "deepfake"
    launcher.write_text("#!/bin/sh\n")
    launcher.chmod(0o755)
    target = Path(install_desktop(launcher)["quickshell"])
    assert {p.name for p in SHELL_DIR.glob("*.qml")} == {p.name for p in target.glob("*.qml")}


@pytest.mark.skipif(QS is None, reason="quickshell not installed")
def test_browse_dialog_opens(tmp_path):
    """Browse… must open a file dialog: declared on ShellRoot (no window) it never became visible."""
    fake = _fake_backend(tmp_path)
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", "DEEPFAKE_CLI": str(fake), "DEEPFAKE_UI_TEST_DIALOG": "1",
           "DEEPFAKE_UI_SCREENSHOT": str(tmp_path / "s.png"), "DEEPFAKE_UI_SCREENSHOT_QUIT": "1"}
    r = subprocess.run([QS, "-p", str(SHELL_DIR)], env=env, capture_output=True, text=True, timeout=60)
    assert "DIALOG visible=true" in r.stdout + r.stderr, (r.stdout + r.stderr)[-2000:]
