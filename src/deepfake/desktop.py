"""Versioned, validated desktop requests over argv/stdout; frames stay on disk.

Quickshell is the only frontend. This bridge reuses the CLI and SessionBus.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .ipc import PROTOCOL_VERSION, read_json, send_control, session_path, state_dir, write_json_atomic
from .paths import config_home


def install_desktop(executable: Path):
    """Update the existing launcher identity, preserving a recoverable backup."""
    from .widget import widget_shell_dir

    executable = executable.expanduser().absolute()
    if not executable.is_file() or not os.access(executable, os.X_OK) or "\n" in str(executable):
        raise ValueError("Select an executable Deepfake launcher")
    packaged = Path(__file__).parent / "quickshell"
    repo = Path(__file__).parents[2] / "overlays/deepfake-preview/quickshell"
    source = packaged if (packaged / "ControlCenter.qml").is_file() else repo
    if not (source / "ControlCenter.qml").is_file():
        raise ValueError("ControlCenter.qml is missing from this installation")
    config_root = config_home().parent
    target = config_root / "quickshell/deepfake"
    backup = state_dir() / ("desktop-backup-" + str(time.time_ns()))
    backup.mkdir(mode=0o700)
    if target.exists():
        shutil.copytree(target, backup / "quickshell")
    target.mkdir(parents=True, exist_ok=True)
    # The Galaxy components (Galaxy.qml singleton, GalaxyButton, GalaxyPanel, ...) live next to the shell.
    for f in sorted(source.glob("*.qml")) + [source / n for n in ("run.sh", "control.py", "noqs.sh") if (source / n).is_file()]:
        shutil.copyfile(f, target / f.name)
        if f.suffix in (".sh", ".py"):
            os.chmod(target / f.name, 0o755)
    # Preserve the existing app-launcher identity and icon name.
    from .paths import data_home

    applications = data_home().parent / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    entry = applications / "galaxy-deepfake.desktop"
    if entry.exists():
        shutil.copyfile(entry, backup / entry.name)
    # Desktop Entry quoting, not shell quoting: no shell ever interprets Exec.
    command = str(executable).replace("\\", "\\\\").replace('"', '\\"').replace("`", "\\`").replace("$", "\\$").replace("%", "%%")
    entry.write_text('[Desktop Entry]\nType=Application\nName=Deepfake\nGenericName=Neural Video Control Center\n'
                     'Comment=Local neural video transformation and disclosed avatars\n'
                     f'Exec="{command}" app\nIcon=galaxy-deepfake\nTerminal=false\nCategories=AudioVideo;Video;Utility;\n'
                     'Keywords=deepfake;face;camera;quickshell;avatar;\n')
    icon = Path(__file__).parent / "assets/galaxy-deepfake.svg"
    if not icon.is_file():
        icon = Path(__file__).parents[2] / "overlays/deepfake-preview/desktop/galaxy-deepfake.svg"
    if icon.is_file():
        destination = data_home().parent / "icons/hicolor/scalable/apps/galaxy-deepfake.svg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.copyfile(destination, backup / "galaxy-deepfake.svg")
        shutil.copyfile(icon, destination)
    return {"launcher": str(entry), "quickshell": str(target), "backup": str(backup),
            "previous_shell": str(widget_shell_dir())}


def preferences_path():
    return config_home() / "desktop.json"


DEFAULT_PREFERENCES = {"theme": "galaxy", "mirror": False, "comparison": "processed", "camera": "0",
                       "preset": "balanced", "engine": "alphaface", "mode": "head", "backend": "liveportrait",
                       "reference": "", "recording_path": "", "encoder": "auto", "frame_gen": "auto",
                       "width": 640, "height": 480, "fps": 30, "output_device": ""}
DEFAULT_PREFERENCES["input_video"] = ""


def preferences():
    return {**DEFAULT_PREFERENCES, **read_json(preferences_path())}


def validate_preferences(data):
    if not isinstance(data, dict) or set(data) - set(DEFAULT_PREFERENCES):
        raise ValueError("Unknown desktop preference")
    enums = {"theme": {"galaxy", "dark", "light", "matrix"},  # only galaxy is drawn; others kept for old files
              "comparison": {"original", "processed", "side-by-side", "slider"},
             "engine": {"alphaface", "experimental"}, "mode": {"head", "fullbody"},
             "backend": {"personalive", "editalive", "liveportrait"}, "preset": {"latency", "balanced", "quality"},
             "frame_gen": {"auto", "off", "2x", "3x", "4x"}, "encoder": {"auto", "libx264", "h264_nvenc"}}
    for key, value in data.items():
        if key in enums and value not in enums[key]:
            raise ValueError(f"Invalid {key}")
        if key in {"width", "height", "fps"} and (type(value) is not int or not 1 <= value <= (240 if key == "fps" else 4096)):
            raise ValueError(f"Invalid {key}")
        if key == "mirror" and not isinstance(value, bool):
            raise ValueError("Invalid mirror preference")
        if key not in enums and key not in {"width", "height", "fps", "mirror"} and (not isinstance(value, str) or len(value) > 4096):
            raise ValueError(f"Invalid {key}")
    return data


def gpu_info():
    """Device name / memory / utilisation from NVML (cheap, no subprocess). Empty when unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        name = pynvml.nvmlDeviceGetName(h)
        return {"name": name.decode() if isinstance(name, bytes) else name,
                "util_pct": pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                "mem_used_mib": round(mem.used / 2**20), "mem_total_mib": round(mem.total / 2**20),
                "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "power_w": round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000)}
    except Exception:  # noqa: BLE001
        return {}


def snapshot():
    from .characters import list_characters
    from .consent import CONSENT_TEXT, is_acknowledged
    from .devices import list_v4l2_devices
    from .engines import engine_status
    from .models import list_models

    session = read_json(session_path())
    session["active"] = bool(session.get("active")) and time.time() - session.get("ts", 0) < 15
    job = read_json(state_dir() / "model-job.json")
    return {"protocol_version": PROTOCOL_VERSION, "session": session, "preferences": preferences(),
            "engines": engine_status(), "models": [{"name": m.name, "installed": m.installed, **m.meta} for m in list_models()],
            "characters": list_characters(), "devices": [vars(d) for d in list_v4l2_devices()],
            "consent": is_acknowledged(), "consent_text": CONSENT_TEXT, "model_job": job,
            "state_dir": str(state_dir()), "gpu": gpu_info()}


def build_session_command(prefs, *, virtual=False, recording=False):
    validate_preferences(prefs)
    args = [sys.executable, "-m", "deepfake", "virtualcam" if virtual else "webcam", "--no-widget", "--preview",
            "--engine", prefs["engine"], "--preset", prefs["preset"], "--input-device", prefs["camera"],
            "--width", str(prefs["width"]), "--height", str(prefs["height"]), "--fps", str(prefs["fps"]),
            "--frame-gen", prefs["frame_gen"]]
    if prefs["reference"]:
        args += ["--reference", prefs["reference"]]
    if prefs["engine"] == "experimental":
        args += ["--mode", prefs["mode"], "--model-backend", prefs["backend"]]
    if virtual and prefs["output_device"]:
        args += ["--output-device", prefs["output_device"]]
    if recording:
        path = Path(prefs["recording_path"]).expanduser()
        if not prefs["recording_path"] or not path.parent.is_dir() or path.exists():
            raise ValueError("Choose a new recording filename in an existing directory")
        args += ["-o", str(path), "--encoder", prefs["encoder"]]
    return args


def request(message):
    if not isinstance(message, dict) or message.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported desktop protocol version")
    action = message.get("action")
    if action == "snapshot":
        return snapshot()
    if action == "preferences":
        updated = {**preferences(), **validate_preferences(message.get("values", {}))}
        write_json_atomic(preferences_path(), updated)
        return {"message": "Preferences saved", "preferences": updated}
    if action == "consent":
        if message.get("accepted") is not True:
            raise ValueError("Explicit acknowledgement required")
        from .consent import record_acknowledgement

        record_acknowledgement()
        return {"message": "Consent notice acknowledged"}
    if action == "stop":
        session = read_json(session_path())
        send_control(str(session.get("session_id", "")), "stop")
        return {"message": "Stopping session and finalizing recording…"}
    if action in {"start", "render_video"}:
        from .consent import is_acknowledged

        if not is_acknowledged():
            raise ValueError("Acknowledge the consent notice before starting a camera")
        session = read_json(session_path())
        if session.get("active") and time.time() - session.get("ts", 0) < 15:
            raise ValueError("Stop the active session before switching settings or outputs")
        args = build_session_command(preferences(), virtual=message.get("virtual") is True,
                                     recording=message.get("recording") is True)
        if action == "render_video":
            prefs = preferences()
            source_video = Path(prefs["input_video"]).expanduser()
            output = Path(prefs["recording_path"]).expanduser()
            if not source_video.is_file() or not prefs["recording_path"] or not output.parent.is_dir() or output.exists():
                raise ValueError("Choose an existing input video and a new output filename")
            args = [sys.executable, "-m", "deepfake", "video", "-i", str(source_video), "-o", str(output),
                    "--no-widget", "--preview", "--engine", prefs["engine"], "--reference", prefs["reference"],
                    "--preset", prefs["preset"], "--frame-gen", "off", "--encoder", prefs["encoder"]]
            if prefs["engine"] == "experimental":
                args += ["--mode", prefs["mode"], "--model-backend", prefs["backend"]]
        log_path = state_dir() / "backend.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        return {"message": "Starting…", "pid": proc.pid, "log": str(log_path)}
    if action == "import_character":
        from .characters import import_character

        return {"character": import_character(Path(message["path"]), message["name"], message["provenance"]),
                "message": "Character imported locally"}
    if action == "remove_character":
        from .characters import remove_character

        remove_character(message["id"])
        return {"message": "Character removed"}
    if action in {"install_model", "verify_model", "remove_model", "reinstall_model"}:
        from .models import list_models

        name = message.get("model")
        if name not in {m.name for m in list_models()}:
            raise ValueError("Unknown model")
        if action == "reinstall_model":
            from .model_store import CORE

            if name not in CORE:
                raise ValueError("Repair (staged redownload) is available for the core models")
        if action != "verify_model" and message.get("confirmed") is not True:
            raise ValueError("Review the model license, download size, or removal and confirm")
        job = read_json(state_dir() / "model-job.json")
        if job.get("state") == "running":
            try:
                os.kill(job["pid"], 0)
                raise ValueError("A model operation is already running")
            except ProcessLookupError:
                pass
        with (state_dir() / "model-job.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "deepfake", "desktop", "model-job", action, name],
                                    stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        return {"message": f"Model operation started (pid {proc.pid})"}
    if action == "cancel_model":
        import signal

        job_path = state_dir() / "model-job.json"
        job = read_json(job_path)
        if job.get("state") != "running" or not isinstance(job.get("pid"), int):
            raise ValueError("No model operation is running")
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        job.update(state="cancelled", message="Partial downloads are retained for resuming")
        write_json_atomic(job_path, job)
        return {"message": job["message"]}
    if action == "log":
        path = state_dir() / "backend.log"
        return {"message": path.read_text(errors="replace")[-5000:] if path.exists() else "No backend log yet"}
    raise ValueError(f"Unsupported desktop action {action!r}")
