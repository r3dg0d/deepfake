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
                       "width": 640, "height": 480, "fps": 30, "output_device": "", "input_video": "",
                       # LivePortrait head engine: matte (rotoscope) and background replacement
                       "rotoscope": "auto", "background": "keep", "background_image": "", "matte_scope": "person",
                       "prompt": ""}
# Face geometry (deepfake.geometry). The legacy values are bbox / oval / yunet / False / stretch.
GEOMETRY_PREFERENCES = {"alignment": "landmark", "mask_mode": "landmark", "landmarker": "auto",
                        "landmark_smooth": True, "fit": "cover"}
DEFAULT_PREFERENCES.update(GEOMETRY_PREFERENCES)


def geometry_args(prefs):
    return ["--align", prefs["alignment"], "--mask", prefs["mask_mode"], "--landmarker", prefs["landmarker"],
            "--landmark-smooth" if prefs["landmark_smooth"] else "--no-landmark-smooth", "--fit", prefs["fit"]]


ENUMS = {"theme": {"galaxy", "dark", "light", "matrix"},  # only galaxy is drawn; others kept for old files
         "comparison": {"original", "processed", "side-by-side", "slider"},
         "engine": {"alphaface", "experimental"}, "mode": {"head", "fullbody"},
         "backend": {"personalive", "editalive", "liveportrait"}, "preset": {"latency", "balanced", "quality"},
         "frame_gen": {"auto", "off", "2x", "3x", "4x"}, "encoder": {"auto", "libx264", "h264_nvenc"},
         "rotoscope": {"auto", "on", "off"}, "matte_scope": {"person", "head"},
         "alignment": {"landmark", "bbox"}, "mask_mode": {"landmark", "oval"},
         "landmarker": {"auto", "mediapipe", "yunet"}, "fit": {"cover", "contain", "stretch"}}


def preferences():
    """Stored preferences over the defaults. Keys from another version are dropped, never fatal."""
    stored = {k: v for k, v in read_json(preferences_path()).items() if k in DEFAULT_PREFERENCES}
    merged = {**DEFAULT_PREFERENCES, **stored}
    for key in stored:
        try:
            validate_preferences({key: merged[key]})
        except ValueError:
            merged[key] = DEFAULT_PREFERENCES[key]
    return merged


def validate_preferences(data):
    if not isinstance(data, dict) or set(data) - set(DEFAULT_PREFERENCES):
        raise ValueError("Unknown desktop preference")
    for key, value in data.items():
        if key in ENUMS and value not in ENUMS[key]:
            raise ValueError(f"Invalid {key}")
        if key in {"width", "height", "fps"} and (type(value) is not int or not 1 <= value <= (240 if key == "fps" else 4096)):
            raise ValueError(f"Invalid {key}")
        if key in {"mirror", "landmark_smooth"} and not isinstance(value, bool):
            raise ValueError(f"Invalid {key} preference")
        if key not in ENUMS and key not in {"width", "height", "fps", "mirror", "landmark_smooth"} and (not isinstance(value, str) or len(value) > 4096):
            raise ValueError(f"Invalid {key}")
        if key == "background" and value != "image":
            from .matting import BACKGROUND_CHOICES, parse_background

            if value not in BACKGROUND_CHOICES and not value.startswith("#"):
                raise ValueError("Invalid background")
            parse_background(value)  # rejects malformed #rrggbb
    return data


def head_matte_args(prefs) -> list[str]:
    """CLI flags for the matte; only the LivePortrait head engine has one."""
    if not (prefs["engine"] == "experimental" and prefs["mode"] == "head" and prefs["backend"] == "liveportrait"):
        return []
    background = prefs["background_image"] if prefs["background"] == "image" else prefs["background"]
    if prefs["background"] == "image" and not Path(background).expanduser().is_file():
        raise ValueError("Choose an existing background image, or another background")
    args = {"on": ["--rotoscope"], "off": ["--no-rotoscope"]}.get(prefs["rotoscope"], [])
    if background != "keep":
        if prefs["rotoscope"] == "off":
            raise ValueError("Background replacement needs rotoscoping: turn it on or keep the camera background")
        args += ["--background", str(Path(background).expanduser()) if prefs["background"] == "image" else background,
                 "--matte-scope", prefs["matte_scope"]]
    return args


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


def _pid_runs(pid, needle: str) -> bool:
    """The process exists and is still the one we started (pids are reused)."""
    try:
        return needle in Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except (OSError, ValueError, TypeError):
        return False


def model_job():
    """The model job record; a job whose process is gone is reported as interrupted, not running forever."""
    path = state_dir() / "model-job.json"
    job = read_json(path)
    if job.get("state") == "running" and not _pid_runs(job.get("pid"), "model-job"):
        job.update(state="interrupted", error="The model operation stopped unexpectedly (crash, reboot or kill). "
                                              "Partial downloads are kept: start it again to resume.")
        write_json_atomic(path, job)
    return job


def _tail(path: Path, limit: int = 5000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(max(0, path.stat().st_size - limit))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def snapshot():
    from . import __version__
    from .characters import list_characters
    from .config import load_config
    from .consent import CONSENT_TEXT, is_acknowledged
    from .devices import list_v4l2_devices
    from .engines import engine_status
    from .matting import model_installed
    from .models import list_models

    session = read_json(session_path())
    session["active"] = bool(session.get("active")) and time.time() - session.get("ts", 0) < 15
    if (not session["active"] and session.get("state") in {"running", "starting", "processing", "paused", "stopping"}
            and not _pid_runs(session.get("pid"), "deepfake")):
        session["state"] = "stopped"  # the process died without closing its state (kill -9, crash)
    return {"protocol_version": PROTOCOL_VERSION, "version": __version__, "session": session,
            "preferences": preferences(), "engines": engine_status(),
            "models": [{"name": m.name, "installed": m.installed, **m.meta} for m in list_models()],
            "characters": list_characters(), "devices": [vars(d) for d in list_v4l2_devices()],
            "consent": is_acknowledged(), "consent_text": CONSENT_TEXT, "model_job": model_job(),
            "matte_installed": model_installed(), "watermark": bool(load_config().watermark),
            "state_dir": str(state_dir()), "gpu": gpu_info()}


def build_session_command(prefs, *, virtual=False, recording=False):
    validate_preferences(prefs)
    args = [sys.executable, "-m", "deepfake", "virtualcam" if virtual else "webcam", "--no-widget", "--preview",
            "--engine", prefs["engine"], "--preset", prefs["preset"], "--input-device", prefs["camera"],
            "--width", str(prefs["width"]), "--height", str(prefs["height"]), "--fps", str(prefs["fps"]),
            "--frame-gen", prefs["frame_gen"]] + geometry_args(prefs)
    if prefs["reference"]:
        args += ["--reference", prefs["reference"]]
    if prefs["engine"] == "experimental":
        args += ["--mode", prefs["mode"], "--model-backend", prefs["backend"], *head_matte_args(prefs)]
    if virtual and prefs["output_device"]:
        args += ["--output-device", prefs["output_device"]]
    if recording:
        path = Path(prefs["recording_path"]).expanduser()
        if not prefs["recording_path"] or not path.parent.is_dir() or path.exists():
            raise ValueError("Choose a new recording filename in an existing directory")
        args += ["-o", str(path), "--encoder", prefs["encoder"]]
    return args


MODEL_ACTIONS = {"install_model", "repair_model", "reinstall_model", "verify_model", "remove_model"}


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
        prefs = preferences()
        if not prefs["reference"] or not Path(prefs["reference"]).expanduser().is_file():
            raise ValueError("Choose a character or reference image first")
        offline = prefs["engine"] == "experimental" and prefs["backend"] != "liveportrait"
        if action == "start":
            if offline:
                raise ValueError(f"{prefs['backend']} renders video files only: use Render, or pick a live engine")
            args = build_session_command(prefs, virtual=message.get("virtual") is True,
                                         recording=message.get("recording") is True)
        else:
            source_video = Path(prefs["input_video"]).expanduser()
            output = Path(prefs["recording_path"]).expanduser()
            if not prefs["input_video"] or not source_video.is_file():
                raise ValueError("Choose an existing input video")
            if not prefs["recording_path"] or not output.parent.is_dir() or output.exists():
                raise ValueError("Choose a new output filename in an existing folder (files are never overwritten)")
            validate_preferences(prefs)
            args = [sys.executable, "-m", "deepfake", "video", "-i", str(source_video), "-o", str(output),
                    "--no-widget", "--preview", "--engine", prefs["engine"], "--reference", prefs["reference"],
                    "--preset", prefs["preset"], "--frame-gen", "off", "--encoder", prefs["encoder"]] + geometry_args(prefs)
            if prefs["engine"] == "experimental":
                args += ["--mode", prefs["mode"], "--model-backend", prefs["backend"], *head_matte_args(prefs)]
                if offline:
                    from .engines.experimental.adapters import backend_status

                    status = backend_status(prefs["backend"])
                    if not status["available"]:  # say why here instead of failing inside a detached process
                        raise ValueError(f"{prefs['backend']} cannot render yet: {status['status']}")
                    if prefs["prompt"] and prefs["backend"] == "editalive":
                        args += ["--prompt", prefs["prompt"]]
        log_path = state_dir() / "backend.log"
        with log_path.open("w") as log:
            proc = subprocess.Popen(args, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        return {"message": "Starting…" if action == "start" else "Render started", "pid": proc.pid, "log": str(log_path)}
    if action == "configure_backend":
        from .engines.experimental.adapters import SOURCES, backend_status, configure_backend

        name = message.get("backend")
        if name not in SOURCES:
            raise ValueError("Unknown backend")
        checkout, python = str(message.get("checkout") or "").strip(), str(message.get("python") or "").strip()
        if not checkout:
            raise ValueError(f"Enter the path of the {SOURCES[name][0].rsplit('/', 1)[-1]} source checkout")
        if not python:
            if name != "liveportrait":
                raise ValueError("Enter the Python interpreter of the isolated worker environment")
            python = sys.executable
        configure_backend(name, Path(checkout), Path(python))
        return {"message": f"{name} configured", "status": backend_status(name)}
    if action in {"enable_backend", "disable_backend"}:
        from .engines.experimental.adapters import SOURCES, backend_status, set_backend_enabled

        name = message.get("backend")
        if name not in SOURCES:
            raise ValueError("Unknown backend")
        if action == "enable_backend" and message.get("confirmed") is not True:
            raise ValueError("Review the memory guard and confirm before enabling this backend")
        limit = message.get("memory_limit_gb")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int | float)):
            raise ValueError("Invalid memory limit")
        set_backend_enabled(name, action == "enable_backend", memory_limit_gb=limit,
                            reason=str(message.get("reason") or "")[:500])
        status = backend_status(name)
        return {"message": f"{name} enabled under the memory guard ({status['memory']['limit_gb']} GiB ceiling)"
                if action == "enable_backend" and status.get("memory") else f"{name} {action.split('_')[0]}d", "status": status}
    if action == "import_character":
        from .characters import import_character

        return {"character": import_character(Path(message["path"]), message["name"], message["provenance"]),
                "message": "Character imported locally"}
    if action == "remove_character":
        from .characters import remove_character

        remove_character(message["id"])
        return {"message": "Character removed"}
    if action in MODEL_ACTIONS:
        from .models import list_models

        name = message.get("model")
        if name not in {m.name for m in list_models()}:
            raise ValueError("Unknown model")
        if action != "verify_model" and message.get("confirmed") is not True:
            raise ValueError("Review the model license, download size, or removal and confirm")
        if model_job().get("state") == "running":
            raise ValueError("A model operation is already running")
        session = read_json(session_path())
        if action in {"remove_model", "reinstall_model"} and session.get("active") and time.time() - session.get("ts", 0) < 15:
            raise ValueError("Stop the active session before removing or replacing model files")
        with (state_dir() / "model-job.log").open("w") as log:
            proc = subprocess.Popen([sys.executable, "-m", "deepfake", "desktop", "model-job", action, name],
                                    stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        # Written here as well, so the very next snapshot already shows the job (the child needs a moment to start).
        write_json_atomic(state_dir() / "model-job.json", {"state": "running", "action": action, "model": name,
                                                            "pid": proc.pid, "stage": "Starting"})
        return {"message": f"{action.split('_')[0].capitalize()} {name} started"}
    if action == "cancel_model":
        import signal

        job_path = state_dir() / "model-job.json"
        job = model_job()
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
        which = message.get("which", "backend")
        names = {"backend": "backend.log", "worker": "experimental-worker.log", "model": "model-job.log"}
        if which not in names:
            raise ValueError("Unknown log")
        text = _tail(state_dir() / names[which])
        return {"message": f"{which} log loaded" if text else f"No {which} log yet", "which": which, "log": text}
    raise ValueError(f"Unsupported desktop action {action!r}")
