"""Pinned, isolated offline adapters for official research repositories.

No shell interpolation, network services, or implicit checkpoint downloads.
The parent retains ownership of consent, session state, output sinks and disclosure.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from ...ipc import read_json, write_json_atomic
from ...paths import config_home, models_dir
from ..base import EngineUnavailable

SOURCES = {
    "liveportrait": ("https://github.com/KlingAIResearch/LivePortrait", "9b294b3d0536135442ea73cb01e6cb3ca7029dd3"),
    "personalive": ("https://github.com/GVCLab/PersonaLive", "abdd112e01dcf7d89122c2e5efa29fcff0669740"),
    "editalive": ("https://github.com/GVCLab/EditaLive", "9fe7fdb4ce3ff69305f026e8dec265aaf7a4196f"),
}


def settings_path():
    return config_home() / "experimental.json"


def configure_backend(name: str, checkout: Path, python: Path):
    if name not in SOURCES:
        raise ValueError(f"Unknown backend {name}")
    checkout, python = checkout.expanduser().resolve(), python.expanduser().absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Worker Python must be an executable file")
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCES[name][1]:
        raise ValueError(f"Expected {SOURCES[name][1]}, found {revision}. Use the documented pinned revision.")
    data = read_json(settings_path())
    data[name] = {"checkout": str(checkout), "python": str(python), "revision": revision}
    write_json_atomic(settings_path(), data)


def backend_status(name):
    import importlib.util

    from ...models import is_model_ready

    cfg = read_json(settings_path()).get(name, {})
    configured = (Path(cfg.get("checkout", "/nonexistent")).is_dir()
                  and Path(cfg.get("python", "/nonexistent")).is_file())
    installed = is_model_ready(name)
    disabled_reason = cfg.get("disabled_reason", "")
    missing_dependencies = [module for module in ("torch", "scipy", "yaml")
                            if name == "liveportrait" and importlib.util.find_spec(module) is None]
    return {"configured": configured, "installed": installed,
            "available": bool(configured and installed and not missing_dependencies and not disabled_reason),
            "disabled_reason": disabled_reason,
            "missing_dependencies": missing_dependencies,
            "validation": "synthetic CUDA sequence measured" if name == "liveportrait" else "integration unvalidated on GPU",
            "status": disabled_reason or (("experimental head adapter" if name == "liveportrait" else "offline adapter") if configured and installed else "needs weights and source/environment setup"),
            "source": SOURCES[name][0], "revision": SOURCES[name][1]}


def worker_command(name, checkout, python, reference, video, output_dir, *, prompt="", fps=25):
    weights = models_dir() / name
    if name == "personalive":
        return [str(python), str(Path(__file__).with_name("personalive_worker.py")),
                "--checkout", str(checkout), "--weights", str(weights), "--reference", str(reference),
                "--video", str(video), "--output", str(output_dir / "generated.mp4")]
    if name == "editalive":
        loras = [str(weights / "EditaLive" / f) for f in (
            "editalive_edit.safetensors", "lightx2v.safetensors", "editalive_streaming.safetensors")]
        return [str(python), str(Path(__file__).with_name("editalive_worker.py")),
                "--checkout", str(checkout), "--metrics", str(output_dir / "generated.metrics.json"),
                "--ckpt_dir", str(weights / "Wan-Animate"),
                "--video", str(video), "--ref_image", str(reference), "--prompt",
                prompt or "Transform the person to match the reference character, preserving the pose and background.",
                "--lora_paths", *loras, "--save_dir", str(output_dir), "--frame_num", "-1",
                "--long_edge", "672", "--short_edge", "384", "--preprocess_long_edge", "672",
                "--preprocess_short_edge", "384", "--preprocess_fps", str(fps), "--fps", str(fps),
                "--enable_compile", "False", "--fast_decode", "False", "--fp8", "--low_vram", "weights+kv",
                "--t5_cpu", "--keep_audio", "False", "--vsa_backend", "sdpa"]
    raise EngineUnavailable(f"Unknown experimental backend: {name}")


def render(name, reference, video, output_dir, session, *, prompt="", fps=25):
    from ...models import verify_model

    status = backend_status(name)
    if status.get("disabled_reason"):
        raise EngineUnavailable(status["disabled_reason"])
    if not status["available"]:
        raise EngineUnavailable(f"{name}: install weights with deepfake models install {name} --yes, then "
                                f"configure its environment using deepfake experimental configure --backend {name}.")
    def verification_progress(filename, done, total):
        if session.stop_requested():
            raise InterruptedError("Render cancelled during checkpoint verification")
        session.bus.update(state="starting", loading_stage="Verifying checkpoints", model_file=filename,
                           model_bytes_verified=done, model_bytes_total=total)

    verification = verify_model(name, progress=verification_progress)
    if not verification["ok"]:
        raise EngineUnavailable(f"{name}: checkpoint verification failed: {verification}")
    cfg = read_json(settings_path())[name]
    checkout, python = Path(cfg["checkout"]), Path(cfg["python"])
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != SOURCES[name][1]:
        raise EngineUnavailable("Worker source revision changed; restore the pinned checkout before rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    command = worker_command(name, checkout, python, reference, video, output_dir, prompt=prompt, fps=fps)
    env = dict(os.environ)
    # A Python 3.10/3.11 worker must not inherit Python 3.13 packages from the host.
    env.pop("PYTHONPATH", None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TORCH_FORCE_WEIGHTS_ONLY_LOAD="1")
    if name == "editalive":
        # Upstream's FP8 helper has an unconditional @torch.compile decorator,
        # even with --enable_compile False. Keep this conservative profile eager.
        env["TORCHDYNAMO_DISABLE"] = "1"
    if Path("/run/opengl-driver/lib/libcuda.so.1").is_file():
        # Triton 3.2 otherwise invokes /sbin/ldconfig, which NixOS does not ship.
        env.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib")
    log_path = output_dir / "worker.log"
    session.bus.update(state="processing", loading_stage="Loading models / offline inference",
                       worker_log=str(log_path), backend=name, engine="experimental")
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=checkout, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                if session.stop_requested():
                    raise InterruptedError("Render cancelled")
                session.bus.update(state="processing", uptime_s=round(time.monotonic() - started, 1))
                time.sleep(0.2)
            if process.returncode:
                tail = log_path.read_text(errors="replace")[-2500:]
                from ...ipc import state_dir

                persistent_log = state_dir() / "experimental-worker-error.log"
                persistent_log.write_text(log_path.read_text(errors="replace"))
                session.bus.update(worker_log=str(persistent_log),
                                   worker_metrics=read_json(output_dir / "generated.metrics.json"))
                raise EngineUnavailable(f"{name} worker exited {process.returncode}:\n{tail}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    videos = [p for p in output_dir.glob("*.mp4") if "comparison" not in p.stem]
    if len(videos) != 1:
        raise EngineUnavailable(f"Expected one generated video, found {len(videos)}. See {log_path}")
    metrics = read_json(output_dir / "generated.metrics.json")
    return videos[0], {"worker_wall_s": time.monotonic() - started, "backend": name, **metrics}
