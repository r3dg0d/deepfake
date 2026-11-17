"""Pinned, isolated offline adapters for official research repositories.

No shell interpolation, network services, or implicit checkpoint downloads.
The parent retains ownership of consent, session state, output sinks and disclosure.
"""
from __future__ import annotations

import os
import shutil
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


def _revision(checkout: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise ValueError(f"{checkout} is not a git checkout of {checkout.name}") from e


def configure_backend(name: str, checkout: Path, python: Path):
    if name not in SOURCES:
        raise ValueError(f"Unknown backend {name}")
    checkout, python = checkout.expanduser().resolve(), python.expanduser().absolute()
    if not checkout.is_dir():
        raise ValueError(f"Source checkout not found: {checkout}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("Worker Python must be an executable file")
    revision = _revision(checkout)
    if revision != SOURCES[name][1]:
        raise ValueError(f"Expected {SOURCES[name][1]}, found {revision}. Use the documented pinned revision: "
                         f"git -C {checkout} checkout {SOURCES[name][1]}")
    data = read_json(settings_path())
    # Keep a disabled flag and the memory limit: re-pointing the paths must not silently re-enable a backend.
    data[name] = {**data.get(name, {}), "checkout": str(checkout), "python": str(python), "revision": revision}
    write_json_atomic(settings_path(), data)
    return data[name]


def suggested_paths(name: str) -> dict[str, str]:
    """Where a checkout / worker interpreter for ``name`` conventionally lives, if present (never created here)."""
    from ...paths import cache_home

    repo = SOURCES[name][0].rsplit("/", 1)[-1]
    checkouts = [cache_home() / "vendor" / repo, Path.home() / "Projects" / repo, Path.home() / "src" / repo]
    pythons = [Path.home() / ".venvs" / f"deepfake-{name}" / "bin" / "python",
               cache_home() / "venvs" / name / "bin" / "python"]
    return {"checkout": next((str(c) for c in checkouts if (c / ".git").exists()), ""),
            "python": next((str(c) for c in pythons if c.is_file()), "")}


def set_backend_enabled(name: str, enabled: bool, *, reason: str = "", memory_limit_gb: float | None = None):
    """Clear or set the disabled flag. Enabling an isolated worker always arms the memory guard."""
    if name not in SOURCES:
        raise ValueError(f"Unknown backend {name}")
    if memory_limit_gb is not None and not 4 <= memory_limit_gb <= 4096:
        raise ValueError("Memory limit must be between 4 and 4096 GiB")
    data = read_json(settings_path())
    cfg = dict(data.get(name, {}))
    if enabled:
        if cfg.pop("disabled_reason", ""):
            cfg["enabled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if memory_limit_gb is not None:
            cfg["memory_limit_gb"] = float(memory_limit_gb)
    else:
        cfg["disabled_reason"] = reason or "Disabled by the user"
    data[name] = cfg
    write_json_atomic(settings_path(), data)
    return cfg


def backend_status(name):
    import importlib.util

    from ...models import is_model_ready

    cfg = read_json(settings_path()).get(name, {})
    checkout_ok = Path(cfg.get("checkout", "/nonexistent")).is_dir()
    python_ok = Path(cfg.get("python", "/nonexistent")).is_file()
    configured = checkout_ok and python_ok
    installed = is_model_ready(name)
    disabled_reason = cfg.get("disabled_reason", "")
    missing_dependencies = [module for module in ("torch", "scipy", "yaml")
                            if name == "liveportrait" and importlib.util.find_spec(module) is None]
    live = name == "liveportrait"
    if disabled_reason:
        status = disabled_reason
    elif not installed:
        status = "weights not installed"
    elif not configured:
        status = "source checkout and worker Python not configured"
    elif missing_dependencies:
        status = "missing Python packages: " + ", ".join(missing_dependencies)
    else:
        status = "experimental head adapter" if live else "offline adapter, ready to render"
    out = {"configured": configured, "installed": installed,
           "available": bool(configured and installed and not missing_dependencies and not disabled_reason),
           "disabled_reason": disabled_reason,
           "missing_dependencies": missing_dependencies,
           "validation": "synthetic CUDA sequence measured" if live else "integration unvalidated on GPU",
           "status": status, "live": live,
           "checkout": cfg.get("checkout", ""), "python": cfg.get("python", ""),
           "checkout_ok": checkout_ok, "python_ok": python_ok, "suggested": suggested_paths(name),
           "source": SOURCES[name][0], "revision": SOURCES[name][1]}
    if not live:
        out["memory"] = memory_plan(name)
    return out


# --------------------------------------------------------------------------- host memory guard
#
# The EditaLive loader builds the text encoder, CLIP and the 14B transformer in host RAM before it
# offloads anything. On a workstation without enough RAM that froze the whole desktop (systemd-oomd
# killed the session). Isolated workers therefore always run under a hard memory ceiling: a cgroup
# (systemd-run --user --scope, MemoryMax + no swap) when the user manager can delegate one, and in
# every case a watchdog in the parent that stops the worker before the host runs out.

GIB = 2**30


def host_memory() -> dict[str, int]:
    """MemTotal / MemAvailable / SwapFree in bytes (zeros when /proc/meminfo is unreadable)."""
    out = {"total": 0, "available": 0, "swap_free": 0}
    keys = {"MemTotal": "total", "MemAvailable": "available", "SwapFree": "swap_free"}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in keys:
                out[keys[key]] = int(rest.split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return out


def _cgroup_usable() -> bool:
    """True when a transient user scope with a memory limit can actually be created (checked once)."""
    global _CGROUP
    if _CGROUP is None:
        exe = shutil.which("systemd-run")
        try:
            _CGROUP = bool(exe) and subprocess.run(
                [exe, "--user", "--scope", "--quiet", "--collect", "-p", "MemoryMax=64M", "-p", "MemorySwapMax=0", "true"],
                capture_output=True, timeout=10, check=False).returncode == 0
        except (OSError, subprocess.SubprocessError):
            _CGROUP = False
    return _CGROUP


_CGROUP: bool | None = None


def memory_plan(name: str, *, probe: bool = False) -> dict:
    """The ceiling a worker for ``name`` runs under and how it is enforced."""
    cfg = read_json(settings_path()).get(name, {})
    mem = host_memory()
    reserve = max(6 * GIB, int(mem["total"] * 0.15))  # what the desktop keeps no matter what the worker does
    automatic = max(0, mem["total"] - reserve)
    configured = cfg.get("memory_limit_gb")
    limit = min(int(float(configured) * GIB), mem["total"]) if configured else automatic
    floor = max(2 * GIB, int(mem["total"] * 0.05))  # host MemAvailable below this stops the worker
    cgroup = _cgroup_usable() if probe else (_CGROUP if _CGROUP is not None else bool(shutil.which("systemd-run")))
    return {"limit_gb": round(limit / GIB, 1), "limit_bytes": limit, "floor_bytes": floor,
            "total_gb": round(mem["total"] / GIB, 1), "available_gb": round(mem["available"] / GIB, 1),
            "configured": bool(configured), "method": "cgroup + watchdog" if cgroup else "watchdog"}


def contain(command: list[str], limit_bytes: int) -> list[str]:
    """Wrap ``command`` in a memory-limited transient scope when the user manager supports it."""
    if limit_bytes <= 0 or not _cgroup_usable():
        return command
    return [shutil.which("systemd-run"), "--user", "--scope", "--quiet", "--collect",
            "-p", f"MemoryMax={limit_bytes}", "-p", "MemorySwapMax=0", "--", *command]


def group_rss(pgid: int) -> int:
    """Resident memory of every process in the process group, in bytes."""
    total = 0
    page = os.sysconf("SC_PAGE_SIZE")
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        try:
            fields = Path(entry.path, "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid:  # pgrp
                total += int(fields[21]) * page  # rss
        except (OSError, ValueError, IndexError):
            continue
    return total


class MemoryGuard:
    """Polled by the render loop; ``check`` returns the reason to stop the worker, or None."""

    def __init__(self, pgid: int, limit_bytes: int, floor_bytes: int, *, interval: float = 1.0,
                 rss=group_rss, meminfo=host_memory, clock=time.monotonic) -> None:
        self.pgid, self.limit, self.floor, self.interval = pgid, limit_bytes, floor_bytes, interval
        self._rss, self._meminfo, self._clock = rss, meminfo, clock
        self._next = 0.0
        self.rss = self.peak = 0
        self.available = 0

    def check(self) -> str | None:
        now = self._clock()
        if now < self._next:
            return None
        self._next = now + self.interval
        self.rss = self._rss(self.pgid)
        self.peak = max(self.peak, self.rss)
        self.available = self._meminfo()["available"]
        if self.limit and self.rss > self.limit:
            return (f"memory guard: the worker reached {self.rss / GIB:.1f} GiB of host RAM "
                    f"(limit {self.limit / GIB:.1f} GiB) and was stopped before the desktop ran out of memory")
        if self.available and self.available < self.floor:
            return (f"memory guard: only {self.available / GIB:.1f} GiB of host RAM was left "
                    f"(worker at {self.rss / GIB:.1f} GiB), so the worker was stopped")
        return None


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
    from ...ipc import state_dir
    from ...models import verify_model

    status = backend_status(name)
    if status.get("disabled_reason"):
        raise EngineUnavailable(f"{name} is disabled: {status['disabled_reason']} "
                                f"Re-enable it under the memory guard with: deepfake experimental enable --backend {name}")
    if not status["available"]:
        raise EngineUnavailable(f"{name}: {status['status']}. Install weights with deepfake models install {name} --yes "
                                f"and configure its environment with deepfake experimental configure --backend {name}.")

    def verification_progress(filename, done, total):
        if session.stop_requested():
            raise InterruptedError("Render cancelled during checkpoint verification")
        session.bus.update(state="starting", loading_stage="Verifying checkpoints", model_file=filename,
                           model_bytes_verified=done, model_bytes_total=total)

    # Files unchanged since their last SHA-256 check are trusted; anything new or modified is hashed now.
    verification = verify_model(name, progress=verification_progress, quick=True)
    if not verification["ok"]:
        bad = ", ".join((verification["missing"] + verification["corrupt"])[:5])
        raise EngineUnavailable(f"{name}: checkpoint verification failed ({bad}). Repair the model in Models.")
    cfg = read_json(settings_path())[name]
    checkout, python = Path(cfg["checkout"]), Path(cfg["python"])
    if _revision(checkout) != SOURCES[name][1]:
        raise EngineUnavailable("Worker source revision changed; restore the pinned checkout before rendering")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = memory_plan(name, probe=True)
    command = contain(worker_command(name, checkout, python, reference, video, output_dir, prompt=prompt, fps=fps),
                      plan["limit_bytes"])
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
    # The log lives in the state directory so the control center can show it while the worker runs
    # and after the temporary render directory is gone.
    log_path = state_dir() / "experimental-worker.log"
    session.bus.update(state="processing", loading_stage="Loading models", worker_log=str(log_path), backend=name,
                       engine="experimental", memory_limit_gb=plan["limit_gb"], memory_guard=plan["method"],
                       model_file=None)
    started = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"== {name} worker, memory ceiling {plan['limit_gb']} GiB ({plan['method']})\n")
        log.flush()
        process = subprocess.Popen(command, cwd=checkout, env=env, stdout=log, stderr=log, start_new_session=True)
        guard = MemoryGuard(process.pid, plan["limit_bytes"], plan["floor_bytes"])
        try:
            while process.poll() is None:
                if session.stop_requested():
                    raise InterruptedError("Render cancelled")
                tripped = guard.check()
                if tripped:
                    log.write(tripped + "\n")
                    raise EngineUnavailable(f"{name}: {tripped}. Close other applications or lower the resolution; "
                                            f"the ceiling is set with deepfake experimental enable --backend {name} "
                                            "--memory-limit-gb N.")
                session.bus.update(state="processing", uptime_s=round(time.monotonic() - started, 1),
                                   loading_stage=_stage_from_log(log_path),
                                   worker_rss_gb=round(guard.rss / GIB, 1), host_available_gb=round(guard.available / GIB, 1))
                time.sleep(0.2)
            if process.returncode:
                tail = log_path.read_text(errors="replace")[-2500:]
                session.bus.update(worker_metrics=read_json(output_dir / "generated.metrics.json"))
                killed = process.returncode in (-signal.SIGKILL, 137)
                hint = (f" The kernel stopped it at the {plan['limit_gb']} GiB memory ceiling; the desktop was protected."
                        if killed else "")
                raise EngineUnavailable(f"{name} worker exited {process.returncode}.{hint}\n{tail}")
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
    return videos[0], {"worker_wall_s": time.monotonic() - started, "backend": name,
                       "host_rss_peak_gb": round(guard.peak / GIB, 1), **metrics}


def _stage_from_log(log_path: Path) -> str:
    """A coarse, honest stage label: the last non-empty line the worker printed."""
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 600))
            lines = [line.strip() for line in f.read().decode(errors="replace").replace("\r", "\n").splitlines() if line.strip()]
    except OSError:
        return "Running worker"
    return ("Worker: " + lines[-1][-110:]) if len(lines) > 1 else "Loading models"
