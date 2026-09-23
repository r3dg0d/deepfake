"""Quickshell desktop widget lifecycle for Deepfake sessions."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .ipc import pid_path, state_dir


def widget_shell_dir() -> Path | None:
    """Resolve the Quickshell project directory that contains shell.qml."""
    env = os.environ.get("DEEPFAKE_PREVIEW_DIR") or os.environ.get("DEEPFAKE_WIDGET_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    here = Path(__file__).resolve()
    # repo overlay shipped with the package source tree
    candidates.append(here.parents[2] / "overlays" / "deepfake-preview" / "quickshell")
    candidates.append(Path.home() / "Projects" / "deepfake-preview" / "quickshell")
    candidates.append(Path.home() / "Projects" / "deepfake" / "overlays" / "deepfake-preview" / "quickshell")
    for c in candidates:
        if (c / "shell.qml").is_file():
            return c
    return None


def find_qs() -> str | None:
    env = os.environ.get("AMBXST_QS") or os.environ.get("DEEPFAKE_QS")
    if env and Path(env).is_file() and os.access(env, os.X_OK):
        return env
    which = shutil.which("qs")
    if which:
        return which
    # common NixOS store path pattern used on this workstation
    for p in Path("/nix/store").glob("*-quickshell-*/bin/qs"):
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def quickshell_available() -> tuple[bool, str]:
    qs = find_qs()
    shell = widget_shell_dir()
    if not qs:
        return False, "qs (Quickshell) not found on PATH"
    if shell is None:
        return False, "Quickshell widget shell.qml not found"
    return True, f"{qs} · {shell}"


def _read_pidfile() -> tuple[int | None, str | None]:
    p = pid_path()
    if not p.is_file():
        return None, None
    try:
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        pid = int(lines[0]) if lines else None
        sid = lines[1] if len(lines) > 1 else None
        return pid, sid
    except (OSError, ValueError):
        return None, None


def cleanup_orphans(*, current_session: str | None = None) -> None:
    """Kill a leftover Quickshell instance tied to a dead Deepfake session."""
    qs = find_qs()
    shell = widget_shell_dir()
    pid, sid = _read_pidfile()
    if pid is not None and sid != current_session:
        try:
            os.kill(pid, 0)
        except OSError:
            # process gone — drop pidfile
            try:
                pid_path().unlink(missing_ok=True)
            except OSError:
                pass
        else:
            # live foreign process: leave it (another intentional session)
            pass
    if qs and shell:
        # qs kill for this config path (idempotent)
        subprocess.run([qs, "kill", "-p", str(shell)], capture_output=True, check=False)


class WidgetHandle:
    """Owns one Quickshell process for a Deepfake session."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.shell_dir: Path | None = None
        self.qs: str | None = None
        self.started = False
        self.message: str | None = None

    def start(self) -> bool:
        ok, msg = quickshell_available()
        if not ok:
            self.message = msg
            return False
        self.qs = find_qs()
        self.shell_dir = widget_shell_dir()
        assert self.qs and self.shell_dir
        cleanup_orphans()
        state_dir()  # ensure exists
        # Detach: qs -n -d -p <dir>
        try:
            self.proc = subprocess.Popen(
                [self.qs, "-n", "-d", "-p", str(self.shell_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.started = True
            self.message = f"Quickshell widget started ({self.shell_dir})"
            time.sleep(0.15)
            return True
        except OSError as e:
            self.message = f"Quickshell failed to start: {e}"
            return False

    def stop(self) -> None:
        qs = self.qs or find_qs()
        shell = self.shell_dir or widget_shell_dir()
        if qs and shell:
            subprocess.run([qs, "kill", "-p", str(shell)], capture_output=True, check=False)
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except OSError:
                try:
                    self.proc.terminate()
                except OSError:
                    pass
        self.proc = None
        self.started = False
