"""Local IPC for the Deepfake desktop widget (JSON state + control files).

No network server. The runtime publishes structured session state under
``~/.local/state/deepfake/``; the Quickshell widget reads it. Controls from
the widget land in ``control.json`` and are drained by the session.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

STATE_DIR_NAME = "deepfake"


def state_dir() -> Path:
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    d = root / STATE_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def session_path() -> Path:
    return state_dir() / "session.json"


def preview_json_path() -> Path:
    return state_dir() / "preview.json"


def control_path() -> Path:
    return state_dir() / "control.json"


def pid_path() -> Path:
    return state_dir() / "session.pid"


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=2, sort_keys=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8") or "{}")
    except (OSError, ValueError):
        return {}


class SessionBus:
    """Publishes live session state and drains widget control commands."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._state: dict[str, Any] = {
            "session_id": session_id,
            "active": False,
            "mode": None,
            "state": "init",
            "pid": os.getpid(),
            "ts": time.time(),
        }
        write_json_atomic(session_path(), self._state)
        pid_path().write_text(f"{os.getpid()}\n{session_id}\n", encoding="utf-8")

    def update(self, **fields: Any) -> None:
        self._state.update(fields)
        self._state["ts"] = time.time()
        self._state["active"] = self._state.get("state") in (
            "running",
            "starting",
            "paused",
            "processing",
        )
        write_json_atomic(session_path(), self._state)
        # Keep preview.json in sync for older overlay shells.
        preview = {
            "active": bool(self._state.get("active")),
            "fps": self._state.get("output_fps") or self._state.get("alphaface_fps") or 0,
            "frames": self._state.get("frames") or 0,
            "width": self._state.get("width") or 0,
            "height": self._state.get("height") or 0,
            "uptime_s": self._state.get("uptime_s") or 0,
            "title": "deepfake",
            "ts": self._state["ts"],
            "session": self._state,
        }
        write_json_atomic(preview_json_path(), preview)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    def drain_controls(self) -> list[dict[str, Any]]:
        path = control_path()
        if not path.is_file():
            return []
        data = read_json(path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        if not data:
            return []
        if isinstance(data, list):
            return [c for c in data if isinstance(c, dict)]
        if isinstance(data, dict) and "cmd" in data:
            return [data]
        if isinstance(data, dict) and "commands" in data:
            return [c for c in data["commands"] if isinstance(c, dict)]
        return []

    def close(self, *, state: str = "stopped") -> None:
        self.update(state=state, active=False)
        try:
            if pid_path().is_file():
                pid_path().unlink()
        except OSError:
            pass
