"""One-time consent / disclosed-synthetic-media acknowledgement.

The notice is shown the first time deepfake runs interactively; once
accepted it is remembered in ``$XDG_CONFIG_HOME/deepfake/consent.json`` and
never asked again. Scripts can pre-accept with ``DEEPFAKE_CONSENT_ACK=1`` or
the (hidden) ``--consent-ack`` flag. The disclosure watermark stays on by
default either way.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime

from .paths import config_home

CONSENT_TEXT = """\
deepfake produces SYNTHETIC face media for research, VFX, avatars, filmmaking
and demos with people who have agreed to it.

Only use faces and footage you have consent for (or own the rights to), and
disclose synthetic media where law or platform rules require it. Impersonation,
fraud, harassment and non-consensual deepfakes are prohibited. A
"SYNTHETIC MEDIA" watermark is on by default (--no-watermark turns it off).
"""

ACK_ENV = "DEEPFAKE_CONSENT_ACK"
NOTICE_VERSION = 1


def _ack_file():
    return config_home() / "consent.json"


def is_acknowledged() -> bool:
    if os.environ.get(ACK_ENV, "").strip().lower() in ("1", "true", "yes"):
        return True
    try:
        data = json.loads(_ack_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return int(data.get("version", 0)) >= NOTICE_VERSION


def record_acknowledgement() -> None:
    path = _ack_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"version": NOTICE_VERSION, "accepted": datetime.now(UTC).isoformat(timespec="seconds")})
        + "\n",
        encoding="utf-8",
    )


def require_consent(*, ack: bool = False, watermark: bool = True) -> None:
    """Pass silently once acknowledged; otherwise ask once (interactive) and remember."""
    if ack and not is_acknowledged():
        record_acknowledgement()
    if not (ack or is_acknowledged()):
        print(CONSENT_TEXT, file=sys.stderr)
        if not sys.stdin.isatty():
            print(
                f"Run deepfake once in a terminal to accept, or set {ACK_ENV}=1 for scripts.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            answer = input("Accept and continue? This is asked only once. [y/N] ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            raise SystemExit(2)
        record_acknowledgement()
    if not watermark:
        print("warning: synthetic-media watermark disabled; disclose by other means", file=sys.stderr)
