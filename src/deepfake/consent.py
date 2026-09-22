"""Consent / disclosed-synthetic-media gates."""

from __future__ import annotations

import os
import sys

CONSENT_TEXT = """\
This tool produces SYNTHETIC face media for research, VFX, avatars, filmmaking,
and consenting demos with disclosed synthetic media.

You must ONLY use source identities and target footage where you have consent
(or own the rights), and you must disclose synthetic media when required by law
or platform policy.

This tool makes NO anonymity claims. Misuse for non-consensual deepfakes,
impersonation, fraud, or harassment is prohibited.
"""

ACK_ENV = "DEEPFAKE_CONSENT_ACK"


def require_consent(*, ack: bool = False, watermark: bool = True) -> None:
    """Exit unless user acknowledges consent framing.

    Pass ``--consent-ack`` or set DEEPFAKE_CONSENT_ACK=1.
    Watermark/disclosure is recommended and on by default for live outputs.
    """
    env_ack = os.environ.get(ACK_ENV, "").strip() in ("1", "true", "yes", "YES")
    if ack or env_ack:
        if not watermark:
            print(
                "warning: synthetic-media watermark disabled; ensure you disclose by other means",
                file=sys.stderr,
            )
        return
    print(CONSENT_TEXT, file=sys.stderr)
    print(
        "Re-run with --consent-ack (or export DEEPFAKE_CONSENT_ACK=1) after reading the above.",
        file=sys.stderr,
    )
    raise SystemExit(2)
