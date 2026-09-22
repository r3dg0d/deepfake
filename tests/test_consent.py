import pytest

from deepfake.consent import require_consent


def test_consent_blocks(monkeypatch):
    monkeypatch.delenv("DEEPFAKE_CONSENT_ACK", raising=False)
    with pytest.raises(SystemExit) as ei:
        require_consent(ack=False)
    assert ei.value.code == 2


def test_consent_ack_flag(monkeypatch):
    monkeypatch.delenv("DEEPFAKE_CONSENT_ACK", raising=False)
    require_consent(ack=True, watermark=True)


def test_consent_env(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_CONSENT_ACK", "1")
    require_consent(ack=False)
