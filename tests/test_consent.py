import io

import pytest

from deepfake.consent import is_acknowledged, require_consent


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.delenv("DEEPFAKE_CONSENT_ACK", raising=False)


def test_non_interactive_without_ack_blocks(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as ei:
        require_consent()
    assert ei.value.code == 2
    assert not is_acknowledged()


def test_asked_once_then_remembered(monkeypatch):
    stdin = io.StringIO("y\n")
    stdin.isatty = lambda: True
    monkeypatch.setattr("sys.stdin", stdin)
    require_consent()
    assert is_acknowledged()
    # second run: no prompt at all (stdin is empty and non-interactive)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    require_consent()


def test_declining_does_not_remember(monkeypatch):
    stdin = io.StringIO("n\n")
    stdin.isatty = lambda: True
    monkeypatch.setattr("sys.stdin", stdin)
    with pytest.raises(SystemExit):
        require_consent()
    assert not is_acknowledged()


def test_flag_and_env_preaccept(monkeypatch):
    require_consent(ack=True)
    assert is_acknowledged()


def test_env(monkeypatch):
    monkeypatch.setenv("DEEPFAKE_CONSENT_ACK", "1")
    require_consent()
