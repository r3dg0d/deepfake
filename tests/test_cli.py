from click.testing import CliRunner

from deepfake.cli import main


def test_help():
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "face-swap" in r.output.lower() or "webcam" in r.output.lower()


def test_subcommand_helps():
    runner = CliRunner()
    for args in (
        ["webcam", "--help"],
        ["video", "--help"],
        ["virtualcam", "--help"],
        ["devices", "--help"],
        ["benchmark", "--help"],
        ["models", "--help"],
        ["models", "list"],
    ):
        r = runner.invoke(main, list(args))
        assert r.exit_code == 0, (args, r.output)


def test_webcam_requires_consent_and_source():
    runner = CliRunner()
    r = runner.invoke(main, ["webcam"])
    assert r.exit_code != 0


def test_models_list_json():
    runner = CliRunner()
    r = runner.invoke(main, ["models", "list", "--json"])
    assert r.exit_code == 0
    assert "alphaface" in r.output
    assert "inswapper" in r.output


def test_help_command():
    runner = CliRunner()
    r = runner.invoke(main, ["help"])
    assert r.exit_code == 0 and "virtualcam" in r.output
    r = runner.invoke(main, ["help", "webcam"])
    assert r.exit_code == 0 and "--frame-gen" in r.output
    assert "--consent-ack" not in r.output


def test_source_is_remembered(tmp_path):
    from pathlib import Path

    from deepfake.cli import _load_settings, _resolve_source

    face = tmp_path / "face.jpg"
    face.write_bytes(b"\xff\xd8\xff")
    assert _resolve_source(face, None) == face.resolve()
    assert _load_settings()["source"] == str(face.resolve())
    assert _resolve_source(None, None) == face.resolve()  # no flag needed next time
    _ = Path


def test_virtualcam_without_loopback_explains(monkeypatch):
    from deepfake import devices

    monkeypatch.setattr(devices, "find_loopback_device", lambda *a, **k: None)
    r = CliRunner().invoke(main, ["virtualcam"])
    assert r.exit_code != 0 and "v4l2loopback" in r.output
