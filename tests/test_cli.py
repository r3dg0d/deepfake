from click.testing import CliRunner

from deepfake.cli import main


def test_help():
    runner = CliRunner()
    r = runner.invoke(main, ["--help"])
    assert r.exit_code == 0
    assert "webcam" in r.output.lower()
    assert "frame generation" in r.output.lower() or "Frame generation" in r.output


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
        ["config", "show"],
    ):
        r = runner.invoke(main, list(args))
        assert r.exit_code == 0, (args, r.output, r.exception)
    r = runner.invoke(main, ["doctor"])
    assert "AlphaFace" in r.output and "FrameGen" in r.output and "Quickshell" in r.output


def test_webcam_requires_consent_and_source():
    runner = CliRunner()
    r = runner.invoke(main, ["webcam", "--no-widget"])
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
    assert "-f" in r.output or "--source" in r.output


def test_source_is_remembered(tmp_path, monkeypatch):
    from pathlib import Path

    from deepfake import cli as cli_mod
    from deepfake.cli import _load_settings, _resolve_source

    face = tmp_path / "face.jpg"
    face.write_bytes(b"\xff\xd8\xff")
    cfg = tmp_path / "settings.json"
    monkeypatch.setattr(cli_mod, "_settings_path", lambda: cfg)
    assert _resolve_source(face, None) == face.resolve()
    assert _load_settings()["source"] == str(face.resolve())
    assert _resolve_source(None, None) == face.resolve()
    _ = Path


def test_virtualcam_without_loopback_explains(monkeypatch):
    from deepfake import devices

    monkeypatch.setattr(devices, "find_loopback_device", lambda *a, **k: None)
    r = CliRunner().invoke(main, ["virtualcam", "--no-widget", "--no-frame-gen"])
    assert r.exit_code != 0 and "v4l2loopback" in r.output


def test_frame_gen_defaults_to_auto():
    from deepfake.session import resolve_frame_gen

    fg = resolve_frame_gen(
        cli_value=None,
        no_frame_gen=False,
        output_fps=None,
        source_fps=30.0,
        preset="balanced",
        backend="rife",
        variant=None,
    )
    assert fg.enabled is True
    assert fg.factor == 2


def test_video_cli_accepts_dash_i():
    r = CliRunner().invoke(main, ["video", "--help"])
    assert r.exit_code == 0
    assert "-i" in r.output and "-f" in r.output
