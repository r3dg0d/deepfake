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
