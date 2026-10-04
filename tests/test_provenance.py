import shutil
import subprocess

import pytest
from click.testing import CliRunner

from deepfake.cli import main
from deepfake.provenance import (
    format_inspection,
    generate_local_dev_cert,
    inspect_file,
    sign_finished_file,
)


def _need(*tools: str) -> None:
    missing = [name for name in tools if shutil.which(name) is None]
    if missing:
        pytest.skip("needs " + ", ".join(missing) + " on PATH")


def test_incomplete_env_does_not_sign(tmp_path, monkeypatch, capsys):
    target = tmp_path / "out.mp4"
    target.write_bytes(b"not a video")
    before = target.read_bytes()
    monkeypatch.setenv("DEEPFAKE_C2PA_CERT", str(tmp_path / "missing.pem"))
    monkeypatch.delenv("DEEPFAKE_C2PA_KEY", raising=False)
    assert sign_finished_file(target, frame_interpolation=False) is False
    assert target.read_bytes() == before
    assert "DEEPFAKE_C2PA_KEY" in capsys.readouterr().err


def test_sign_temp_cert_and_inspect(tmp_path, monkeypatch):
    _need("ffmpeg", "c2patool", "openssl")
    unsigned = tmp_path / "plain.mp4"
    signed = tmp_path / "signed.mp4"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=red:s=64x64:d=0.2",
        "-r", "10", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(unsigned),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        pytest.skip("ffmpeg could not write a tiny mp4: " + (proc.stderr or "")[:200])
    signed.write_bytes(unsigned.read_bytes())
    cert, key = generate_local_dev_cert(tmp_path / "dev-cert")
    monkeypatch.setenv("DEEPFAKE_C2PA_CERT", str(cert))
    monkeypatch.setenv("DEEPFAKE_C2PA_KEY", str(key))

    assert sign_finished_file(signed, frame_interpolation=True) is True
    inspection = inspect_file(signed)
    text = format_inspection(inspection)
    assert inspection.present
    assert inspection.validation_state == "Valid"
    assert "not a public trust chain" in text
    assert "local dev cert" in text
    assert "face replacement" in text
    assert "frame interpolation" in text
    assert "deepfake" in text
    assert "SynthID" not in text
    assert "✓" not in text
    assert any(item.startswith("claimSignature.mismatch") for item in inspection.failures) is False

    plain = format_inspection(inspect_file(unsigned))
    assert plain == "C2PA: no manifest"

    runner = CliRunner()
    shown = runner.invoke(main, ["provenance", "inspect", str(signed)])
    assert shown.exit_code == 0, shown.output
    assert "validation: Valid" in shown.output
    assert "face replacement" in shown.output
    assert "SynthID" not in shown.output
    empty = runner.invoke(main, ["provenance", "inspect", str(unsigned)])
    assert empty.exit_code == 0
    assert empty.output.strip() == "C2PA: no manifest"
