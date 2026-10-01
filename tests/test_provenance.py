import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake.provenance import DIGITAL_SOURCE, inspect, manifest_definition, sign_export


def test_disclosure_classifies_modification_without_claiming_synthid():
    definition = manifest_definition("AlphaFace", "nvof")
    text = json.dumps(definition)
    assert DIGITAL_SOURCE in text
    assert "SynthID" not in text
    assert "c2pa.ai-disclosure" in text


def test_signed_mp4_verifies_and_tampering_fails(tmp_path):
    pytest.importorskip("c2pa")
    pytest.importorskip("cryptography")
    original = tmp_path / "original.mp4"
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(10):
        writer.write(np.full((64, 64, 3), i * 20, np.uint8))
    writer.release()
    target = tmp_path / "signed.mp4"
    target.write_bytes(original.read_bytes())
    result = sign_export(target, original=original)
    assert result["signature_valid"] and result["asset_binding_valid"] and result["valid"]
    assert not result["trusted"]
    assert inspect(original)["c2pa_present"] is False
    encoded = bytearray(target.read_bytes())
    pos = encoded.index(b"mdat") + 20
    encoded[pos] ^= 1
    corrupted = tmp_path / "corrupt.mp4"
    corrupted.write_bytes(encoded)
    assert not inspect(corrupted).get("valid", False)
    from deepfake.provenance import signing_paths

    cert, key = signing_paths()
    assert key.stat().st_mode & 0o077 == 0
    assert Path(cert).is_file()


def test_failed_signing_does_not_publish_or_replace_requested_output(tmp_path, monkeypatch):
    pytest.importorskip("c2pa")
    pytest.importorskip("cryptography")
    from click.testing import CliRunner

    from deepfake import cli, provenance, realtime

    original = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(original), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    writer.write(np.zeros((64, 64, 3), np.uint8))
    writer.release()
    face = tmp_path / "face.png"
    cv2.imwrite(str(face), np.zeros((64, 64, 3), np.uint8))
    requested = tmp_path / "requested.mp4"
    requested.write_bytes(b"previous export")
    monkeypatch.setenv("DEEPFAKE_CONSENT_ACK", "1")

    def fake_sink(**kwargs):
        kwargs["output_video"].write_bytes(original.read_bytes())
        return object()

    def fake_render(_input, _source, factory, _cfg, _fg, **kwargs):
        factory(64, 64, 10)
        return {"source_frames": 1, "swap_backend": "alphaface"}

    def reject(*args, **kwargs):
        raise RuntimeError("signature verification failed")

    monkeypatch.setattr(cli, "create_sink", fake_sink)
    monkeypatch.setattr(realtime, "run_offline", fake_render)
    monkeypatch.setattr(provenance, "sign_export", reject)
    result = CliRunner().invoke(
        cli.main,
        [
            "video",
            "-i",
            str(original),
            "-f",
            str(face),
            "-o",
            str(requested),
            "--provenance",
            "c2pa",
            "--frame-gen",
            "off",
            "--no-widget",
            "--no-preview",
        ],
    )
    assert result.exit_code != 0
    assert "signature verification failed" in result.output
    assert requested.read_bytes() == b"previous export"
    assert not list(tmp_path.glob(".deepfake-encoding-*"))
