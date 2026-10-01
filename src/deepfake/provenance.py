"""C2PA signing AFTER final encoding; validation and trust are separate facts."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from . import __version__
from .paths import config_home

DIGITAL_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"


def signing_paths() -> tuple[Path, Path]:
    root = config_home() / "provenance"
    return (
        Path(os.environ.get("DEEPFAKE_C2PA_CERT", str(root / "certificate.pem"))),
        Path(os.environ.get("DEEPFAKE_C2PA_KEY", str(root / "private-key.pem"))),
    )


def available() -> bool:
    return importlib.util.find_spec("c2pa") is not None and importlib.util.find_spec("cryptography") is not None


def ensure_local_signer() -> tuple[Path, Path]:
    """Create a local development identity, explicitly NOT a platform-trusted signer."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    cert_path, key_path = signing_paths()
    if cert_path.is_file() and key_path.is_file():
        return cert_path, key_path
    if os.environ.get("DEEPFAKE_C2PA_CERT") or os.environ.get("DEEPFAKE_C2PA_KEY"):
        raise RuntimeError("configured C2PA signing certificate/key missing; will not replace them")
    cert_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if cert_path.exists() or key_path.exists():
        raise RuntimeError("incomplete local signing identity; refusing to overwrite it")
    now = datetime.now(UTC)
    root_key, leaf_key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Deepfake local development CA")])
    leaf_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Deepfake local signer - not publicly trusted")])
    root = (
        x509.CertificateBuilder()
        .subject_name(root_name)
        .issuer_name(root_name)
        .public_key(root_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, True, True, False, False), True)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), False)
        .sign(root_key, hashes.SHA256())
    )
    leaf = (
        x509.CertificateBuilder()
        .subject_name(leaf_name)
        .issuer_name(root_name)
        .public_key(leaf_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
        .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), True)
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), False)
        .sign(root_key, hashes.SHA256())
    )
    key_bytes = leaf_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    # Exclusive creation and 0600 prevent overwriting/leaking the signing key.
    with os.fdopen(os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        f.write(key_bytes)
    cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM) + root.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


def manifest_definition(backend: str, framegen: str | None, *, parent: bool = False) -> dict:
    actions = (
        [{"action": "c2pa.opened", "parameters": {"ingredientIds": ["original-input"]}}]
        if parent
        else [{"action": "c2pa.created", "digitalSourceType": DIGITAL_SOURCE}]
    )
    actions.append(
        {
            "action": "c2pa.edited",
            "digitalSourceType": DIGITAL_SOURCE,
            "description": "AI face replacement with target occlusion preservation",
        }
    )
    if framegen:
        actions.append(
            {
                "action": "c2pa.edited",
                "digitalSourceType": DIGITAL_SOURCE,
                "description": "Motion-based frame interpolation: " + framegen,
            }
        )
    actions.append({"action": "c2pa.transcoded", "description": "Final video encoding before C2PA signing"})
    assertions = [
        {"label": "c2pa.actions", "data": {"actions": actions}},
        {
            "label": "c2pa.ai-disclosure",
            "data": {"modelType": "c2pa.types.model.pytorch", "modelName": backend, "scientificDomain": ["cs.CV"]},
        },
    ]
    # NVOFA is optical flow + deterministic warp; never call it a generative AI model.
    return {
        "claim_generator_info": [{"name": "deepfake", "version": __version__}],
        "title": "AI-modified video",
        "format": "video/mp4",
        "assertions": assertions,
    }


def inspect(path: Path, *, trust_cert: Path | None = None) -> dict:
    import c2pa

    settings = {"verify": {"verify_cert_anchors": False}}
    if trust_cert:
        settings = {"verify": {"verify_cert_anchors": True}, "trust": {"trust_anchors": trust_cert.read_text()}}
    try:
        with (
            c2pa.Context(c2pa.Settings.from_dict(settings)) as context,
            c2pa.Reader(str(path), context=context) as reader,
        ):
            data = json.loads(reader.json())
    except c2pa.C2paError as e:
        return {
            "c2pa_present": False,
            "signature_valid": False,
            "asset_binding_valid": False,
            "trusted": False,
            "reason": str(e),
            "invisible_watermark": "not checked",
        }
    active = data.get("active_manifest")
    manifest = data.get("manifests", {}).get(active, {})
    results = data.get("validation_results", {}).get("activeManifest", {})
    successes = {x.get("code") for x in results.get("success", [])}
    failures = results.get("failure", [])
    # SDK validates the signature and content binding separately. Never infer
    # integrity merely from a manifest's existence or from the AI disclosure text.
    signature = "claimSignature.validated" in successes
    binding = any(
        x in successes for x in ("assertion.bmffHash.match", "assertion.dataHash.match", "assertion.boxesHash.match")
    )
    integrity_failures = [f for f in failures if f.get("code") != "signingCredential.untrusted"]
    valid = signature and binding and not integrity_failures
    assertions = manifest.get("assertions", [])
    actions = [
        a
        for assertion in assertions
        if assertion.get("label", "").startswith("c2pa.actions")
        for a in assertion.get("data", {}).get("actions", [])
    ]
    return {
        "c2pa_present": bool(active),
        "signature_valid": signature,
        "asset_binding_valid": binding,
        "valid": valid,
        "trusted": "signingCredential.trusted" in successes,
        "validation_state": data.get("validation_state"),
        "failures": failures,
        "digital_source_types": sorted({a["digitalSourceType"] for a in actions if "digitalSourceType" in a}),
        "actions": actions,
        "tool": manifest.get("claim_generator_info", []),
        "invisible_watermark": "not checked",
        "validation_results": data.get("validation_results", {}),
    }


def sign_export(
    path: Path,
    *,
    original: Path | None = None,
    backend: str = "AlphaFace",
    framegen: str | None = None,
    watermark: dict | None = None,
) -> dict:
    import c2pa
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    cert, key = ensure_local_signer()
    if key.stat().st_mode & 0o077:
        raise RuntimeError("C2PA private key permissions must be 0600")
    private_key = serialization.load_pem_private_key(key.read_bytes(), password=None)

    def callback(data: bytes) -> bytes:
        return private_key.sign(data, ec.ECDSA(hashes.SHA256()))

    fd, tmp = tempfile.mkstemp(prefix=".deepfake-signed-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    candidate = Path(tmp)
    candidate.unlink()  # SDK requires a new destination
    try:
        definition = manifest_definition(backend, framegen, parent=original is not None)
        if watermark and watermark.get("present"):
            definition["assertions"].append({"label": "org.r3dg0d.deepfake.watermark", "data": watermark})
        with c2pa.Context(c2pa.Settings.from_dict({"verify": {"verify_cert_anchors": False}})) as context:
            with c2pa.Signer.from_callback(callback, c2pa.C2paSigningAlg.ES256, cert.read_text(), None) as signer:
                with c2pa.Builder(definition, context) as builder:
                    if original:
                        with original.open("rb") as source:
                            builder.add_ingredient(
                                {
                                    "title": "Original input",
                                    "relationship": "parentOf",
                                    "instance_id": "original-input",
                                },
                                "video/mp4",
                                source,
                            )
                    builder.sign_file(str(path), str(candidate), signer)
        verified = inspect(candidate)
        if not verified.get("valid"):
            raise RuntimeError("C2PA export verification failed: " + json.dumps(verified))
        os.replace(candidate, path)
        # Verify the actual final filename too, after atomic replacement.
        return inspect(path)
    finally:
        candidate.unlink(missing_ok=True)
