"""C2PA Content Credentials for a finished output file.

Signing uses the Content Authenticity Initiative `c2patool` binary (c2pa-rs).
Webcam preview frames are not signed. A missing tool, a failed sign, or a
manifest that does not verify is a stderr warning, not a success.

Credentials come from DEEPFAKE_C2PA_CERT and DEEPFAKE_C2PA_KEY (PEM). If both
are unset, a local dev CA and end-entity cert are generated into the user
cache (or a directory the caller passes). That cert is not on a public trust
chain. c2patool 0.27.4 reports a cryptographically good signature as
claimSignature.mismatch when the certificate subject has no Organization
attribute (c2pa-rs issue 2262), so the generated subject includes one.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .paths import cache_home

_DIGITAL_SOURCE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
_LOCAL_ORG = "deepfake local dev"


class ProvenanceToolError(RuntimeError):
    """c2patool is not available, so nothing can be verified."""


@dataclass
class Inspection:
    present: bool
    validation_state: str | None = None
    signature: str = ""
    tool: str | None = None
    actions: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    library: str | None = None


def c2patool_path() -> str | None:
    override = os.environ.get("DEEPFAKE_C2PATOOL")
    if override:
        return override if Path(override).is_file() else None
    return shutil.which("c2patool")


def _warn(message: str) -> None:
    print(f"deepfake: {message}", file=sys.stderr)


def _run_c2patool(args: list[str]) -> subprocess.CompletedProcess[str]:
    tool = c2patool_path()
    if tool is None:
        raise ProvenanceToolError("c2patool not on PATH (or DEEPFAKE_C2PATOOL); cannot sign or verify")
    return subprocess.run([tool, *args], capture_output=True, text=True, check=False)


def generate_local_dev_cert(directory: Path) -> tuple[Path, Path]:
    """Write an ES256 end-entity key and a PEM chain (end-entity, then CA).

    The CA key stays in ``directory`` so the pair can be reused. It is a local
    dev credential, not a public trust anchor. Returns ``(cert_chain, private_key)``.
    """
    openssl = shutil.which("openssl")
    if openssl is None:
        raise ProvenanceToolError("openssl not on PATH; cannot create a local C2PA dev cert")
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    key_path = directory / "ee.key"
    chain_path = directory / "chain.pem"
    if key_path.is_file() and chain_path.is_file():
        return chain_path, key_path

    ca_key = directory / "ca.key"
    ca_crt = directory / "ca.crt"
    ee_crt = directory / "ee.crt"
    ee_csr = directory / "ee.csr"
    ext = directory / "ee.ext"
    ext.write_text("basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=emailProtection\n")

    def run(args: list[str]) -> None:
        proc = subprocess.run([openssl, *args], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise ProvenanceToolError(f"openssl failed: {detail[:300]}")

    # notBefore is backdated. A cert stamped "now" can still be not-yet-valid
    # for a second, and c2patool then refuses the signature.
    run(["genrsa", "-out", str(ca_key), "2048"])
    run(
        [
            "req", "-new", "-x509", "-key", str(ca_key), "-out", str(ca_crt),
            "-not_before", "20200101000000Z", "-not_after", "20300101000000Z",
            "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-subj", f"/C=US/O={_LOCAL_ORG}/CN=deepfake local dev CA",
        ]
    )
    run(["ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(key_path)])
    run(["req", "-new", "-key", str(key_path), "-out", str(ee_csr), "-subj", f"/C=US/O={_LOCAL_ORG}/CN={_LOCAL_ORG}"])
    run(
        [
            "x509", "-req", "-in", str(ee_csr), "-CA", str(ca_crt), "-CAkey", str(ca_key),
            "-CAcreateserial", "-out", str(ee_crt), "-extfile", str(ext),
            "-not_before", "20200101000000Z", "-not_after", "20300101000000Z",
        ]
    )
    chain_path.write_bytes(ee_crt.read_bytes() + b"\n" + ca_crt.read_bytes())
    for secret in (key_path, ca_key):
        os.chmod(secret, 0o600)
    for leftover in (ee_csr, ext):
        leftover.unlink(missing_ok=True)
    return chain_path, key_path


def resolve_credentials() -> tuple[Path, Path] | None:
    """Env pair if set, otherwise a generated local dev cert in the user cache."""
    cert = os.environ.get("DEEPFAKE_C2PA_CERT")
    key = os.environ.get("DEEPFAKE_C2PA_KEY")
    if cert or key:
        if not cert or not key:
            _warn("set both DEEPFAKE_C2PA_CERT and DEEPFAKE_C2PA_KEY, or neither; output left unsigned")
            return None
        cert_path, key_path = Path(cert), Path(key)
        if not cert_path.is_file() or not key_path.is_file():
            _warn("DEEPFAKE_C2PA_CERT / DEEPFAKE_C2PA_KEY is not a readable file; output left unsigned")
            return None
        return cert_path, key_path
    try:
        return generate_local_dev_cert(cache_home() / "c2pa")
    except ProvenanceToolError as exc:
        _warn(f"{exc}; output left unsigned")
        return None


def _actions(frame_interpolation: bool) -> list[dict]:
    agent = {"name": "deepfake", "version": __version__}
    actions = [
        {
            "action": "c2pa.edited",
            "softwareAgent": agent,
            "description": "face replacement",
            "digitalSourceType": _DIGITAL_SOURCE,
        }
    ]
    if frame_interpolation:
        actions.append(
            {
                "action": "c2pa.edited",
                "softwareAgent": agent,
                "description": "frame interpolation",
                "digitalSourceType": _DIGITAL_SOURCE,
            }
        )
    return actions


def _manifest(path: Path, cert: Path, key: Path, *, frame_interpolation: bool) -> dict:
    return {
        "alg": "es256",
        "private_key": str(key.resolve()),
        "sign_cert": str(cert.resolve()),
        "claim_generator": f"deepfake/{__version__}",
        "claim_generator_info": [{"name": "deepfake", "version": __version__}],
        "title": path.name,
        "assertions": [{"label": "c2pa.actions.v2", "data": {"actions": _actions(frame_interpolation)}}],
    }


def sign_finished_file(path: Path, *, frame_interpolation: bool = False) -> bool:
    """Attach a manifest after the file is fully written. True only if verify passed."""
    if not path.is_file():
        _warn(f"not signing {path}: not a finished file")
        return False
    creds = resolve_credentials()
    if creds is None:
        return False
    cert, key = creds
    tool_missing = c2patool_path() is None
    if tool_missing:
        _warn("c2patool not on PATH (or DEEPFAKE_C2PATOOL); output left unsigned")
        return False
    tmp = path.with_name(f".{path.stem}.c2pa-signing{path.suffix}")
    published = False
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            manifest_path = Path(handle.name)
            json.dump(_manifest(path, cert, key, frame_interpolation=frame_interpolation), handle)
        try:
            signed = _run_c2patool([str(path), "-m", str(manifest_path), "-o", str(tmp), "--force"])
        finally:
            manifest_path.unlink(missing_ok=True)
        if signed.returncode != 0 or not tmp.is_file():
            detail = (signed.stderr or signed.stdout or "c2patool sign failed").strip()
            _warn(f"C2PA sign failed; output left unsigned ({detail[:300]})")
            tmp.unlink(missing_ok=True)
            return False
        try:
            inspection = inspect_file(tmp)
        except ProvenanceToolError as exc:
            _warn(f"{exc}; output left unsigned")
            tmp.unlink(missing_ok=True)
            return False
        if not _signature_verified(inspection):
            why = ", ".join(inspection.failures) or inspection.validation_state or "not verified"
            _warn(f"C2PA manifest did not verify; output left unsigned ({why})")
            tmp.unlink(missing_ok=True)
            return False
        os.replace(tmp, path)
        published = True
        return True
    except ProvenanceToolError as exc:
        _warn(f"{exc}; output left unsigned")
        tmp.unlink(missing_ok=True)
        return False
    finally:
        if not published:
            tmp.unlink(missing_ok=True)


def _signature_verified(inspection: Inspection) -> bool:
    if not inspection.present:
        return False
    if inspection.validation_state not in {"Valid", "Trusted"}:
        return False
    return not any(item.startswith("claimSignature.mismatch") for item in inspection.failures)


def inspect_file(path: Path) -> Inspection:
    proc = _run_c2patool([str(path)])
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 and "No claim found" in stderr:
        return Inspection(present=False, signature="no manifest")
    payload = _json_object(stdout)
    if payload is None:
        if proc.returncode != 0:
            raise ProvenanceToolError(stderr[:300] or "c2patool could not read a manifest")
        return Inspection(present=False, signature="no manifest")
    return _inspection_from_report(payload)


def _json_object(text: str) -> dict | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(text[start:])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _inspection_from_report(report: dict) -> Inspection:
    manifests = report.get("manifests") or {}
    active_id = report.get("active_manifest")
    manifest = manifests.get(active_id) if isinstance(manifests, dict) else None
    if not isinstance(manifest, dict):
        return Inspection(present=False, signature="no manifest")
    failures: list[str] = []
    for status in report.get("validation_status") or []:
        if not isinstance(status, dict):
            continue
        code = str(status.get("code") or "")
        explanation = str(status.get("explanation") or "")
        if code:
            failures.append(f"{code}: {explanation}".rstrip(": "))
    state = report.get("validation_state")
    state_s = str(state) if state else None
    sig = manifest.get("signature_info") if isinstance(manifest.get("signature_info"), dict) else {}
    return Inspection(
        present=True,
        validation_state=state_s,
        signature=_signature_line(state_s, sig, failures),
        tool=_tool_name(manifest),
        actions=_action_lines(manifest),
        failures=failures,
        library=_library_name(manifest),
    )


def _signature_line(state: str | None, sig: dict, failures: list[str]) -> str:
    name = sig.get("common_name") or sig.get("issuer") or "unnamed certificate"
    mismatch = any(item.startswith("claimSignature.mismatch") for item in failures)
    if mismatch or state not in {"Valid", "Trusted"}:
        return f"not verified ({name})"
    if state == "Trusted":
        return f"trusted ({name})"
    return f"local dev cert ({name}); not a public trust chain"


def _tool_name(manifest: dict) -> str | None:
    for action in _iter_actions(manifest):
        agent = action.get("softwareAgent")
        if isinstance(agent, str) and agent.strip():
            return agent.strip()
        if isinstance(agent, dict) and agent.get("name"):
            version = agent.get("version")
            return f"{agent['name']} {version}" if version else str(agent["name"])
    for info in manifest.get("claim_generator_info") or []:
        if isinstance(info, dict) and info.get("name"):
            version = info.get("version")
            return f"{info['name']} {version}" if version else str(info["name"])
    generator = manifest.get("claim_generator")
    return str(generator) if generator else None


def _library_name(manifest: dict) -> str | None:
    for info in manifest.get("claim_generator_info") or []:
        if not isinstance(info, dict):
            continue
        for key, value in info.items():
            if key.startswith("org.contentauth."):
                return f"{key} {value}"
    return None


def _iter_actions(manifest: dict):
    for assertion in manifest.get("assertions") or []:
        if not isinstance(assertion, dict):
            continue
        label = str(assertion.get("label") or "")
        if not label.startswith("c2pa.actions"):
            continue
        data = assertion.get("data") or {}
        actions = data.get("actions") if isinstance(data, dict) else None
        if not isinstance(actions, list):
            continue
        for action in actions:
            if isinstance(action, dict):
                yield action


def _action_lines(manifest: dict) -> list[str]:
    lines: list[str] = []
    for action in _iter_actions(manifest):
        name = str(action.get("action") or "unknown")
        description = action.get("description")
        if description:
            lines.append(f"{name}: {description}")
        else:
            lines.append(name)
    return lines


def format_inspection(inspection: Inspection) -> str:
    """Text for `deepfake provenance inspect`. Only fields that were read back."""
    if not inspection.present:
        return "C2PA: no manifest"
    lines = [
        "C2PA: present",
        f"validation: {inspection.validation_state or 'unknown'}",
        f"signature: {inspection.signature}",
    ]
    if inspection.tool:
        lines.append(f"tool: {inspection.tool}")
    if inspection.library:
        lines.append(f"signer library: {inspection.library}")
    if inspection.actions:
        lines.append("actions:")
        lines.extend(f"- {item}" for item in inspection.actions)
    else:
        lines.append("actions: none found")
    interesting = [
        item
        for item in inspection.failures
        if not item.startswith("signingCredential.untrusted")
    ]
    if interesting and inspection.validation_state not in {"Valid", "Trusted"}:
        lines.append("failures:")
        lines.extend(f"- {item}" for item in interesting)
    return "\n".join(lines)
