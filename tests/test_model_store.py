import hashlib
import http.server
import json
import threading
import zipfile

import pytest

from deepfake import model_store as ms


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body, ctype = self.files.get(self.path, (None, None))
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            body = body[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(ms, "ALLOWED_HOSTS", ms.ALLOWED_HOSTS + (base,))
    _Handler.files = {}
    yield base, _Handler.files
    srv.shutdown()


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path


def _spec(base, name, body):
    return {"path": name, "url": f"{base}/{name}", "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def test_download_verifies_digest(server, tmp_path):
    base, files = server
    body = b"\x80\x02weights" * 1000
    files["/w.pkl"] = (body, "application/octet-stream")
    out = ms.download(_spec(base, "w.pkl", body), tmp_path / "w.pkl")
    assert out.read_bytes() == body
    assert not (tmp_path / "w.pkl.part").exists()


def test_html_error_page_is_rejected(server, tmp_path):
    base, files = server
    page = b"<!DOCTYPE html><html><body>Quota exceeded</body></html>"
    files["/a.pt"] = (page, "text/html; charset=utf-8")
    with pytest.raises(ms.ModelStoreError, match="HTML"):
        ms.download({"path": "a.pt", "url": f"{base}/a.pt", "size": None, "sha256": None}, tmp_path / "a.pt")
    files["/b.pt"] = (page, "application/octet-stream")  # lying content type
    with pytest.raises(ms.ModelStoreError, match="HTML"):
        ms.download({"path": "b.pt", "url": f"{base}/b.pt", "size": None, "sha256": None}, tmp_path / "b.pt")
    assert not (tmp_path / "a.pt").exists() and not (tmp_path / "b.pt").exists()


def test_digest_mismatch_discards_partial(server, tmp_path):
    base, files = server
    files["/w.bin"] = (b"x" * 100, "application/octet-stream")
    spec = {"path": "w.bin", "url": f"{base}/w.bin", "size": 100, "sha256": "0" * 64}
    with pytest.raises(ms.ModelStoreError, match="SHA-256 mismatch"):
        ms.download(spec, tmp_path / "w.bin")
    assert not (tmp_path / "w.bin").exists() and not (tmp_path / "w.bin.part").exists()


def test_download_resumes_from_partial(server, tmp_path):
    base, files = server
    body = bytes(range(256)) * 400
    files["/r.bin"] = (body, "application/octet-stream")
    (tmp_path / "r.bin.part").write_bytes(body[:5000])
    ms.download(_spec(base, "r.bin", body), tmp_path / "r.bin")
    assert (tmp_path / "r.bin").read_bytes() == body


def test_non_allowlisted_host_refused(tmp_path):
    with pytest.raises(ms.ModelStoreError, match="allowlisted"):
        ms.download({"path": "x", "url": "http://example.com/x", "size": 1, "sha256": None}, tmp_path / "x")


def test_validate_format(tmp_path):
    p = tmp_path / "fake.pt"
    p.write_bytes(b"not a zip at all")
    with pytest.raises(ms.ModelStoreError):
        ms.validate_format(p, "torch-zip")
    z = tmp_path / "ok.pt"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("archive/data.pkl", b"\x80\x02}q\x00.")
    assert "torch-zip" in ms.validate_format(z, "torch-zip")


def test_zip_slip_rejected(tmp_path):
    z = tmp_path / "evil.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../escape.onnx", b"x")
    with pytest.raises(ms.ModelStoreError, match="unsafe"):
        ms._safe_extract(z, tmp_path / "out")


def _tiny_core(monkeypatch, base, files):
    body = b"PK-like-not-really" * 10
    files["/tiny.bin"] = (body, "application/octet-stream")
    spec = {**_spec(base, "tiny.bin", body), "trust": "pinned", "format": "torch-pickle"}
    monkeypatch.setitem(ms.CORE, "tinymodel", {"files": [spec]})
    monkeypatch.setattr(ms, "validate_format", lambda p, f: "ok")
    from deepfake import models

    monkeypatch.setitem(models.CATALOG, "tinymodel", {"code_license": "MIT", "weights_license": "MIT"})
    return body


def test_install_then_verify_detects_corruption(server, cache, monkeypatch):
    base, files = server
    _tiny_core(monkeypatch, base, files)
    rec = ms.install("tinymodel", load_check=False)
    assert rec["integrity"] == "verified"
    assert ms.verify("tinymodel")["ok"]
    live = ms.models_dir() / "tinymodel" / "tiny.bin"
    live.write_bytes(b"corrupted")
    v = ms.verify("tinymodel")
    assert not v["ok"] and "size" in v["files"][0]["reason"]


def test_failed_load_check_restores_previous(server, cache, monkeypatch):
    base, files = server
    body = _tiny_core(monkeypatch, base, files)
    ms.install("tinymodel", load_check=False)
    from deepfake import model_checks

    monkeypatch.setattr(model_checks, "run_check", lambda name: {"ok": False, "error": "boom"})
    with pytest.raises(ms.ModelStoreError, match="previous files restored"):
        ms.install("tinymodel", force=True, load_check=True)
    assert (ms.models_dir() / "tinymodel" / "tiny.bin").read_bytes() == body
    assert (ms.models_dir() / ".failed" / "tinymodel").is_dir()


def test_installed_json_alone_is_not_installed(cache):
    from deepfake import models

    models.ensure_dirs()
    (models.models_dir() / "installed.json").write_text(json.dumps({"alphaface": {"name": "alphaface"}}))
    (models.models_dir() / "alphaface").mkdir()
    (models.models_dir() / "alphaface" / "random.pt").write_bytes(b"x")
    row = next(r for r in models.list_models() if r.name == "alphaface")
    assert row.installed is False
