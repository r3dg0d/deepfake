"""Experimental model installer: PersonaLive pins, resume/retry, repair, reinstall and the verification stamp."""

import hashlib
import http.server
import threading

import pytest
from click.testing import CliRunner

import deepfake.experimental_models as module
from deepfake.cli import main
from deepfake.experimental_models import download_file, install, integrity, manifests, validate_manifest, verify
from deepfake.models import list_models, verify_model
from deepfake.paths import models_dir

PAYLOADS = {"/a.bin": b"A" * 300_000, "/dir/b.bin": bytes(range(256)) * 400}


class Handler(http.server.BaseHTTPRequestHandler):
    drop_after: dict[str, int] = {}  # path -> bytes to send before cutting the connection (once)
    requests: list[tuple[str, str | None]] = []

    def do_GET(self):
        body = PAYLOADS.get(self.path)
        type(self).requests.append((self.path, self.headers.get("Range")))
        if body is None:
            self.send_error(404)
            return
        start = 0
        if rng := self.headers.get("Range"):
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            if start >= len(body):
                self.send_error(416)
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
        else:
            self.send_response(200)
        self.send_header("Content-Length", str(len(body) - start))
        self.end_headers()
        cut = type(self).drop_after.pop(self.path, None)
        self.wfile.write(body[start:start + cut] if cut is not None else body[start:])
        if cut is not None:
            self.wfile.flush()
            self.connection.close()

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Handler.drop_after, Handler.requests = {}, []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def spec(base, path):
    body = PAYLOADS[path]
    return {"path": path.lstrip("/"), "url": base + path, "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}


@pytest.fixture
def bundle(server, monkeypatch):
    """A two-file model served locally (the real manifests only allow huggingface.co)."""
    manifest = {**manifests()["liveportrait"], "name": "liveportrait", "files": [spec(server, p) for p in PAYLOADS]}
    monkeypatch.setattr(module, "manifests", lambda: {"liveportrait": manifest})
    monkeypatch.setattr(module, "validate_manifest", lambda m: None)
    return manifest


def test_personalive_is_installable_with_verified_base_checkpoints():
    manifest = manifests()["personalive"]
    validate_manifest(manifest)
    assert not manifest.get("blocked")
    paths = {f["path"]: f for f in manifest["files"]}
    assert len(paths) == 13
    # everything upstream's tools/download_weights.py fetches, in the layout its loader reads
    for needed in ("pretrained_weights/personalive/denoising_unet.pth",
                   "pretrained_weights/sd-image-variations-diffusers/model_index.json",
                   "pretrained_weights/sd-image-variations-diffusers/unet/config.json",
                   "pretrained_weights/sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin",
                   "pretrained_weights/sd-image-variations-diffusers/image_encoder/config.json",
                   "pretrained_weights/sd-image-variations-diffusers/image_encoder/pytorch_model.bin",
                   "pretrained_weights/sd-vae-ft-mse/config.json",
                   "pretrained_weights/sd-vae-ft-mse/diffusion_pytorch_model.bin"):
        assert needed in paths
    base = paths["pretrained_weights/sd-image-variations-diffusers/unet/diffusion_pytorch_model.bin"]
    # the original repository of the (renamed) Lambda organisation at an immutable revision,
    # not the empty repository that now answers under the old "lambdalabs" name
    assert base["url"].startswith("https://huggingface.co/lambda/sd-image-variations-diffusers/resolve/"
                                  "42bc0ee1726b141d49f519a6ea02ccfbf073db2e/")
    assert all("/lambdalabs/" not in f["url"] and "/resolve/main/" not in f["url"] for f in manifest["files"])
    assert "OpenRAIL" in manifest["weights_license"]
    row = next(m for m in list_models() if m.name == "personalive")
    assert row.installed is False and row.meta["integrity"] == "not installed" and "blocked" not in row.meta
    assert "files" not in row.meta and row.meta["file_count"] == 13  # the catalog view carries no per-file rows


def test_download_resumes_after_dropped_connection(server, tmp_path):
    s = spec(server, "/a.bin")
    Handler.drop_after["/a.bin"] = 100_000
    seen = []
    assert download_file(s, tmp_path / "a.bin", lambda path, done, size: seen.append(done), sleep=lambda _: None) is True
    assert (tmp_path / "a.bin").read_bytes() == PAYLOADS["/a.bin"] and not (tmp_path / "a.bin.part").exists()
    assert Handler.requests == [("/a.bin", None), ("/a.bin", "bytes=100000-")]  # second attempt resumed
    assert seen[-1] == s["size"]
    assert (tmp_path / "a.bin").stat().st_mode & 0o777 == 0o600
    Handler.requests.clear()
    assert download_file(s, tmp_path / "a.bin") is False and not Handler.requests  # already matches: no network


def test_download_gives_up_with_resumable_part_and_never_keeps_a_bad_file(server, tmp_path):
    s = spec(server, "/a.bin")
    wrong = {**s, "sha256": "0" * 64}
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_file(wrong, tmp_path / "a.bin")
    assert not (tmp_path / "a.bin").exists() and not (tmp_path / "a.bin.part").exists()
    with pytest.raises(RuntimeError, match="HTTP 404"):
        download_file({**s, "url": server + "/missing"}, tmp_path / "x.bin", sleep=lambda _: None)
    assert len([r for r in Handler.requests if r[0] == "/missing"]) == 1  # a 404 is not retried
    dead = {**s, "url": "http://127.0.0.1:9/a.bin"}
    with pytest.raises(RuntimeError, match="run the install again to resume"):
        download_file(dead, tmp_path / "d.bin", retries=2, sleep=lambda _: None)


def test_forced_download_keeps_the_installed_file_until_the_new_one_verified(server, tmp_path):
    s = spec(server, "/a.bin")
    target = tmp_path / "a.bin"
    target.write_bytes(PAYLOADS["/a.bin"])
    with pytest.raises(RuntimeError, match="installed file is untouched"):
        download_file({**s, "sha256": "1" * 64}, target, force=True)
    assert target.read_bytes() == PAYLOADS["/a.bin"]
    assert download_file(s, target, force=True) is True


def test_install_repair_and_reinstall(bundle):
    with pytest.raises(RuntimeError, match="install --yes"):
        install("liveportrait")
    progress, stages = [], []
    message = install("liveportrait", yes=True, progress=lambda *a: progress.append(a), stage=stages.append)
    assert "2 downloaded" in message
    total = sum(f["size"] for f in bundle["files"])
    assert progress[-1] == ("dir/b.bin", total, total)  # overall bytes, not per file
    assert [d for _, d, _ in progress] == sorted(d for _, d, _ in progress)
    assert stages == ["Downloading a.bin", "Downloading dir/b.bin"]
    assert integrity("liveportrait") == "verified"
    row = next(m for m in list_models() if m.name == "liveportrait")
    assert row.installed and row.meta["integrity"] == "verified"

    # repair: a corrupt file is replaced, the intact one is only re-hashed
    root = models_dir() / "liveportrait"
    (root / "a.bin").write_bytes(b"x" * len(PAYLOADS["/a.bin"]))
    assert integrity("liveportrait") == "unverified"
    assert verify("liveportrait")["corrupt"] == ["a.bin"]
    Handler.requests.clear()
    assert "1 downloaded" in install("liveportrait", yes=True)
    assert [r[0] for r in Handler.requests] == ["/a.bin"]
    assert verify("liveportrait")["ok"]

    # reinstall: everything is fetched again
    Handler.requests.clear()
    assert "2 downloaded" in install("liveportrait", yes=True, force=True)
    assert sorted(r[0] for r in Handler.requests) == ["/a.bin", "/dir/b.bin"]
    assert not list(root.rglob("*.part"))


def test_install_reports_missing_disk_space(bundle, monkeypatch):
    import shutil
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage(10**9, 10**9, 1000))
    with pytest.raises(RuntimeError, match="GiB free"):
        install("liveportrait", yes=True)
    monkeypatch.undo()


def test_quick_verify_trusts_the_stamp_and_notices_changes(bundle, monkeypatch):
    install("liveportrait", yes=True)
    hashed = []
    real = module.hash_file
    monkeypatch.setattr(module, "hash_file", lambda path, tick=None: (hashed.append(path.name), real(path, tick))[1])
    quick = verify_model("liveportrait", quick=True)
    assert quick["ok"] and quick["hashed"] == 0 and quick["trusted_from_stamp"] == 2 and not hashed
    # a rewritten file (same size) has a new mtime: only that one is hashed again, and it fails
    target = models_dir() / "liveportrait" / "a.bin"
    target.write_bytes(b"y" * len(PAYLOADS["/a.bin"]))
    quick = verify_model("liveportrait", quick=True)
    assert not quick["ok"] and quick["corrupt"] == ["a.bin"] and hashed == ["a.bin"]
    assert integrity("liveportrait") == "unverified"
    # a full verification hashes everything and reports progress while hashing
    hashed.clear()
    target.write_bytes(PAYLOADS["/a.bin"])
    ticks = []
    full = verify_model("liveportrait", progress=lambda path, done, total: ticks.append((path, done, total)))
    assert full["ok"] and full["hashed"] == 2 and sorted(hashed) == ["a.bin", "b.bin"]
    assert ticks[-1][1] == ticks[-1][2] == sum(f["size"] for f in bundle["files"])
    assert integrity("liveportrait") == "verified"


def test_cli_repair_and_reinstall_route_experimental_models(bundle):
    runner = CliRunner()
    refused = runner.invoke(main, ["models", "reinstall", "liveportrait"])
    assert refused.exit_code != 0 and "GiB" in refused.output and "--yes" in refused.output
    done = runner.invoke(main, ["models", "reinstall", "liveportrait", "--yes"])
    assert done.exit_code == 0 and "2 downloaded" in done.output
    repaired = runner.invoke(main, ["models", "repair", "liveportrait", "--yes"])
    assert repaired.exit_code == 0 and "0 downloaded" in repaired.output
    assert runner.invoke(main, ["models", "verify", "liveportrait"]).exit_code == 0
