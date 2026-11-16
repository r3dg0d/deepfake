"""Isolated worker backends: configuration from the control center, the host memory guard and render stages."""

import subprocess
import sys
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from deepfake.cli import main
from deepfake.desktop import DEFAULT_PREFERENCES, head_matte_args, model_job, preferences, preferences_path, request
from deepfake.engines import EngineUnavailable
from deepfake.engines.experimental import adapters
from deepfake.ipc import read_json, state_dir, write_json_atomic

GIB = 2**30


def call(action, **extra):
    return request({"protocol_version": 1, "action": action, **extra})


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """A directory that reports the pinned EditaLive revision."""
    path = tmp_path / "EditaLive"
    path.mkdir()
    monkeypatch.setattr(adapters, "_revision", lambda p: adapters.SOURCES["editalive"][1])
    return path


class Session:
    """The slice of DeepfakeSession the render loop uses."""

    def __init__(self, stop_after=None):
        self.updates, self.stop_after, self.polls = [], stop_after, 0

    @property
    def bus(self):
        return self

    def update(self, **fields):
        self.updates.append(fields)

    def stop_requested(self):
        self.polls += 1
        return self.stop_after is not None and self.polls > self.stop_after

    def stages(self):
        return [u["loading_stage"] for u in self.updates if u.get("loading_stage")]


def ready_backend(monkeypatch, checkout, script: str, tmp_path):
    """EditaLive configured and 'installed', with the worker replaced by a small script."""
    adapters.configure_backend("editalive", checkout, Path(sys.executable))
    worker = tmp_path / "worker.py"
    worker.write_text(script)
    monkeypatch.setattr("deepfake.models.is_model_ready", lambda name: True)
    monkeypatch.setattr("deepfake.models.verify_model", lambda name, progress=None, quick=False: (
        progress and progress("weights.safetensors", 5, 10), {"ok": True, "missing": [], "corrupt": [], "quick": quick})[1])
    monkeypatch.setattr(adapters, "worker_command", lambda name, checkout, python, reference, video, output_dir, **kw:
                        [sys.executable, str(worker), str(output_dir)])
    monkeypatch.setattr(adapters, "_CGROUP", False)  # no systemd user manager in the test environment


def test_configure_keeps_disabled_flag_and_reports_what_is_missing(checkout, tmp_path):
    with pytest.raises(ValueError, match="executable"):
        adapters.configure_backend("editalive", checkout, tmp_path / "missing-python")
    with pytest.raises(ValueError, match="not found"):
        adapters.configure_backend("editalive", tmp_path / "nowhere", Path(sys.executable))
    adapters.set_backend_enabled("editalive", False, reason="Disabled after host memory-pressure incident")
    adapters.configure_backend("editalive", checkout, Path(sys.executable))
    status = adapters.backend_status("editalive")
    # re-pointing the paths must not silently re-enable a backend that was switched off
    assert status["configured"] and status["checkout_ok"] and status["python_ok"]
    assert status["disabled_reason"].startswith("Disabled after") and not status["available"]
    assert status["memory"]["limit_gb"] >= 0 and status["live"] is False
    adapters.set_backend_enabled("editalive", True, memory_limit_gb=24)
    status = adapters.backend_status("editalive")
    assert not status["disabled_reason"] and status["status"] == "weights not installed"
    assert status["memory"]["configured"] and status["memory"]["limit_gb"] <= 24
    with pytest.raises(ValueError, match="between 4"):
        adapters.set_backend_enabled("editalive", True, memory_limit_gb=1)


def test_wrong_revision_is_refused_with_the_fix(tmp_path):
    repo = tmp_path / "EditaLive"
    repo.mkdir()
    with pytest.raises(ValueError, match="not a git checkout"):
        adapters.configure_backend("editalive", repo, Path(sys.executable))
    git = ["git", "-C", str(repo), "-c", "user.email=t@example.invalid", "-c", "user.name=t", "-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    with pytest.raises(ValueError, match="git -C .* checkout 9fe7fdb4"):
        adapters.configure_backend("editalive", repo, Path(sys.executable))


def test_memory_plan_reserves_ram_for_the_desktop(monkeypatch):
    monkeypatch.setattr(adapters, "host_memory", lambda: {"total": 64 * GIB, "available": 50 * GIB, "swap_free": 0})
    monkeypatch.setattr(adapters, "_CGROUP", True)
    plan = adapters.memory_plan("editalive")
    assert plan["limit_gb"] == 54.4 and plan["method"] == "cgroup + watchdog"  # 64 - 15 %
    assert plan["floor_bytes"] == int(64 * GIB * 0.05)
    monkeypatch.setattr(adapters, "host_memory", lambda: {"total": 16 * GIB, "available": 8 * GIB, "swap_free": 0})
    assert adapters.memory_plan("editalive")["limit_gb"] == 10.0  # at least 6 GiB stay free
    adapters.set_backend_enabled("editalive", True, memory_limit_gb=512)
    assert adapters.memory_plan("editalive")["limit_gb"] == 16.0  # never above what the host has
    command = adapters.contain(["python", "worker.py"], 10 * GIB)
    assert command[-3:] == ["--", "python", "worker.py"]
    assert f"MemoryMax={10 * GIB}" in command and "MemorySwapMax=0" in command and "--scope" in command
    monkeypatch.setattr(adapters, "_CGROUP", False)
    assert adapters.contain(["python", "worker.py"], 10 * GIB) == ["python", "worker.py"]


def test_memory_guard_trips_on_worker_size_and_on_host_exhaustion():
    now = [0.0]
    rss, available = [1 * GIB], [20 * GIB]
    guard = adapters.MemoryGuard(123, 8 * GIB, 2 * GIB, rss=lambda pgid: rss[0],
                                 meminfo=lambda: {"available": available[0]}, clock=lambda: now[0])
    assert guard.check() is None and guard.rss == 1 * GIB
    rss[0] = 9 * GIB
    assert guard.check() is None  # polled at most once per interval
    now[0] += 1.0
    assert "limit 8.0 GiB" in guard.check() and guard.peak == 9 * GIB
    rss[0], available[0] = 4 * GIB, 1 * GIB
    now[0] += 1.0
    assert "only 1.0 GiB of host RAM was left" in guard.check()


def test_group_rss_counts_this_process_group():
    import os

    assert adapters.group_rss(os.getpgid(0)) > 5 * 2**20
    assert adapters.group_rss(2**22 + 12345) == 0
    assert adapters.host_memory()["total"] > 0


def test_render_runs_worker_with_stages_and_quick_verification(monkeypatch, checkout, tmp_path):
    ready_backend(monkeypatch, checkout, tmp_path=tmp_path, script=(
        "import sys, time, pathlib\n"
        "print('loading text encoder', flush=True); time.sleep(0.5)\n"
        "print('denoising 3/4', flush=True); time.sleep(0.5)\n"
        "pathlib.Path(sys.argv[1], 'generated.mp4').write_bytes(b'video')\n"
        "pathlib.Path(sys.argv[1], 'generated.metrics.json').write_text('{\"native_inference_fps\": 2.5}')\n"))
    session = Session()
    video, summary = adapters.render("editalive", tmp_path / "ref.png", tmp_path / "in.mp4", tmp_path / "out", session)
    assert video.name == "generated.mp4" and summary["native_inference_fps"] == 2.5 and summary["backend"] == "editalive"
    assert "host_rss_peak_gb" in summary
    stages = session.stages()
    assert stages[0] == "Verifying checkpoints" and "Loading models" in stages
    assert any(s.startswith("Worker: ") for s in stages)  # the worker's own last line, not an invented percentage
    running = [u for u in session.updates if "worker_rss_gb" in u]
    assert running and all("host_available_gb" in u for u in running)
    announced = next(u for u in session.updates if u.get("memory_guard"))
    assert announced["memory_guard"] == "watchdog" and announced["memory_limit_gb"] > 0
    log = (state_dir() / "experimental-worker.log").read_text()
    assert "memory ceiling" in log and "denoising 3/4" in log  # readable after the temporary directory is gone


def test_render_stops_worker_at_the_memory_ceiling(monkeypatch, checkout, tmp_path):
    ready_backend(monkeypatch, checkout, tmp_path=tmp_path, script="import time\ntime.sleep(60)\n")

    class Tripping(adapters.MemoryGuard):
        def __init__(self, pgid, limit, floor):
            super().__init__(pgid, 1 * GIB, floor, interval=0, rss=lambda pgid: 2 * GIB, meminfo=lambda: {"available": 64 * GIB})

    monkeypatch.setattr(adapters, "MemoryGuard", Tripping)
    started = time.monotonic()
    with pytest.raises(EngineUnavailable, match="memory guard: the worker reached 2.0 GiB"):
        adapters.render("editalive", tmp_path / "r.png", tmp_path / "in.mp4", tmp_path / "out", Session())
    assert time.monotonic() - started < 10  # terminated, not left running for its 60 s
    assert "memory guard" in (state_dir() / "experimental-worker.log").read_text()


def test_render_reports_worker_failure_and_cancel(monkeypatch, checkout, tmp_path):
    ready_backend(monkeypatch, checkout, tmp_path=tmp_path, script="import sys\nprint('CUDA out of memory')\nsys.exit(3)\n")
    with pytest.raises(EngineUnavailable, match="(?s)worker exited 3.*CUDA out of memory"):
        adapters.render("editalive", tmp_path / "r.png", tmp_path / "in.mp4", tmp_path / "out", Session())
    ready_backend(monkeypatch, checkout, tmp_path=tmp_path, script="import time\ntime.sleep(60)\n")
    with pytest.raises(InterruptedError, match="cancelled"):
        adapters.render("editalive", tmp_path / "r.png", tmp_path / "in.mp4", tmp_path / "out", Session(stop_after=3))


def test_disabled_backend_names_the_way_back(monkeypatch, checkout, tmp_path):
    ready_backend(monkeypatch, checkout, tmp_path=tmp_path, script="")
    adapters.set_backend_enabled("editalive", False, reason="Disabled after host memory-pressure incident.")
    with pytest.raises(EngineUnavailable, match="memory-pressure incident.*experimental enable --backend editalive"):
        adapters.render("editalive", None, None, tmp_path / "out", Session())
    result = CliRunner().invoke(main, ["experimental", "enable", "--backend", "editalive", "--memory-limit-gb", "20"])
    assert result.exit_code == 0 and "memory guard" in result.output and "ceiling" in result.output
    assert adapters.backend_status("editalive")["available"]
    assert CliRunner().invoke(main, ["experimental", "disable", "--backend", "editalive", "--reason", "testing"]).exit_code == 0
    assert adapters.backend_status("editalive")["disabled_reason"] == "testing"


# ------------------------------------------------------------------ desktop bridge


def test_desktop_configures_and_enables_backends(checkout):
    with pytest.raises(ValueError, match="source checkout"):
        call("configure_backend", backend="editalive", checkout="", python=sys.executable)
    with pytest.raises(ValueError, match="isolated worker"):
        call("configure_backend", backend="editalive", checkout=str(checkout), python="")
    with pytest.raises(ValueError, match="Unknown backend"):
        call("configure_backend", backend="rm -rf", checkout=str(checkout), python=sys.executable)
    response = call("configure_backend", backend="editalive", checkout=str(checkout), python=sys.executable)
    assert response["status"]["configured"] and response["status"]["checkout"] == str(checkout)
    adapters.set_backend_enabled("editalive", False, reason="incident")
    with pytest.raises(ValueError, match="confirm"):
        call("enable_backend", backend="editalive")
    with pytest.raises(ValueError, match="Invalid memory limit"):
        call("enable_backend", backend="editalive", confirmed=True, memory_limit_gb="lots")
    enabled = call("enable_backend", backend="editalive", confirmed=True, memory_limit_gb=16)
    assert "memory guard" in enabled["message"] and not enabled["status"]["disabled_reason"]
    engines = {e["name"]: e for e in call("snapshot")["engines"]}
    assert engines["editalive"]["checkout_ok"] and "memory" in engines["editalive"]
    assert "suggested" in engines["personalive"] and engines["liveportrait"]["live"] is True


def test_snapshot_marks_dead_model_job_interrupted():
    path = state_dir() / "model-job.json"
    write_json_atomic(path, {"state": "running", "action": "install_model", "model": "personalive", "pid": 2**22 + 4242})
    job = call("snapshot")["model_job"]
    assert job["state"] == "interrupted" and "resume" in job["error"]
    assert read_json(path)["state"] == "interrupted"
    # a live pid that is not a model job (pid reuse) does not count as running either
    import os

    write_json_atomic(path, {"state": "running", "action": "install_model", "model": "personalive", "pid": os.getpid()})
    assert model_job()["state"] == "interrupted"
    with pytest.raises(ValueError, match="No model operation"):
        call("cancel_model")


def test_model_actions_need_confirmation_and_known_models():
    for action in ("install_model", "repair_model", "reinstall_model", "remove_model"):
        with pytest.raises(ValueError, match="confirm"):
            call(action, model="personalive")
    with pytest.raises(ValueError, match="Unknown model"):
        call("repair_model", model="../../etc", confirmed=True)


def test_preferences_survive_keys_from_other_versions():
    write_json_atomic(preferences_path(), {"engine": "experimental", "removed_in_0_6": 1, "preset": "ultra", "fps": 30})
    prefs = preferences()
    assert set(prefs) == set(DEFAULT_PREFERENCES)
    assert prefs["engine"] == "experimental" and prefs["preset"] == "balanced" and prefs["fps"] == 30
    saved = call("preferences", values={"background": "#12ab34", "rotoscope": "on"})["preferences"]
    assert saved["background"] == "#12ab34" and "removed_in_0_6" not in saved
    for bad in ({"background": "plaid"}, {"background": "#12"}, {"rotoscope": "maybe"}, {"matte_scope": "torso"}):
        with pytest.raises(ValueError):
            call("preferences", values=bad)


def test_head_matte_arguments(tmp_path):
    head = {**DEFAULT_PREFERENCES, "engine": "experimental", "mode": "head", "backend": "liveportrait"}
    assert head_matte_args(DEFAULT_PREFERENCES) == []  # AlphaFace: no matte flags at all
    assert head_matte_args(head) == []  # auto + keep: defaults
    assert head_matte_args({**head, "rotoscope": "off"}) == ["--no-rotoscope"]
    assert head_matte_args({**head, "rotoscope": "on", "background": "green", "matte_scope": "head"}) == [
        "--rotoscope", "--background", "green", "--matte-scope", "head"]
    with pytest.raises(ValueError, match="needs rotoscoping"):
        head_matte_args({**head, "rotoscope": "off", "background": "blur"})
    with pytest.raises(ValueError, match="background image"):
        head_matte_args({**head, "background": "image", "background_image": str(tmp_path / "none.png")})
    image = tmp_path / "room.png"
    image.write_bytes(b"png")
    assert head_matte_args({**head, "background": "image", "background_image": str(image)})[:2] == ["--background", str(image)]
    assert head_matte_args({**head, "backend": "personalive", "background": "green"}) == []


def test_start_and_render_explain_what_is_missing(tmp_path, monkeypatch):
    from deepfake.consent import record_acknowledgement

    record_acknowledgement()
    with pytest.raises(ValueError, match="Choose a character"):
        call("start")
    reference = tmp_path / "ref.png"
    reference.write_bytes(b"png")
    call("preferences", values={"reference": str(reference), "engine": "experimental", "mode": "fullbody", "backend": "editalive"})
    with pytest.raises(ValueError, match="renders video files only"):
        call("start")
    with pytest.raises(ValueError, match="input video"):
        call("render_video")
    video = tmp_path / "in.mp4"
    video.write_bytes(b"mp4")
    call("preferences", values={"input_video": str(video), "recording_path": str(video)})
    with pytest.raises(ValueError, match="never overwritten"):
        call("render_video")
    call("preferences", values={"recording_path": str(tmp_path / "out.mp4")})
    with pytest.raises(ValueError, match="editalive cannot render yet: weights not installed"):
        call("render_video")


def test_log_action_reads_each_log():
    assert call("log")["message"] == "No backend log yet"
    (state_dir() / "experimental-worker.log").write_text("x" * 9000 + "tail")
    response = call("log", which="worker")
    assert response["log"].endswith("tail") and len(response["log"]) == 5000
    with pytest.raises(ValueError, match="Unknown log"):
        call("log", which="../../etc/passwd")
