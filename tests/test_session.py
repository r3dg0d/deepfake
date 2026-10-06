from deepfake.framegen.settings import FrameGenSettings
from deepfake.session import heuristic_factor, resolve_frame_gen


def test_heuristic_factor():
    assert heuristic_factor(30) == 2
    assert heuristic_factor(60) == 2
    assert heuristic_factor(24) == 3


def test_resolve_off():
    fg = resolve_frame_gen(
        cli_value="off",
        no_frame_gen=False,
        output_fps=None,
        source_fps=30,
        preset="balanced",
        backend="nvof",
        variant=None,
    )
    assert fg.enabled is False


def test_no_frame_gen_flag():
    fg = resolve_frame_gen(
        cli_value="auto",
        no_frame_gen=True,
        output_fps=None,
        source_fps=30,
        preset="balanced",
        backend="nvof",
        variant=None,
    )
    assert isinstance(fg, FrameGenSettings)
    assert fg.enabled is False


def test_auto_recomputes_stale_heuristic_for_source_timing(monkeypatch):
    import time

    from deepfake import session

    monkeypatch.setattr(
        session, "load_framegen_cache", lambda: {"mode": "3x", "reason": "heuristic", "ts": time.time()}
    )
    saved = []
    monkeypatch.setattr(session, "save_framegen_cache", saved.append)
    settings = session._resolve_auto(FrameGenSettings(), source_fps=29.9992, force_bench=False, progress=None)
    assert settings.factor == 2
    assert saved[0]["mode"] == "2x"
