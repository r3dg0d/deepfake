from deepfake.session import heuristic_factor, resolve_frame_gen
from deepfake.framegen.settings import FrameGenSettings


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
        backend="rife",
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
        backend="rife",
        variant=None,
    )
    assert isinstance(fg, FrameGenSettings)
    assert fg.enabled is False
