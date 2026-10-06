import numpy as np
import pytest

from deepfake.framegen.base import timesteps_for
from deepfake.framegen.pacing import OutputTimeline
from deepfake.framegen.registry import create_backend, select_backend_name
from deepfake.framegen.settings import FrameGenSettings, factor_for, parse_frame_gen


def test_timesteps_for():
    assert timesteps_for(1) == []
    assert timesteps_for(2) == [0.5]
    assert timesteps_for(3) == pytest.approx([1 / 3, 2 / 3])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, (False, None)),
        ("off", (False, None)),
        ("on", (True, 2)),
        ("2x", (True, 2)),
        ("3", (True, 3)),
        ("auto", (True, None)),
    ],
)
def test_parse_frame_gen(value, expected):
    assert parse_frame_gen(value) == expected


@pytest.mark.parametrize("bad", ["1x", "5x", "fast"])
def test_parse_frame_gen_rejects(bad):
    with pytest.raises(ValueError):
        parse_frame_gen(bad)


def test_factor_for_and_output_fps():
    assert factor_for(60, 30) == 2
    assert factor_for(60, 29.9992) == 2
    assert factor_for(60, 30000 / 1001) == 2
    assert factor_for(60, 25) == 3
    assert factor_for(60, 24) == 3
    assert factor_for(120, 30) == 4
    assert FrameGenSettings(enabled=True, factor=3).resolve_output_fps(20) == 60
    assert FrameGenSettings(enabled=True, factor=2, output_fps=50).resolve_output_fps(30) == 50


def test_timeline_exact_doubling():
    tl = OutputTimeline(60.0)
    assert tl.start(0.0).index == 0
    pts = tl.points_between(0.0, 1 / 30)
    assert [p.index for p in pts] == [1, 2]
    assert pts[0].s == pytest.approx(0.5)
    assert pts[1].s is None


def test_select_backend_prefers_nvof_or_passthrough():
    name = select_backend_name("auto")
    assert name in ("nvof", "passthrough", "maxine")


def test_rife_alias_maps_to_nvof():
    assert select_backend_name("rife") in ("nvof", "passthrough", "maxine")


def test_passthrough_backend():
    b = create_backend("passthrough")
    b.initialize(64, 64)
    a = np.zeros((64, 64, 3), np.uint8)
    b.push(a)
    b.push(a)
    out = b.interpolate([0.5])
    assert len(out) == 1 and out[0].shape == (64, 64, 3)
    b.shutdown()


def test_timeline_irregular_source_never_duplicates_indices():
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    times = [0.0, 0.041, 0.07, 0.118, 0.15, 0.2]
    seen: list[int] = [0]
    for a, b in zip(times, times[1:], strict=False):
        for p in tl.points_between(a, b):
            assert p.index > seen[-1]
            seen.append(p.index)
            if p.s is not None:
                assert 0.0 < p.s < 1.0
    assert seen == list(range(0, 13))


def test_timeline_skips_slots_after_stall():
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pts = tl.points_between(0.5, 0.5 + 1 / 30)
    assert pts and pts[0].time > 0.5


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def time(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
