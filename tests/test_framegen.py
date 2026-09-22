import numpy as np
import pytest

from deepfake.detect import FaceBox
from deepfake.framegen.base import timesteps_for
from deepfake.framegen.pacing import FrameKind, FramePacer, OutputTimeline
from deepfake.framegen.registry import create_backend
from deepfake.framegen.settings import FrameGenSettings, factor_for, parse_frame_gen
from deepfake.pipeline import motion_scaled_smoothing


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
    assert pts[1].s is None  # the real keyframe fills its own slot


def test_timeline_irregular_source_never_duplicates_indices():
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    times = [0.0, 0.041, 0.07, 0.118, 0.15, 0.2]  # jittery ~25 fps swap output
    seen: list[int] = [0]
    for a, b in zip(times, times[1:], strict=False):
        for p in tl.points_between(a, b):
            assert p.index > seen[-1]
            seen.append(p.index)
            if p.s is not None:
                assert 0.0 < p.s < 1.0
    # every slot up to 0.2 s × 60 fps is covered exactly once
    assert seen == list(range(0, 13))


def test_timeline_skips_slots_after_stall():
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pts = tl.points_between(0.5, 0.5 + 1 / 30)  # nothing produced for 0.5 s
    assert pts and pts[0].time > 0.5


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _frame(v: int) -> np.ndarray:
    return np.full((4, 4, 3), v, np.uint8)


def test_pacer_presents_in_order_and_holds_when_late():
    clock = FakeClock()
    written: list[tuple[int, FrameKind, int]] = []
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pacer = FramePacer(
        tl, lambda f, i, k: written.append((i, k, int(f[0, 0, 0]))), clock=clock, initial_delay=0.05, max_delay=0.2
    )
    pacer.submit(0, _frame(0), FrameKind.KEY, 0.0)
    pacer.submit(1, _frame(1), FrameKind.GENERATED, 0.03)
    pacer.present_due(0)
    pacer.present_due(1)
    pacer.present_due(2)  # nothing queued for slot 2 → previous frame is held
    assert written == [(0, FrameKind.KEY, 0), (1, FrameKind.GENERATED, 1), (2, FrameKind.HELD, 1)]
    assert pacer.stats.generated == 1 and pacer.stats.keyframes == 1 and pacer.stats.held == 1
    # slot 2 arrives after it was presented: dropped and counted, delay raised (bounded)
    clock.t = 0.1
    assert pacer.submit(2, _frame(2), FrameKind.KEY, 0.05) is False
    assert pacer.stats.late_dropped == 1
    assert 0.05 < pacer.delay <= 0.2


def test_pacer_ignores_outage_beyond_max_delay():
    clock = FakeClock()
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pacer = FramePacer(tl, lambda *_: None, clock=clock, initial_delay=0.05, max_delay=0.2)
    pacer.present_due(0)
    clock.t = 5.0
    pacer.submit(0, _frame(0), FrameKind.KEY, 0.0)
    assert pacer.delay == pytest.approx(0.05)


def test_pacer_bounded_queue():
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pacer = FramePacer(tl, lambda *_: None, clock=FakeClock(), max_queue=4)
    for i in range(10):
        pacer.submit(i, _frame(i), FrameKind.GENERATED, 0.0)
    assert pacer.stats.queue_depth == 4
    assert pacer.stats.late_dropped == 6


def test_pacer_delay_shrinks_with_steady_slack():
    clock = FakeClock()
    tl = OutputTimeline(60.0)
    tl.start(0.0)
    pacer = FramePacer(tl, lambda *_: None, clock=clock, initial_delay=0.12)
    for i in range(200):
        clock.t = tl.time_of(i)  # every frame arrives right at its grid time → 120 ms spare
        pacer.submit(i, _frame(0), FrameKind.KEY, clock.t)
        pacer.present_due(i)
    assert pacer.delay < 0.05


def test_motion_scaled_smoothing():
    a = FaceBox(100, 100, 200, 200)
    assert motion_scaled_smoothing(0.4, None, a) == 0.0
    assert motion_scaled_smoothing(0.4, a, a) == pytest.approx(0.4)
    moved = FaceBox(120, 100, 200, 200)  # 10% of face width
    assert motion_scaled_smoothing(0.4, a, moved) == 0.0


def test_unknown_backend():
    with pytest.raises(KeyError):
        create_backend("nope")


@pytest.mark.gpu
def test_rife_interpolates_midpoint_between_shifted_frames():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from deepfake.framegen.rife import weights_path

    if weights_path("4.25") is None:
        pytest.skip("run: deepfake models install rife --yes")
    be = create_backend("rife", variant="4.25")
    be.initialize(256, 128)
    img = np.zeros((128, 256, 3), np.uint8)
    img[40:88, 40:88] = 255
    shifted = np.roll(img, 32, axis=1)
    be.push(img)
    be.push(shifted)
    (mid,) = be.interpolate([0.5])
    cols = np.where(mid[64, :, 0] > 128)[0]
    # the square should sit roughly halfway (x≈56..104), not duplicated at either end
    assert 50 <= cols.min() <= 62 and 98 <= cols.max() <= 110
    be.shutdown()


def test_shed_controller_degrades_and_recovers():
    from deepfake.framegen.pacing import GridPoint
    from deepfake.realtime import ShedController

    c = ShedController(window=1.0, high=0.15, low=0.03, cooldown=2.0)
    c.update(0.0, 0, 0)
    assert c.update(1.0, 30, 60) == 1  # 50% of slots late/held → shed
    assert c.update(2.0, 60, 120) == 2
    pts = [GridPoint(1, 0.0, 1 / 3), GridPoint(2, 0.0, 2 / 3), GridPoint(3, 0.0, 0.5)]
    assert ShedController.select(pts, 1) == [pts[2]]
    assert ShedController.select(pts, 2) == []
    assert c.update(3.0, 60, 180) == 2  # calm starts
    assert c.update(5.1, 60, 300) == 1  # calm for >= cooldown → step down
