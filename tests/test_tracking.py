import numpy as np

from deepfake.detect import FaceBox
from deepfake.tracking import FaceTracker


def test_reordered_detections_keep_identity_state():
    tracker = FaceTracker()
    frame = np.zeros((200, 300, 3), np.uint8)
    a, b = FaceBox(20, 30, 50, 60), FaceBox(180, 40, 70, 70)
    first = tracker.update(frame, [a, b])
    assert [t.track_id for t in first] == [1, 2]
    second = tracker.update(frame, [b, a])
    assert second[0].box.x == a.x and second[1].box.x == b.x
    assert [t.track_id for t in second] == [1, 2]


def test_no_stale_track_after_detector_loss():
    tracker = FaceTracker(max_missed=1)
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    tracker.update(frame, [FaceBox(15, 15, 60, 60)])
    tracker.update(frame, [])
    assert tracker.update(frame, []) == []
    assert tracker.update(frame, [FaceBox(15, 15, 60, 60)])[0].track_id == 2
