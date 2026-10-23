"""Exercise real head inference, RIFE, file recording, previews and session IPC.

Uses only the bundled fictional portrait and a synthetic camera. No physical
camera is activated. Run inside the configured CUDA development environment.
"""
import json
from dataclasses import replace
from pathlib import Path

import cv2

from deepfake.benchmark import SyntheticCamera, bench_face_path
from deepfake.framegen.settings import FrameGenSettings
from deepfake.outputs.sinks import create_sink
from deepfake.pipeline import build_config, load_bgr
from deepfake.realtime import run_offline, run_realtime
from deepfake.session import DeepfakeSession

folder = Path("/tmp/deepfake-integration-artifacts")
folder.mkdir(exist_ok=True)
source = load_bgr(bench_face_path())
config = build_config(device="cuda", engine="experimental", mode="head", model_backend="liveportrait",
                      show_metrics=False, preview_original=True, max_frames=45)
config.preset = replace(config.preset, width=640, height=480, fps=30)
reports = {}
for factor in (None, 2):
    label = "native" if factor is None else "rife-2x"
    fps = 30 if factor is None else 60
    camera = SyntheticCamera(source, (640, 480), 30, 100)
    camera.bg[:] = 48
    path = folder / f"live-{label}.mp4"
    sink = create_sink(output_video=path, width=640, height=480, fps=fps, encoder="libx264", realtime_recording=True)
    session = DeepfakeSession("webcam", source_face=bench_face_path(), widget=False, resolution=(640, 480))
    try:
        session.start()
        fg = FrameGenSettings(enabled=factor is not None, factor=factor)
        session.apply_frame_gen(fg)
        session.mark_running()
        reports[label] = run_realtime(None, None, sink, config, fg, capture=camera, source_bgr=source,
                                      source_fps=30, print_stats=False, on_stats=session.publish_stats)
    finally:
        session.close()
    probe = cv2.VideoCapture(str(path))
    assert probe.isOpened()
    reports[label]["recorded_frames"] = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    reports[label]["recorded_timeline_fps"] = probe.get(cv2.CAP_PROP_FPS)
    probe.release()
    assert reports[label]["recorded_frames"] > 0

# Exact ordering and source-rate preservation through the CLI's shared offline loop.
input_path = folder / "input.mp4"
camera = SyntheticCamera(source, (640, 480), 17, None)
camera.bg[:] = 48
writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 17, (640, 480))
for i in range(12):
    writer.write(camera.frame_at(i))
writer.release()
output = folder / "offline.mp4"
reports["offline"] = run_offline(str(input_path), bench_face_path(), lambda w, h, fps: create_sink(
    output_video=output, width=w, height=h, fps=fps, encoder="libx264"), config, None, print_stats=False)
probe = cv2.VideoCapture(str(output))
assert int(probe.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
assert abs(probe.get(cv2.CAP_PROP_FPS) - 17) < 0.01
probe.release()
(folder / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
print(json.dumps(reports, indent=2))
