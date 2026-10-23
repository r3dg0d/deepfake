"""Reproducible head benchmark using the bundled fictional identity.

No camera, uploads, or external demonstration media. Tests translation, not
expression quality. Store unmodified JSON alongside the human-readable report.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from deepfake.benchmark import SyntheticCamera, bench_face_path
from deepfake.engines import FrameContext, create_engine
from deepfake.pipeline import build_config, load_bgr
from deepfake.watermark import apply_watermark

parser = argparse.ArgumentParser()
parser.add_argument("--backend", default="liveportrait", choices=["liveportrait", "alphaface"])
parser.add_argument("--resolution", default="640x480")
parser.add_argument("--frames", default=60, type=int)
parser.add_argument("--json", required=True, type=Path)
parser.add_argument("--video", type=Path)
args = parser.parse_args()
width, height = map(int, args.resolution.split("x"))
source = load_bgr(bench_face_path())
camera = SyntheticCamera(source, (width, height), 30, None)
camera.bg[:] = 48  # Avoid Haar false positives in random synthetic texture.
config = build_config(device="cuda", engine="experimental" if args.backend == "liveportrait" else "alphaface",
                      mode="head" if args.backend == "liveportrait" else "face",
                      model_backend=args.backend if args.backend == "liveportrait" else None,
                      backend="alphaface", swap_precision="bf16", preview_original=False)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
engine = create_engine(config)
load_s = time.perf_counter() - start
start = time.perf_counter()
engine.prepare_reference(source)
reference_s = time.perf_counter() - start
for i in range(4):
    engine.process_frame(camera.frame_at(i), FrameContext(i, i/30))
torch.cuda.synchronize()
writer = None
if args.video:
    writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width*2, height))
latencies, changes = [], []
start_all = time.perf_counter()
try:
    for i in range(args.frames):
        frame = camera.frame_at(i)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out, _ = engine.process_frame(frame, FrameContext(i, i/30))
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
        changes.append(float(np.abs(frame.astype(float) - out.astype(float)).mean()))
        if writer:
            writer.write(np.concatenate((frame, apply_watermark(out, enabled=True)), axis=1))
    wall = time.perf_counter() - start_all
    result = {"backend": args.backend, "resolution": args.resolution, "frames": args.frames,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "model_load_s": load_s, "reference_prepare_s": reference_s,
              "native_inference_fps": 1000 / np.mean(latencies), "wall_fps_including_export": args.frames / wall,
              "mean_latency_ms": np.mean(latencies), **{f"latency_p{p}_ms": float(np.percentile(latencies, p)) for p in (50, 95, 99)},
              "peak_vram_mb_torch": torch.cuda.max_memory_allocated()/2**20,
              "steady_vram_mb_torch": torch.cuda.memory_allocated()/2**20,
              "mean_pixel_change": np.mean(changes), "quality_validation": "not evaluated; synthetic translation sequence only",
              "capture_to_sink_ms": None, "framegen": False, "streaming_camera_test": False}
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
finally:
    engine.unload()
    if writer:
        writer.release()
