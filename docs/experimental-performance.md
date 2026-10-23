# Performance and validation

Measured results, raw JSON and scope: [RTX 4090 report](benchmarks/experimental-rtx4090.md).

```bash
deepfake benchmark --resolution 720p --seconds 10 --json
python scripts/benchmark-character.py --frames 300 --resolution 640x480 \
  --json head-480p.json --video head-comparison.mp4
python scripts/benchmark-character.py --frames 300 --resolution 1280x720 --json head-720p.json
python scripts/benchmark-character.py --frames 300 --resolution 1920x1080 --json head-1080p.json
python scripts/validate-character-pipeline.py
```

Use a configured CUDA environment and the bundled fictional identity. The first
script records model loading, reference preparation, native frame time and
P50/P95/P99, peak/steady Torch allocation and export wall rate. It synchronizes
CUDA around timing. Its moving still does not measure expression or pose
fidelity, identity similarity, occlusion quality, temporal drift or long-run
stability. The second exercises shared capture/FrameGen/preview/recording/IPC
and offline source-rate preservation with synthetic input.

Native inference, new presented frames, held repeats, camera input rate and
encoded timeline FPS differ. Native inference never includes RIFE frames.
Live latency includes capture→sink only when the corresponding pipeline records
it; standalone model timing does not establish end-to-end camera latency.
Torch allocation excludes display/driver allocations and other processes.

The existing FrameGen shed controller remains active under load. No new general
adaptive resolution, precision selection, VRAM budget, quantization/compile
auto-tuning or diffusion quality controller is claimed. Evaluate optimizations
one at a time against repeatable clips and retain native/no-FrameGen comparisons.

Release validation still needs consented talking/head rotation, walking/sitting,
hand occlusion, fast movement, lighting changes, long sessions, proper quality
metrics, actual webcam/OBS/browser checks and full-body/hybrid generation.
