# deepfake

Linux **real-time face-swap CLI** wrapping [AlphaFace](https://github.com/andrewyu90/Alphaface_Official) research ([arXiv:2601.16429](https://arxiv.org/abs/2601.16429)).

Upstream AlphaFace is a **still/batch** research demo. This project adds a real application shell: webcam/video capture, face detect → swap → composite, presets, metrics, and OBS-friendly outputs (preview, file, FFmpeg pipe, optional GStreamer, v4l2loopback virtual cam).

## Research & consent framing

This tool is for **research, VFX, avatars, filmmaking, consenting demos, and disclosed synthetic media**.

- **No anonymity claims.** Output is synthetic; disclose when required.
- **Consent gates** (`--consent-ack` or `DEEPFAKE_CONSENT_ACK=1`) before live/video runs.
- **Synthetic-media watermark** on by default (`--no-watermark` to disable — you still must disclose by other means).
- Use **only** identities and footage you own or have consent to process.
- Misuse for non-consensual deepfakes, impersonation, fraud, or harassment is prohibited.

## Install

```bash
pip install -e ".[dev]"
# optional: pip install -e ".[gdown]"          # Drive downloads
# optional: pip install -e ".[insightface]"    # research fallback
deepfake --help
deepfake devices
deepfake models list
```

Nix: `nix develop` via `flake.nix` (CPU-friendly by default; CUDA/torch not forced).

Weights are **not** in this repo. Install explicitly:

```bash
deepfake models install alphaface --yes   # Drive weights — license undocumented; no redistribute
# fallback (NON-COMMERCIAL research models):
deepfake models install inswapper --yes
```

## Usage

```bash
# Devices / models
deepfake devices
deepfake models list

# Live webcam (requires --consent-ack and --source)
deepfake webcam --source person.jpg --consent-ack --preset balanced
deepfake webcam --source person.jpg --consent-ack --preset low-latency --device cuda

# Video file
deepfake video input.mp4 --source person.jpg --consent-ack -o out.mp4 --no-preview

# Virtual webcam (v4l2loopback) — OBS-friendly
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=deepfake
deepfake virtualcam --source person.jpg --consent-ack --v4l2 /dev/video10

# Optional fakeperson synthetic identity
deepfake webcam --identity alice --consent-ack

# Real benchmark: swap only vs swap + AI frame generation (720p/1080p)
deepfake benchmark --consent-ack
deepfake benchmark --json --consent-ack > bench.json
```

### AI frame generation (`--frame-gen`)

Doubles or triples the presented frame rate by **interpolating** between swapped
frames with RIFE (Practical-RIFE 4.25) instead of running AlphaFace on every
displayed frame. Generated frames are real in-betweens at the exact output
timestamp; nothing is duplicated to inflate the counter (late slots are
re-sent as *held* frames and reported separately).

```bash
deepfake models install rife --yes                   # ~74 MB, MIT, SHA-256 pinned
deepfake webcam --source person.jpg --consent-ack --frame-gen          # 2x
deepfake webcam --source person.jpg --consent-ack --frame-gen 3x
deepfake webcam --source person.jpg --consent-ack --output-fps 60      # picks the multiplier
deepfake webcam --source person.jpg --consent-ack --frame-gen auto     # measures, then decides
deepfake virtualcam --source person.jpg --consent-ack --frame-gen 2x --preset latency
deepfake video in.mp4 --source person.jpg --consent-ack --frame-gen 2x -o out60.mp4
```

| Flag | Meaning |
|---|---|
| `--frame-gen [2x\|3x\|4x\|auto]` | enable (bare flag = 2x); off by default |
| `--no-frame-gen` | force off |
| `--output-fps 60\|120` | target presented rate (implies frame generation) |
| `--frame-gen-backend rife` | backend (pluggable, see `deepfake/framegen/registry.py`) |
| `--frame-gen-model 4.25\|4.25.lite\|4.26` | RIFE variant (default from `--preset`) |
| `--preset latency\|balanced\|quality` | latency budget 120/200/300 ms, RIFE lite/4.25/4.26, swap bf16/bf16/fp32 |
| `--swap-precision auto\|fp32\|bf16` | AlphaFace precision (bf16 ≈ 40 % faster, measured) |

Measured on an RTX 4090 (30 fps camera, 2026-09-22):

| Res | Mode | Output fps (new frames) | Capture→sink latency |
|---|---|---|---|
| 720p | off | 30.0 | 28 ms |
| 720p | 2x | 59.5 | 82 ms |
| 720p | 3x | 89.7 | 87 ms |
| 1080p | off | 30.0 | 31 ms |
| 1080p | 2x | 59.4 | 128 ms |

Interpolation needs the *next* swapped frame, so it always adds at least one
source interval of latency — that is why it is opt-in. Design, backend
comparison, pacing and full numbers: [docs/frame-generation.md](docs/frame-generation.md).

### Presets

| Preset | Intent |
|--------|--------|
| `low-latency` (alias `latency`) | 640×480 @ 30, sparse detect, light blend |
| `balanced` | 960×540 @ 24, color match (default) |
| `high-quality` (alias `quality`) | 1280×720 @ 24, multi-face, stronger temporal smooth |

### Controls (common flags)

`--device`, `--input-device`, `--output-device`, `--width/--height/--fps`, `--face-index`, `--multi-face`, `--blend-feather`, `--color-match`, `--temporal-smooth`, `--backend auto|alphaface|inswapper|passthrough`, `--watermark`, `--ffmpeg-out`, `--gstreamer`, `--preview`.

### Outputs

- Preview window (OpenCV)
- Saved video (`-o`)
- FFmpeg raw pipe (`--ffmpeg-out …`)
- GStreamer (errors with install hint if `gst-launch-1.0` missing)
- v4l2loopback virtual webcam (`virtualcam` / `--output-device`)

### Live metrics

Once per second: camera fps, swap fps and latency, output fps (new frames
only), frame-generation cost per frame, generated/held/late counts, queue
depth, capture→sink latency, dropped source frames, GPU utilisation and VRAM.

### v4l2loopback on NixOS

The module has to be part of the kernel package set:

```nix
boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
boot.kernelModules = [ "v4l2loopback" ];
boot.extraModprobeConfig = ''options v4l2loopback devices=1 video_nr=10 card_label="deepfake" exclusive_caps=1'';
```

## Models & licenses

| Model | Code | Weights | Notes |
|-------|------|---------|-------|
| **alphaface** | MIT (upstream) | **Undocumented** (Google Drive) | We **refuse to redistribute** weights; installer downloads with `--yes` ack + checksum record |
| **inswapper** | MIT (InsightFace code) | **Non-commercial** research | Fallback behind `models install`; commercial use needs InsightFace license |
| **rife** | MIT (Practical-RIFE; vs-rife refactor) | MIT | Frame generation; SHA-256 verified, converted to safetensors |

Our wrapper code is **MIT**. We do **not** relicense AlphaFace. See [NOTICE](NOTICE).

## Pipeline (honest)

1. Open webcam/video via OpenCV  
2. Detect faces (OpenCV Haar; YuNet/InsightFace optional)  
3. Call AlphaFace when installed **or** InsightFace inswapper **or** clear error with install instructions  
4. Composite (oval mask, LAB colour match, motion-aware temporal smoothing)  
5. Optional RIFE frame generation on its own CUDA stream + paced output  
6. Disclosure watermark on every presented frame  
7. Output to window / file / ffmpeg / gstreamer / v4l2loopback  

See [STATUS.md](STATUS.md) for what works on a CUDA-less box vs with weights.

## Citation

```bibtex
@article{yu2026alphaface,
  title={AlphaFace: High Fidelity and Real-time Face Swapper Robust to Facial Pose},
  author={Yu, Jongmin and Oh, Hyeontaek and Sun, Zhongtian and Aviles-Rivero, Angelica I and Jeon, Moongu and Yang, Jinhong},
  journal={arXiv preprint arXiv:2601.16429},
  year={2026}
}
```

- Paper: https://arxiv.org/abs/2601.16429  
- Code: https://github.com/andrewyu90/Alphaface_Official  

## License

MIT for **this** repository — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
