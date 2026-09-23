# deepfake

Linux **real-time face-swap CLI** wrapping [AlphaFace](https://github.com/andrewyu90/Alphaface_Official) research ([arXiv:2601.16429](https://arxiv.org/abs/2601.16429)).

Upstream AlphaFace is a **still/batch** research demo. This project adds a real application shell: webcam/video capture, face detect → swap → composite, presets, metrics, and OBS-friendly outputs (preview, file, FFmpeg pipe, optional GStreamer, v4l2loopback virtual cam).

## Research & consent framing

This tool is for **research, VFX, avatars, filmmaking, consenting demos, and disclosed synthetic media**.

- **No anonymity claims.** Output is synthetic; disclose when required.
- A **one-time notice** on first run (remembered in `~/.config/deepfake/consent.json`; scripts can pre-accept with `DEEPFAKE_CONSENT_ACK=1`).
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
deepfake webcam                         # FrameGen + Quickshell start automatically
deepfake webcam -f person.png           # first run / change face (remembered)
deepfake virtualcam                     # virtual camera for OBS / browsers / calls
deepfake virtualcam -f person.png
deepfake video -i input.mp4 -f person.png -o output.mp4
deepfake video -i input.mp4 -f person.png   # writes input-deepfake.mp4 (never overwrites)
deepfake devices
deepfake doctor                         # AlphaFace / CUDA / FrameGen / Quickshell / IPC
deepfake benchmark                      # AlphaFace + FrameGen pipeline on this GPU
deepfake config show
```

**Defaults:** AI frame generation is **on** (`auto`), and the matrix Quickshell widget
starts with every processing mode. Escape hatches for debugging:

```bash
deepfake webcam --frame-gen off
deepfake webcam --no-widget
deepfake webcam --frame-gen 3x --output-fps 120
```

The last `-f` / `--source` is remembered in `~/.config/deepfake/settings.json`.
Preferences live in `~/.config/deepfake/config.toml` (created on first run):

```toml
frame_generation = "auto"
quickshell = "auto"
gpu = "auto"
preset = "balanced"
encoder = "auto"
preview = true
```

### AI frame generation (automatic)

Doubles or triples the presented frame rate by **interpolating** between swapped
frames with RIFE instead of duplicating frames. Generated frames are real
in-betweens; held/late frames are reported separately in metrics and the widget.

```bash
deepfake models install rife --yes                   # ~74 MB, MIT, SHA-256 pinned
deepfake webcam                                      # auto FrameGen (default)
deepfake webcam --frame-gen 2x
deepfake webcam --frame-gen 3x
deepfake webcam --output-fps 60
deepfake virtualcam --preset latency
deepfake video -i in.mp4 -f person.png -o out60.mp4
```

| Flag | Meaning |
|---|---|
| `--frame-gen [auto\|2x\|3x\|4x\|off]` | default **auto** (on); `off` disables |
| `--no-frame-gen` | force off |
| `--output-fps 60\|120` | target presented rate |
| `--frame-gen-backend rife` | backend (pluggable) |
| `--frame-gen-model 4.25\|4.25.lite\|4.26` | RIFE variant (from `--preset`) |
| `--preset latency\|balanced\|quality` | latency budget + RIFE + swap precision |
| `--widget / --no-widget` | desktop status widget (default auto/on) |

If FrameGen fails to initialize, Deepfake **falls back** to native AlphaFace
output, warns in the CLI and widget, and keeps running.

Measured on an RTX 4090 (30 fps camera, 2026-09-22):

| Res | Mode | Output fps (new frames) | Capture→sink latency |
|---|---|---|---|
| 720p | off | 30.0 | 28 ms |
| 720p | 2x | 59.5 | 82 ms |
| 720p | 3x | 89.7 | 87 ms |
| 1080p | off | 30.0 | 31 ms |
| 1080p | 2x | 59.4 | 128 ms |

Design, backend comparison, pacing and full numbers: [docs/frame-generation.md](docs/frame-generation.md).

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

### Virtual camera (v4l2loopback)

`deepfake virtualcam` auto-detects a v4l2loopback device (preferring one
labelled `deepfake`) and announces the real output rate with
`v4l2loopback-ctl set-fps`, so OBS/browsers timestamp 60 fps frame-generated
output correctly. On NixOS:

```nix
{ config, pkgs, ... }:
let v4l2loopback = config.boot.kernelPackages.v4l2loopback; in {
  boot.extraModulePackages = [ v4l2loopback ];
  boot.kernelModules = [ "v4l2loopback" ];
  boot.extraModprobeConfig = ''options v4l2loopback devices=1 video_nr=10 card_label="deepfake" exclusive_caps=1'';
  # allow the `video` group to set the announced frame rate
  services.udev.extraRules = ''
    ACTION=="add", SUBSYSTEM=="video4linux", ATTR{name}=="deepfake", RUN+="${pkgs.coreutils}/bin/chgrp video /sys%p/format", RUN+="${pkgs.coreutils}/bin/chmod g+w /sys%p/format"
  '';
  environment.systemPackages = [ v4l2loopback.bin ];
}
```

After `nixos-rebuild switch` the module loads at boot; to load it right away:
`sudo env MODULE_DIR=/run/current-system/kernel-modules/lib/modules modprobe v4l2loopback`.
Other distros: `sudo modprobe v4l2loopback devices=1 video_nr=10 card_label=deepfake exclusive_caps=1`.

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
