# NixOS runtime

`nix build` builds the Python 3.12 CPU package, includes Pillow, ONNX Runtime, ONNX and packaged Quickshell assets, and runs pytest. The Linux CPU ONNX Runtime 1.22.0 wheels are fetched from official PyPI with pinned SHA-256 hashes and patched to Nix shared-library paths. This matches the validated runtime family and avoids a large C++ source build. The x86_64 package is validated locally; aarch64 uses its separately pinned official wheel but was not run on this workstation.

`nix develop` includes FFmpeg, OpenSSL, V4L2 tools, Ruff and pytest. Quickshell itself must be present on PATH (`qs`). GPU workloads need a matching CUDA PyTorch/ONNX environment; the CPU flake does not pretend to ship a working CUDA model stack.

`nix develop .#cuda` adds the host NVIDIA driver and Nix shared-library paths for isolated binary wheels; it does not download CUDA models or globally install Python packages.

For CUDA 12/cuDNN 9, create an isolated venv and install `.[vision-cuda12,provenance,watermark]`, plus a compatible CUDA build of PyTorch. Load PyTorch before the CUDA ONNX sessions; the adapter does that. ONNX Runtime 1.22.0 was tested with PyTorch 2.6.0+cu124 on RTX 4090. Newer ONNX packages can require a different CUDA major. Installing CPU and GPU ONNX distributions into the same environment is unsupported: select one extra.

NixOS needs host driver and C++ runtime libraries visible to these binary wheels. A launch wrapper may include:

```bash
export LD_LIBRARY_PATH="/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
exec /absolute/path/to/isolated/venv/bin/deepfake "$@"
```

Use your Nix devShell's compiler-runtime/zlib library paths too if importing a wheel reports a missing shared library. Do not use global pip or overwrite the system Python. NVIDIA optical flow comes from `libnvidia-opticalflow.so.1` in the host NVIDIA driver; no SDK compiler or RIFE weight download is needed. The SDK headers retain their BSD-3 license.

For OBS/browsers:

```nix
{ config, pkgs, ... }: {
  boot.extraModulePackages = [ config.boot.kernelPackages.v4l2loopback ];
  boot.kernelModules = [ "v4l2loopback" ];
  boot.extraModprobeConfig = ''options v4l2loopback devices=1 video_nr=10 card_label="deepfake" exclusive_caps=1'';
  environment.systemPackages = [ pkgs.v4l2loopback.bin pkgs.v4l-utils pkgs.ffmpeg pkgs.quickshell ];
  services.udev.extraRules = ''
    ACTION=="add", SUBSYSTEM=="video4linux", ATTR{name}=="deepfake", RUN+="${pkgs.coreutils}/bin/chgrp video /sys%p/format", RUN+="${pkgs.coreutils}/bin/chmod g+w /sys%p/format"
  '';
}
```

The account must be allowed to write the virtual camera and change its rate. Deepfake announces the presented FPS and repeats the announcement after FFmpeg initializes the video format, which otherwise resets it to 30 FPS. Read back the device while the producer is active; `exclusive_caps=1` deliberately changes its advertised capabilities when the writer opens/closes it.

C2PA's official native wheel is optional in the isolated environment. Without it the CPU flake keeps visible disclosure. Signing certificates and keys live in `$XDG_CONFIG_HOME/deepfake/provenance`; keys are private (0600). The default development certificate is not on public trust lists.
