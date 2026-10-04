# Research notes — 2026-10-03

Notes from reading the tree and public provenance work. Nothing in this file is a claim that those papers or tools are integrated here. Product version stays 0.4.0. `flake.nix` still says 0.2.1; that pin was left alone.

## What the webcam path actually does

`realtime.run_realtime` / `run_offline` (and `FaceSwapPipeline.swap_frame`) today:

1. Capture (camera or decoded video).
2. Face detect with OpenCV Haar (`create_detector` → `OpenCVHaarDetector`). Boxes are squared in `align_crop` / `face_square_box`.
3. Swap the crop with AlphaFace when those weights are installed, otherwise inswapper. There is no other swapper on this path.
4. `composite.paste_face` builds `_oval_soft_mask` (a feathered ellipse) and blends.
5. Optional NVIDIA Optical Flow frame generation (`framegen`, default when enabled). RIFE is gone.
6. On-screen disclosure via `watermark.apply_watermark` (a drawn label, not an invisible watermark).

There is still no face parser on that path. `FaceBox.landmarks` is never filled. `OpenCVYuNetDetector` exists and can read a YuNet file, but `create_detector` never returns it, and its `detect` drops the landmark columns.

## Occlusion slice

`FaceSwapPipeline.swap_frame` calls `OcclusionEngine` for webcam, virtualcam, and video. `paste_face` intersects the ellipse with the visible-face mask only when a parser actually produced one. With no working parser the ellipse is unchanged, and doctor plus one stderr line say occlusion is inactive. See `docs/architecture/occlusion.md`.

The cached `bisenet_resnet_18.onnx` runs through optional onnxruntime (CPU on the nixpkgs build; CUDA only if that provider is already in the build). It is not downloaded. SegFace, SAM 2, invisible watermark embedding, and C2PA signing are not in this tree.

## Provenance (read, not implemented)

**SynthID.** Google's product watermark covers image, audio, text, and video inside Google's own generators. There is no open SynthID video embedder we can ship. The public embedder is text-only: [google-deepmind/synthid-text](https://github.com/google-deepmind/synthid-text) and the Transformers logits processor. SynthID-Image (the 2025 deployment paper) describes Google's image/video-frame watermark; the embedder stays product-side. This repo does not embed or detect SynthID.

**C2PA.** The practical signer is the Content Authenticity Initiative stack: [c2pa-rs](https://github.com/contentauth/c2pa-rs) (`c2patool`) and [c2pa-python](https://github.com/contentauth/c2pa-python), speaking [C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/specs/C2PA_Specification.html) (April 2026, including live-video notes). That is a signed manifest beside or inside the file, not a pixel trick. We do not sign anything yet. We do not claim X, or any other platform, will show a content-credential label for files this tool writes.

**VideoSeal.** The practical open invisible video mark is Meta's VideoSeal ([facebookresearch/videoseal](https://github.com/facebookresearch/videoseal), arXiv:2412.09492, MIT). The same repo later added PixelSeal and ChunkySeal; those were only read. Nothing here embeds VideoSeal. The on-screen string in `watermark.py` is the disclosure we actually draw.

## Webcam recommendation (not this commit)

Keep AlphaFace as the swap. Add a small face parser for the blend mask: BiSeNet (CelebAMask-HQ, 19 parts) or SegFace-Mobile (MobileNetV3 SegFace, AAAI 2025). Either one should produce the face region; hair, hat, glasses, and cloth become occluders. Optional later: a tiny SAM 2 for occluders the parser misses (hands, microphones). Do not take that dependency until weights are an explicit install, same rule as AlphaFace.

A local `~/.cache/deepfake/models/vision/bisenet_resnet_18.onnx` (~51 MB) was already on this machine. OpenCV 4.13 DNN cannot execute that export. Optional `onnxruntime` can: the graph is 19-class CelebAMask-HQ (checked on `bench_face.jpg`). The nixpkgs runtime used here exposes CPU, not CUDA. No replacement weights were fetched.

## Out of scope for this webcam slice

Do not integrate LivingSwap, DynamicFace, Stand-In, FaceDancer, or a full-frame RAFT pass. CanonSwap and VFace are cinematic / offline quality paths, not this real-time composite. Frame generation stays NVIDIA Optical Flow, not RAFT.
