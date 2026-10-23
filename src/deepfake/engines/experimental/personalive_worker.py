"""Isolated PersonaLive offline worker, using the official model classes.

Executed by the configured Python environment, never imported by the base CLI.
Portrait reenactment generates a reference-scene portrait, not an in-place head mask.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def run(args):
    sys.path.insert(0, str(args.checkout))
    import cv2
    import mediapipe as mp
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from omegaconf import OmegaConf
    from PIL import Image
    from src.liveportrait.motion_extractor import MotionExtractor
    from src.models.motion_encoder.encoder import MotEncoder
    from src.models.pose_guider import PoseGuider
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel
    from src.pipelines.pipeline_pose2vid import Pose2VideoPipeline_Stream
    from src.scheduler.scheduler_ddim import DDIMScheduler
    from src.utils.util import crop_face
    from transformers import CLIPVisionModelWithProjection

    if not torch.cuda.is_available():
        raise RuntimeError("PersonaLive requires CUDA in the isolated worker environment")
    device, dtype = "cuda", torch.float16
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    weights = args.weights / "pretrained_weights"
    base = weights / "sd-image-variations-diffusers"
    inference = OmegaConf.load(args.checkout / "configs/inference/inference_stage3.yaml")
    vae = AutoencoderKL.from_pretrained(str(weights / "sd-vae-ft-mse"), local_files_only=True).to(device, dtype=dtype)
    reference_unet = UNet2DConditionModel.from_pretrained(str(base), subfolder="unet", local_files_only=True).to(device, dtype=dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        str(base), "", subfolder="unet", unet_additional_kwargs=inference.unet_additional_kwargs,
    ).to(device, dtype=dtype)
    motion = MotEncoder().to(device, dtype=dtype).eval()
    guider = PoseGuider().to(device, dtype=dtype)
    extractor = MotionExtractor(num_kp=21).to(device, dtype=dtype).eval()
    encoder = CLIPVisionModelWithProjection.from_pretrained(str(base / "image_encoder"), local_files_only=True).to(device, dtype=dtype)
    for model, filename, strict in (
        (denoising_unet, "denoising_unet", False), (reference_unet, "reference_unet", True),
        (motion, "motion_encoder", True), (guider, "pose_guider", True),
        (denoising_unet, "temporal_module", False), (extractor, "motion_extractor", False),
    ):
        model.load_state_dict(torch.load(weights / "personalive" / f"{filename}.pth", map_location="cpu", weights_only=True), strict=strict)
        model.eval()
    scheduler = DDIMScheduler(**OmegaConf.to_container(inference.noise_scheduler_kwargs))
    pipe = Pose2VideoPipeline_Stream(vae=vae, image_encoder=encoder, reference_unet=reference_unet,
                                   denoising_unet=denoising_unet, motion_encoder=motion,
                                   pose_encoder=extractor, pose_guider=guider, scheduler=scheduler).to(device)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - start
    # The upstream stream pipeline uses four-frame temporal updates. Pad the final
    # update and trim afterwards instead of silently discarding the source tail.
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    count = len(frames)
    if not count:
        raise RuntimeError("Input contains no decodable frames")
    while len(frames) % 4:
        frames.append(frames[-1])
    reference = Image.open(args.reference).convert("RGB")
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as mesh:
        reference_face = Image.fromarray(crop_face(reference, mesh)).convert("RGB")
        faces = [Image.fromarray(crop_face(f, mesh)).convert("RGB") for f in frames]
    start = time.perf_counter()
    with torch.inference_mode():
        generated = pipe(frames, reference, faces, reference_face, 512, 512, len(frames),
                         num_inference_steps=4, guidance_scale=1.0,
                         generator=torch.Generator(device=device).manual_seed(42),
                         temporal_window_size=4, temporal_adaptive_step=4).videos
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - start
    output = generated[0, :, :count].permute(1, 2, 3, 0).float().clamp(0, 1).cpu().numpy()
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (512, 512))
    if not writer.isOpened():
        raise RuntimeError("Worker cannot create generated video")
    try:
        for frame in output:
            writer.write(cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    args.output.with_suffix(".metrics.json").write_text(json.dumps({
        "model_load_s": load_s, "inference_wall_s": infer_s, "source_frames": count,
        "native_inference_fps": count / infer_s, "peak_vram_mb_torch": torch.cuda.max_memory_allocated() / 2**20,
        "steady_vram_mb_torch": torch.cuda.memory_allocated() / 2**20,
        "resolution": "512x512", "mode": "offline portrait reenactment",
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkout", "weights", "reference", "video", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    run(parser.parse_args())
