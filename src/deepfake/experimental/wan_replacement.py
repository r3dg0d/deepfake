"""Submit an actual offline Wan-Animate replacement graph to the local runtime."""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .wan_inputs import load_controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8193")
    args = parser.parse_args()
    if urllib.parse.urlparse(args.server).hostname not in ("localhost", "127.0.0.1", "::1"):
        parser.error("only a local ComfyUI server is supported")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    controls = load_controls(args.controls)
    length = len(controls["background"])
    height, width = controls["reference"].shape[:2]
    if width % 32 or height % 32 or length % 4 != 1:
        parser.error("prepared dimensions must be multiples of 32; frame count must equal 4n+1")

    def node(name, **kwargs):
        return {"class_type": name, "inputs": kwargs}

    graph = {
        "1": node("LocalReplacementInputs", input_file=str(args.controls.resolve())),
        "2": node("UnetLoaderGGUF", unet_name="Wan2.2-Animate-14B-Q4_K_M.gguf"),
        "15": node(
            "LoraLoaderModelOnly",
            model=["2", 0],
            lora_name="wan2.2_animate_14B_relight_lora_bf16.safetensors",
            strength_model=1,
        ),
        "3": node("CLIPLoader", clip_name="umt5_xxl_fp8_e4m3fn_scaled.safetensors", type="wan", device="default"),
        "4": node(
            "CLIPTextEncode",
            clip=["3", 0],
            text="A realistic live-action video of the person in the reference image, wearing the reference clothing. Preserve the original camera angle, body movement, foreground objects and background.",
        ),
        "5": node(
            "CLIPTextEncode",
            clip=["3", 0],
            text="blurry face, blurry eyes, plastic skin, cartoon, extra fingers, distorted limbs, changed background",
        ),
        "6": node("VAELoader", vae_name="wan_2.1_vae.safetensors"),
        "7": node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors"),
        "8": node("CLIPVisionEncode", clip_vision=["7", 0], image=["1", 0], crop="none"),
        "9": node(
            "WanAnimateToVideo",
            positive=["4", 0],
            negative=["5", 0],
            vae=["6", 0],
            width=width,
            height=height,
            length=length,
            batch_size=1,
            continue_motion_max_frames=5,
            video_frame_offset=0,
            reference_image=["1", 0],
            clip_vision_output=["8", 0],
            background_video=["1", 1],
            pose_video=["1", 2],
            face_video=["1", 3],
            character_mask=["1", 4],
        ),
        "10": node("ModelSamplingSD3", model=["15", 0], shift=5),
        "11": node(
            "KSampler",
            model=["10", 0],
            seed=42,
            steps=20,
            cfg=1,
            sampler_name="uni_pc",
            scheduler="simple",
            positive=["9", 0],
            negative=["9", 1],
            latent_image=["9", 2],
            denoise=1,
        ),
        "12": node("TrimVideoLatent", samples=["11", 0], trim_amount=["9", 3]),
        "13": node("VAEDecode", samples=["12", 0], vae=["6", 0]),
        "14": node("SaveImage", images=["13", 0], filename_prefix="private-wananimate-body"),
    }
    (root / "replacement-api.json").write_text(json.dumps(graph, indent=2) + "\n")
    payload = json.dumps({"prompt": graph, "client_id": "local-deepfake-body-test"}).encode()
    req = urllib.request.Request(
        args.server.rstrip("/") + "/prompt", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        result = json.load(urllib.request.urlopen(req, timeout=30))
        print(json.dumps(result), flush=True)
    except urllib.error.HTTPError as error:
        print(error.read().decode(), flush=True)
        raise
    pid = result["prompt_id"]
    (root / "prompt.json").write_text(json.dumps(result) + "\n")
    start = time.perf_counter()
    while True:
        history = json.load(urllib.request.urlopen(args.server.rstrip("/") + "/history/" + pid, timeout=30))
        if pid in history:
            (root / "render-history.json").write_text(json.dumps(history[pid], indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "elapsed_seconds": time.perf_counter() - start,
                        "status": history[pid]["status"],
                        "outputs": history[pid].get("outputs"),
                    }
                ),
                flush=True,
            )
            if history[pid]["status"].get("status_str") != "success":
                sys.exit(1)
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
