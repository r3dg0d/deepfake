"""Instrument official EditaLive inference without modifying its source.

Executed only in the configured worker environment. No server or camera.
"""
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--checkout", required=True, type=Path)
parser.add_argument("--metrics", required=True, type=Path)
arguments, upstream_args = parser.parse_known_args()
sys.path.insert(0, str(arguments.checkout))

import edit_streaming  # noqa: E402
import torch  # noqa: E402
from attention_sdpa import sdpa_attention  # noqa: E402
from editalive.models import (  # noqa: E402
    editalive_transformer3d,
    wan_image_encoder,
)
from editalive.models.modules import face_adapter  # noqa: E402
from editalive.pipeline import EditaLiveStreamingPipeline  # noqa: E402

# --vsa_backend sdpa only covers upstream's sparse branch. Its reference and
# cross-attention call FlashAttention directly and need the same dense fallback.
editalive_transformer3d.flash_attention = sdpa_attention
wan_image_encoder.flash_attention = sdpa_attention
face_adapter.flash_attn_func = sdpa_attention

metrics = {"backend": "editalive", "streaming": False, "precision": "FP8 with BF16 retained layers",
           "attention_backend": "sdpa", "quality_validation": "not evaluated"}
original_load = EditaLiveStreamingPipeline.from_pretrained
original_generate = EditaLiveStreamingPipeline.generate


@classmethod
def measured_load(cls, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = original_load(*args, **kwargs)
    torch.cuda.synchronize()
    metrics["model_load_s"] = time.perf_counter() - start
    return result


def measured_generate(self, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = original_generate(self, *args, **kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    frames, height, width = result
    metrics.update(inference_wall_s=elapsed, generated_frames=frames, width=width, height=height,
                   native_inference_fps=frames / elapsed, peak_vram_mb_torch=torch.cuda.max_memory_allocated()/2**20,
                   steady_vram_mb_torch=torch.cuda.memory_allocated()/2**20)
    return result


EditaLiveStreamingPipeline.from_pretrained = measured_load
EditaLiveStreamingPipeline.generate = measured_generate
sys.argv = [str(arguments.checkout / "edit_streaming.py"), *upstream_args]
start_all = time.perf_counter()
try:
    edit_streaming.generate(edit_streaming._parse_args())
finally:
    metrics["worker_wall_s"] = time.perf_counter() - start_all
    if torch.cuda.is_initialized():
        metrics["peak_vram_mb_torch"] = torch.cuda.max_memory_allocated()/2**20
        metrics["steady_vram_mb_torch"] = torch.cuda.memory_allocated()/2**20
    arguments.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
