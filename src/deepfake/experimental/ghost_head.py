"""Private offline GHOST 2.0 experiment. Uses released inference tensors only.

Upstream: https://github.com/ai-forever/ghost-2.0 (Apache-2.0).
StyleMatte: https://github.com/chroneus/stylematte (CC-BY-SA-4.0).
No trained model weights are bundled with this adapter.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import yaml
from PIL import Image
from scipy.spatial import cKDTree

from deepfake.detect import create_detector

DEVICE = torch.device("cpu")
ROOT = Path(os.environ.get("DEEPFAKE_GHOST_RUNTIME", "ghost-runtime")).resolve()
REPO = ROOT / "ghost-2.0"
sys.path.insert(0, str(REPO))
pkg = types.ModuleType("src.aligner")
pkg.__path__ = [str(REPO / "src/aligner")]
sys.modules["src.aligner"] = pkg


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_submodel(model, state, prefix):
    # Every inference tensor must match; training-only tensors are intentionally unused.
    subset = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    model.load_state_dict(subset, strict=True)
    return model.eval().to(DEVICE)


def similarity(points, target):
    a = points - points.mean(0)
    b = target - target.mean(0)
    u, s, v = np.linalg.svd(b.T @ a / len(a))
    sign = np.ones(2)
    sign[-1] = np.sign(np.linalg.det(u @ v))
    rot = (u * sign) @ v
    scale = (s * sign).sum() / (a * a).sum() * len(a)
    return np.column_stack((scale * rot, target.mean(0) - scale * rot @ points.mean(0)))


def main(args):
    import src.blender.generator as blender_module
    from src.aligner.generator import Generator
    from src.aligner.iresnet import iresnet50
    from src.blender.generator import BlenderGenerator
    from src.utils.inference import copy_head_back
    from src.utils.preblending import dilate_torch, get_mask

    # Upstream helper methods omit these imports. Flat grayscale morphology is
    # evaluated with OpenCV on CPU to avoid 61x61 unfold tensors of several GB.
    def morphology(tensor, kernel, operation):
        data = tensor.detach().cpu().numpy()
        element = kernel.detach().cpu().numpy().astype(np.uint8)
        arrays = [cv2.morphologyEx(x, operation, element) for x in data.reshape(-1, *data.shape[-2:])]
        return torch.from_numpy(np.stack(arrays).reshape(data.shape)).to(tensor)

    blender_module.np = np
    blender_module.kornia_morphology = types.SimpleNamespace(
        closing=lambda tensor, kernel: morphology(tensor, kernel, cv2.MORPH_CLOSE),
        dilation=lambda tensor, kernel: morphology(tensor, kernel, cv2.MORPH_DILATE),
    )
    torch.set_num_threads(6)
    start = time.perf_counter()
    cfg = yaml.safe_load((REPO / "configs/aligner.yaml").read_text())["model"]
    state = torch.load(ROOT / "weights/aligner_1020_gaze_final.ckpt", map_location="cpu", weights_only=True, mmap=True)
    por = torchvision.models.resnext50_32x4d(weights=None)
    por.fc = torch.nn.Linear(2048, 512)
    pose = torchvision.models.mobilenet_v2(weights=None)
    pose.classifier[-1] = torch.nn.Linear(1280, 256)
    por = load_submodel(por, state, "embedder.por_encoder.")
    pose = load_submodel(pose, state, "embedder.pose_encoder.")
    identity = load_submodel(iresnet50(fp16=True), state, "embedder.id_encoder.")
    gen = load_submodel(Generator(**cfg["gen"], **cfg["embed"]), state, "gen.")
    del state
    # FPN weights are in the blender checkpoint; avoid a redundant ImageNet download.
    original_vgg = torchvision.models.vgg19
    torchvision.models.vgg19 = lambda **kwargs: original_vgg(weights=None)
    try:
        blender = BlenderGenerator()
    finally:
        torchvision.models.vgg19 = original_vgg
    state = torch.load(ROOT / "weights/blender_lama.ckpt", map_location="cpu", weights_only=True, mmap=True)[
        "state_dict"
    ]
    blender = load_submodel(blender, state, "gen.")
    del state
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    sm = load_module("stylematte_models", ROOT / "stylematte_models.py")
    matte = sm.StyleMatte.__new__(sm.StyleMatte)
    torch.nn.Module.__init__(matte)
    matte.fpn = sm.FPN_fuse(feature_channels=[256] * 4, fpn_out=256)
    config = Mask2FormerConfig.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    matte.pixel_decoder = Mask2FormerForUniversalSegmentation(config).base_model.pixel_level_module
    matte.fgf = sm.FastGuidedFilter(eps=1e-4)
    matte.conv = torch.nn.Conv2d(256, 1, 3, padding=1)
    matte.load_state_dict(
        torch.load(ROOT / "weights/stylematte_synth.pth", map_location="cpu", weights_only=True), strict=True
    )
    matte = matte.eval().to(DEVICE)
    import onnxruntime as ort

    parser = ort.InferenceSession(str(ROOT / "weights/segformer_B5_ce.onnx"), providers=["CPUExecutionProvider"])
    os.environ["LAMA_MODEL"] = str(ROOT / "weights/big-lama.pt")
    import src.utils.inpainter as inpainter_module
    from simple_lama_inpainting import SimpleLama
    from src.utils.inpainter import LamaInpainter

    inpainter_module.SimpleLama = lambda: SimpleLama(device=DEVICE)
    inpainter = LamaInpainter()
    detector = create_detector("auto")
    # Read exact upstream alignment templates without importing training-side EMOCA.
    croptext = (REPO / "src/utils/crops.py").read_text()
    scope = {"np": np}
    exec(croptext[croptext.index("src1 =") : croptext.index("# lmk is prediction")], scope)

    def crop_image(path):
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError("Cannot read " + str(path))
        boxes = detector.detect(frame)
        if not boxes:
            raise RuntimeError("No face in " + str(path))
        points = np.asarray(max(boxes, key=lambda b: b.w * b.h).landmarks)
        candidates = [similarity(points, t) for t in scope["wide_src_map"][512]]
        errors = [
            np.linalg.norm(points @ m[:, :2].T + m[:, 2] - t, axis=1).sum()
            for m, t in zip(candidates, scope["wide_src_map"][512], strict=False)
        ]
        M = candidates[int(np.argmin(errors))]
        arc = similarity(points, scope["arcface_src"][0])

        def tensor(img):
            return torch.from_numpy(img.copy()).permute(2, 0, 1)[None].float().to(DEVICE) / 127.5 - 1

        wide = tensor(cv2.warpAffine(frame, M, (512, 512)))
        arc = tensor(cv2.warpAffine(frame, arc, (112, 112)))
        return frame, wide, arc, M

    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[None, :, None, None]

    def imagemnet(x):
        return ((x[:, [2, 1, 0]] + 1) / 2 - mean) / std

    def mat(x):
        return matte(((x + 1) / 2 - mean) / std)

    def parsing(x):
        arr = (
            (x[:, [2, 1, 0]].cpu().numpy() / 2 + 0.5)
            - np.array([0.51315393, 0.48064056, 0.46301059])[None, :, None, None]
        ) / np.array([0.21438347, 0.20799829, 0.20304542])[None, :, None, None]
        return torch.from_numpy(parser.run(None, {parser.get_inputs()[0].name: arr.astype(np.float32)})[0]).to(
            DEVICE, dtype=torch.float32
        )

    source, sw, sa, _ = crop_image(args.source)
    with torch.inference_mode():
        smask = mat(sw)
        embeds = {"por_embed": por(imagemnet(sw * smask)), "id_embed": identity(sa[:, [2, 1, 0]]), "pose_embed": None}
        for targetpath in args.target:
            target, tw, ta, M = crop_image(targetpath)
            tmask = mat(tw)
            embeds["pose_embed"] = pose(tw * tmask)
            generated = gen(embeds)
            labels = parsing(tw)
            # Exact nearest-border operation with a CPU KD-tree, rather than allocating
            # the upstream dense N-head-pixels x N-border-pixels CUDA distance matrix.
            mask = dilate_torch(get_mask(labels), 15)
            ring = dilate_torch(mask, 3) - mask
            inside = np.argwhere(mask[0, 0].cpu().numpy() > 0)
            border = np.argwhere(ring[0, 0].cpu().numpy() > 0)
            if not len(border):
                raise RuntimeError("No surrounding background for this head crop")
            index = cKDTree(border).query(inside)[1]
            bg = tw.clone()
            blur = torchvision.transforms.GaussianBlur(3)(tw)
            bg[0, :, inside[:, 0], inside[:, 1]] = blur[0, :, border[index, 0], border[index, 1]]
            soft = mat((generated["fake_rgbs"] * generated["fake_segm"])[:, [2, 1, 0]])
            combined = generated["fake_rgbs"] * soft + bg * (1 - soft)
            gray = torchvision.transforms.functional.rgb_to_grayscale(combined[:, [2, 1, 0]])
            output = blender(
                combined,
                gray,
                tw,
                parsing(generated["fake_rgbs"] * generated["fake_segm"]),
                labels,
                gt=tw,
                cycle=False,
                train=False,
                return_inputs=True,
                inpainter=inpainter,
            )[0]
            image = (output[0].cpu().numpy().transpose(1, 2, 0)[..., ::-1] / 2 + 0.5) * 255
            result = copy_head_back(np.clip(image, 0, 255).astype(np.uint8), target[..., ::-1], M)
            # Experiments are visibly disclosed even before production export signing.
            cv2.putText(
                result,
                "SYNTHETIC HEAD EXPERIMENT",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )
            dest = Path(args.output) / ("ghost-" + Path(targetpath).stem + ".png")
            dest.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(result).save(dest)
            print(
                json.dumps(
                    {
                        "output": str(dest),
                        "elapsed_seconds": time.perf_counter() - start,
                        "max_cuda_bytes": torch.cuda.max_memory_allocated(),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", nargs="+", required=True)
    p.add_argument("--output", required=True)
    main(p.parse_args())
