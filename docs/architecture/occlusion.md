# Occlusion pipeline

Intended path for a consensual local swap. The blend must keep the original pixels wherever something covers the face (hand, hair, glasses, hat, mic).

```
face track
    → face parse
    → occluder
    → temporal filter
    → visible face mask   (face region minus occluder)
    → composite           (ellipse ∩ visible mask)
```

`FaceSwapPipeline.swap_frame` (webcam, virtualcam, and video) calls `OcclusionEngine.estimate` on the paste box and passes `visible_face_mask` into `paste_face` only when `apply_to_composite` is set. No new CLI flags.

## What is implemented

| Stage | Status |
| --- | --- |
| Face track | **Still Haar.** `OpenCVHaarDetector` only. `FaceBox.landmarks` stays empty. YuNet is unused. The engine accepts a box and ignores landmarks. |
| Face parse | **BiSeNet when it can run.** Looks for `bisenet_resnet_18.onnx` under the deepfake cache (`$XDG_CACHE_HOME/deepfake` or `~/.cache/deepfake`) and `models/vision` in the repo. Runtime is optional `onnxruntime` (`parse` extra / devShell). CUDA execution provider is used only if that build already has it; otherwise CPU. OpenCV DNN cannot run this export. If the file is missing, onnxruntime is missing, or the graph is not 19-class, the engine says `parser unavailable`, `apply_to_composite` is false, and paste keeps the ellipse. `deepfake doctor` shows Occlusion as failed in that case. The first swapped frame also prints `occlusion: inactive — …` on stderr. No download. |
| Occluder | **Class map below, plus optional XSeg, plus the color-key test double.** `combine_visible_mask` is `clip(face * (1 - occluder))`. |
| Temporal filter | **On the visible mask.** `smooth_visible_mask`, fed back through `temporal_state`. Rule below. No SAM. |
| Visible face mask | **Swap classes only** when BiSeNet runs (see the table). Fallback visible mask equals the ellipse and is **not** passed into `paste_face` (a second multiply would shrink the ellipse). |
| Composite | **Ellipse ∩ visible mask** when a mask is passed. Omitting it is the old ellipse. |

`parser_available` is true only after a BiSeNet forward. The color-key double sets `apply_to_composite` so tests can see a mask, and still reports the BiSeNet parser unavailable.

## CelebAMask-HQ labels

The ONNX file has no label list. The head is the usual 19-class CelebAMask-HQ order (face-parsing BiSeNet). A CPU forward of `src/deepfake/assets/bench_face.jpg` matches it: class 17 sits in the hair, class 1 in the face, class 16 on the clothes, class 0 in the background, brows above the eyes.

| id | name | blend |
| --- | --- | --- |
| 0 | background | stay target |
| 1 | skin | swap |
| 2 | l_brow | swap |
| 3 | r_brow | swap |
| 4 | l_eye | swap |
| 5 | r_eye | swap |
| 6 | eye_g (glasses) | occluder, stay target |
| 7 | l_ear | stay target |
| 8 | r_ear | stay target |
| 9 | ear_r (earring) | occluder, stay target |
| 10 | nose | swap |
| 11 | mouth | swap |
| 12 | u_lip | swap |
| 13 | l_lip | swap |
| 14 | neck | stay target |
| 15 | neck_l (necklace) | occluder, stay target |
| 16 | cloth | occluder, stay target |
| 17 | hair | occluder, stay target |
| 18 | hat | occluder, stay target |

Swap ids are `1, 2, 3, 4, 5, 10, 11, 12, 13`. The crop is resized to 512, ImageNet-normalized RGB, and the `output` logits are argmaxed. That label map is resized with nearest-neighbor back onto the paste box. `paste_face` then intersects it with the feathered ellipse.

## XSeg occluder (optional)

BiSeNet has no hand class. Hands become skin or background. Background already stays the target. Skin would be swapped unless something else punches it out.

`xseg_2.onnx` next to the BiSeNet file (this machine: `~/.cache/deepfake/models/vision/xseg_2.onnx`, about 68 MB) is a DeepFaceLab XSeg matte. Input is 256×256 RGB in 0..1, NHWC, output one channel. High means "face", not "hand". The occluder added on top of the class map is BiSeNet skin times `(1 - matte)`. Hair, cloth, and glasses stay on the class map. If the file is missing, onnxruntime is missing, or the graph will not run, that term is skipped and the mask is BiSeNet-only. Nothing is downloaded.

It does not run every frame. The first frame and every 4th frame recompute it (about 30 ms on CPU for the matte alone). Between those, a frame also recomputes when skin covers at least 12% of the crop and eyes/brows/nose/mouth/lips cover under 2% (the parser's skin mask flooded). Otherwise the last matte is held. This is not a hand detector and it does not claim SAM or SynthID.

## SAM 2.1 Hiera-Tiny (optional, not a proven hand mask)

The only SAM file downloaded is the official Meta asset `sam2.1_hiera_tiny.pt` from `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt` (156008466 bytes, sha256 `7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69`). It lives in `~/.cache/deepfake/models/vision/`. Apache-2.0. Not base, small, or large.

The AlphaFace swapper env is `/home/neo/.local/pipx/venvs/deepfake/bin/python` with torch **2.6.0+cu124** (CUDA reported false in this shell). `sam2` 1.0 was installed there with `pip install --no-deps` plus hydra-core, iopath, and portalocker. Torch stayed 2.6.0+cu124. The nix pytest env still has no torch and no sam2; doctor then says occlusion stays BiSeNet+XSeg and the webcam still runs.

In that pipx env, one CPU box prompt on `src/deepfake/assets/bench_face.jpg` (Haar box 228×228, prompt expanded 1.25×) took about **0.72 s** to predict after a **0.39 s** load. The face-crop mask mean was 0.72. The inner 60% was 0.9998 and the center was 1.0, so it does not delete the face. A flat skin-colored rectangle on the cheek fell from 0.87 to 0.003 while the inner face stayed about 0.83. The live interval (every 8 frames, or when skin floods) therefore stays enabled: skin outside that mask is an occluder only when coverage is at least 55% and the hole is under 35% of the skin. `deepfake doctor` in this env reports the checkpoint importable and that a mask which does not cover the face is discarded.

This portrait has no hand, mic, cup, or phone. The split is "main object versus a flat patch," not a labeled hand. Do not read it as those objects being masked in the wild.

## Temporal rule

`smooth_visible_mask(raw, temporal_state)` runs on every parser or test-double mask. The ellipse fallback does not smooth and returns the same `temporal_state` object it was given (`None` stays `None`). The pipeline stores the returned state per face and passes it on the next frame. `None` means no history.

Let `prev` be `temporal_state["visible"]` when it exists and matches this frame's shape. Otherwise the raw mask is kept and stored.

1. **Dropout (one frame).** If `prev` covers at least 32 pixels and `raw.sum() < 0.05 * prev.sum()`, the output is `prev` and `dropout_holds` becomes 1. A single empty parse does not zero the composite. Confidence on that result is still the raw mean (near 0); the held mask is what is composited.
2. **Second empty frame.** `dropout_holds` is already 1, so the empty mask is not held again. It falls through to the EMA and closes quickly.
3. **Occlusion arriving** (pixel `raw < prev`, including a hand-sized hole that does not wipe the whole mask). `out = prev + 0.9 * (raw - prev)`. A pixel at 1 with raw 0 becomes 0.1 in one frame and 0.01 in the next. The hole is trusted; it is not smeared shut.
4. **Occlusion leaving** (pixel `raw > prev`). `out = prev + 0.35 * (raw - prev)`. Opening blends back slower than closing. A pixel that has reached 0.1 and then sees raw 1 becomes about 0.42, not 1.

Shape changes drop history. There is no multi-frame hole fill.

## Later

1. SAM 2.1 Hiera-Tiny is cached. It is not a proven hand mask yet. See the SAM section. Base, small, and large were not downloaded.
2. SegFace-Mobile as an alternate small parser. Not this commit.
3. VideoSeal is still later. Finished files can get a C2PA manifest (`deepfake provenance inspect`); the on-screen label stays. Nothing here claims a platform will show a credential.

Debug overlays are not part of this commit.
