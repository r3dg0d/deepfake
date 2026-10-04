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
| Occluder | **Class map below, plus the color-key test double.** `combine_visible_mask` is `clip(face * (1 - occluder))`. |
| Temporal filter | **Not implemented.** `temporal_state` is accepted and returned unchanged. `None` is valid. No SAM. |
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

## Later

1. Temporal filter on the mask (the state argument is the hook).
2. SAM 2 tiny for hands and mics the parser calls background or skin.
3. SegFace-Mobile as an alternate small parser. Not this commit.
4. Provenance (C2PA 2.4 via c2patool / c2pa-python, or VideoSeal) is a different change. The on-screen label stays. Nothing here claims a platform will show a credential.

Debug overlays are not part of this commit.
