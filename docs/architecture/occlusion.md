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

## What this commit implements

| Stage | Status |
| --- | --- |
| Face track | **Not this commit.** Live detect is still one-shot Haar (`OpenCVHaarDetector`). `FaceBox.landmarks` stays empty. YuNet is unused. The engine accepts a `FaceBox` and ignores landmarks. |
| Face parse | **Not live.** `OcclusionEngine.estimate` looks for `bisenet_resnet_18.onnx` under the deepfake cache (`$XDG_CACHE_HOME/deepfake` or `~/.cache/deepfake`) and `models/vision` in the repo. If the file is missing, or OpenCV DNN cannot run it, the result says `parser unavailable` and the face mask is today's ellipse. Logits are not turned into parts: an untested class map would punch holes in the face. No download. |
| Occluder | **Math + test double only.** `combine_visible_mask` is `clip(face * (1 - occluder))`. `ColorKeyOccluder` is a deterministic stand-in for tests (a flat color counts as occluder). It is not a parser. |
| Temporal filter | **Not implemented.** `temporal_state` is accepted and returned unchanged. `None` is valid. |
| Visible face mask | **Implemented** as the product above. With no occluder this equals the ellipse, which is current behavior. |
| Composite | **Implemented, opt-in.** `paste_face(..., visible_mask=)` multiplies the feathered ellipse by that mask (full frame or ROI). Omitting it keeps the old ellipse. `webcam`, `virtualcam`, and `video` do not pass a mask and grow no new flags. |

`parser_available` is false in this slice even when the test double draws an occluder. The double is not a face parser. Confidence is 0 when the engine is on the ellipse fallback, and 1 when the test double produced the occluder.

## Later

1. Explicit install of a small parser (BiSeNet or SegFace-Mobile). Wire CelebAMask-HQ-style ids only after a CPU fixture checks them. Intended groups, not applied now: face parts `1–8, 10–13`; occluders hair/hat/cloth/neck/`ear_r` (`17, 18, 16, 14, 9`).
2. Call `estimate` from `FaceSwapPipeline` and pass `visible_face_mask` into `paste_face` only once that parser is real.
3. Temporal filter on the mask (the state argument is the hook).
4. Optional SAM 2 tiny for hands and other objects the parser calls background.
5. Provenance (C2PA 2.4 via c2patool / c2pa-python, or VideoSeal) is a different change. The on-screen label stays.

Debug overlays are not part of this commit.
