# Occlusion architecture (0.5.0)

The reusable `OcclusionEngine.estimate(frame, face_track, landmarks, face_mask, temporal_state)` consumes a canonical 256×256 face crop and returns `face_mask`, `occluder_mask`, `visible_mask`, measured certainty, status, per-track state and latency. Landmark conditioning is reserved, not fabricated. Our crop is a padded, axis-aligned square, not a reconstructed 3D face.

## Evidence and mask coordinates

YuNet produces five measured landmarks and a detector score. Its score is output column 14; the old adapter incorrectly treated an eye coordinate as confidence. Boxes and landmarks map back from the detector's small image. Stable track IDs use IoU association; pyramidal LK predicts motion between detections, rejects forward/backward errors and fits a robust similarity transform. Smoothing affects detector correction rather than delaying every motion update. State is removed after loss, and appearance cuts clear tracks. Extreme yaw is estimated only by a nose/eye horizontal ratio, not invented pose angles; a strong profile preserves the original.

BiSeNet RN18 selects facial classes, excluding hair, ears and glasses. XSeg produces current-frame visible-face evidence. XSeg runs on **every crop**, including parser propagation frames. Refreshing an old semantic mask alone cannot protect a newly entering hand. Mouth-aware BiSeNet parsing now runs every frame in every preset to follow articulation. Semantic-only adapters can still propagate between refreshes. Farneback backward flow at 128×128 transports previous semantic and visible masks into the current crop. Appearance mismatch forces fresh parsing. Detection, crop, mask and pasted ROI use the same coordinate geometry, including offscreen clipping.

Final support requires semantic face probability above 0.75 and XSeg above 0.85. A 5×5 erosion adds a conservative margin. A padded distance transform and inward smoothstep ramp feather support without extending it into a declared occluder. Fresh semantic mouth/lip classes are dilated slightly and protected with a second distance ramp, preserving target articulation without a hard oral boundary. Temporal alpha recovery mixes 85% current / 15% motion-warped previous alpha and takes the minimum with current alpha. Thus disappearance is immediate; recovery takes a short fade. Certainty below 0.50 or coverage below 2.5% preserves the target. Missing models, invalid shapes, non-finite masks and provider errors also preserve the target and report why.

These thresholds are conservative engineering choices validated on our fixtures, not calibrated probability claims. `occluder_mask` is withheld semantic face support, including its safety margin; it is not a universal object-category segmentation map. Hair outside the semantic face is preserved even when it does not appear in that debug mask.

## Compositing and temporal state

The swap backend still computes AlphaFace once per visible, reliable face. Source identity encoding is cached. LAB gains and offsets are estimated from confident visible pixels, bounded and smoothed separately per track. Color statistics are sampled at up to 128×128 and applied to the full ROI. Hidden pixels, hair and unrelated background do not influence these statistics. Previous swapped face pixels are not averaged, which avoids a stale expression or doubled eyes. Target illumination gradients are preserved only to the extent of the current swap and color matching; this is not a physically modeled relighting system.

The visible alpha is resized into the complete paste box, then clipped. Nearest-neighbor support is intersected with bilinear alpha so resizing cannot revive a zero mask across an occluder edge. No synthetic seam is allowed outside that support. Severe uncertainty prefers a smaller swap or the original frame.

## Frame generation

NVOFA estimates flow; PyTorch resamples on CUDA. This is interpolation with a hardware flow accelerator, **not diffusion or a newly downloaded generative model**. A corrected backward-resampling sign fixes motion moving in the wrong direction. Quality uses forward/backward consistency sampled at matching coordinates. Scene cuts choose a real keyframe.

The keyframe carries eight channels: swapped BGR, original BGR, visible alpha and paste region. All use the same interpolation grids. Generated visibility takes the conservative minimum of the two warped visible masks; the paste region uses the maximum. Hidden foreground is restored from the warped original after sharpening. This protects declared occlusions in interpolated frames. Flow errors and model errors remain possible; the mechanism cannot recover a foreground object missed by both mask models.

The capture/swap/FrameGen/pacer stages have bounded queues. Overload shedding reduces generated slots before abandoning input processing. New, generated, held, late and dropped counts are distinct. A 60 FPS sink can consist of repeated frames under overload; we never equate its rate with 60 unique frames. The live pipeline warms up the actual source face, then resets tracker and mask state so startup does not reuse synthetic tracks.

## Runtime and integrity

BiSeNet CUDA logits remain in a preallocated GPU tensor; only the reduced mask is copied out. XSeg's asymmetric ConvTranspose nodes otherwise partly fall back to CPU. An in-memory symmetric-convolution-plus-crop rewrite preserves the mathematical operator and is tested against CPU output. The downloaded ONNX file is unchanged. GPU kernels may differ numerically from CPU kernels; a separate random-input check observed maximum output difference 0.002568. Adapter inference uses ONNX models and does not execute vendor Python.

Model loads require exact bytes and SHA-256. Downloads are explicit, bounded, written to private temporary files and installed atomically. Hashes pin our reviewed downloads; they are not upstream cryptographic signatures. XSeg model GPL-3.0 and dataset terms remain applicable. We did not copy FaceFusion's separately licensed Python wrapper.

## Validation limits

Automated tests cover immediate entry, smooth recovery, full occlusion, clipped paste geometry, inference failures, detector-order changes, track loss, flow direction and generated foreground preservation. Real-model benchmarks use a fictional portrait with stylized shapes, known masks and contrasting substitute pixels; they isolate compositor correctness rather than measure every real-world hand or eyewear. The local webcam test included an actual hand, hair and head rotation. Cup, microphone, phone and eyewear were tested as stylized shapes, not a labeled live dataset.

We have no ground-truth ArcFace identity accuracy, LPIPS study, true alpha matte for individual hair strands, 3D pose ground truth or multi-person interaction study. Motion-compensated alpha transport and static-repeat stability are measured; raw moving-mask delta also includes real motion. SAM2/SAM-MT, learned flow, depth and restoration remain researched alternatives. No unvalidated cinematic preset is advertised.

## 0.5.1rc1 profile and timing corrections

YuNet uses a 0.6 detection threshold; measured detector scores retain their meaning. Quality mode selects the largest face by default, with explicit multi-face opt-in. Local target LAB chroma is estimated only within visible support; normalized masked convolution excludes hidden pixels from local statistics; bounded low-frequency luminance and target chroma transfer reduce lighting seams. Exposure clipping and profile seams remain possible. Fractional rates near an integer target multiplier use that multiplier instead of an unnecessary extra pass; stale heuristic choices are recomputed. The hidden `compare` command renders clips at source timestamps and clearly labels recomputed masks as diagnostics, not stored output-frame ground truth.

### Eye detail (0.5.1rc4)

BiSeNet exposes current eye classes alongside mouth/face regions from the same inference. The estimate carries an eye mask intersected with visible alpha. A bounded luminance detail filter operates on generated eye pixels before paste enlargement; the final visible matte still controls all compositing. No target eye pixels or restoration model are substituted. Area crop downsampling and cubic enlargement reduce avoidable resampling softness; AlphaFace's native 256-pixel synthesis remains the resolution limit.
