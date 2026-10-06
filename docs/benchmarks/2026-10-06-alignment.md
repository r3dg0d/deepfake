# Alignment and remaining eye softness — 0.5.1rc5

The preceding bounded sharpening was insufficient. Read the actual remembered source setting and tested that photo alongside the earlier source, instead of assuming the earlier source was still selected. Private filenames and images stay outside this repository.

Compared direct 256/384/512 inference using existing weights. The convolutional model executes all three sizes, but 384/512 produced worse facial/eye artifacts; 512 took about 111 ms in this one-frame probe. These are out-of-training-distribution trials, not a released high-resolution checkpoint. They are rejected and not exposed as a quality mode.

The retained change fits a validated five-landmark similarity transform before target inference, warps the native ROI directly to the 256-pixel model template, and inversely warps the result into the original ROI. Source identity encoding now uses a measured five-point ArcFace 112 alignment when valid. Missing/invalid landmarks retain the prior crop fallback. Model weights and inference resolution are unchanged.

Current occlusion/mouth masks remain in their existing square-crop coordinates; inverse-warp coverage can only reduce visible alpha, including the alpha passed to FrameGen. Unmapped areas cannot paste extrapolated face pixels. Eye-detail masks are resized to the returned native ROI before the bounded filter.

The attached screenshot had eye separation of approximately **69.35** model pixels before versus **70.48** in the template, so the improvement is mainly correct alignment/source normalization and native-image sampling, **not** a substantial resolution increase. Screenshot replay showed clearer eye boundaries but is not an original-capture reconstruction or a validated general realism score. A single private actual-camera frame with a detected face was also compared in balanced/BF16 mode; it remains softer than high-resolution original eyes. This does not establish sustained live throughput or temporal superiority.

Full private clip with the currently selected source: 514 source/keyframes, 513 generated frames, one tail hold; 1028 frames at 59.998 FPS. Processing **36.25 s**, swap p50 **47.92 ms**, FrameGen p50 **10.82 ms**. These are offline measurements. Verified C2PA signature/video binding with publicly untrusted local certificate; TrustMark Q detected in 10/10 samples. Previous codec/watermark limits remain.

Regression checks cover inverse geometry, invalid landmarks, mapped coverage and exclusion of unmapped pixels from both composition and FrameGen. Full suite and separately selected GPU tests passed. No restoration model was downloaded or silently substituted. Eye texture absent from the model remains absent; a restoration model or better trained swapper needs separate identity/temporal validation.
