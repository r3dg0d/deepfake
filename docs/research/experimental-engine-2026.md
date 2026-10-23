# Experimental engine research — cutoff 2026-10-20

Labels: **Verified** means source/metadata inspected; **Measured** means this
machine ran inference; **Estimated** is an inference, not a measurement;
**Experimental** means integration needs broader validation; **Unavailable**
means an essential reproducible component was not located at the checked source.
No unrelated weights are merged. GPU measurements are in the [benchmark report](../benchmarks/experimental-rtx4090.md).

| Candidate | Code / weights / licensing | VRAM and latency | Quality, motion and temporal evidence | Streaming / integration / 4090 | Maintenance |
|---|---|---|---|---|---|
| AlphaFace | Verified official code and locally installed weights; MIT code, undocumented weight terms | Measured: 720p native pipeline 29.6 FPS, median capture→sink 25.8 ms; 1,128.5 MiB peak Torch | Existing face-only identity cache, compositing and motion-aware smoothing retained; no new identity-quality evaluation | Existing live/offline path preserved, measured on 4090 | Existing wrapper tests pass; optional upstream imports |
| LivePortrait | Verified official MIT source and five SHA-256-pinned human animation weights; InsightFace excluded | Measured 640×480: 24.3 native FPS, 41.1 ms mean, 56.8 ms P95, 1,057.3 MiB peak Torch | Genuine neural head-crop animation inspected; pose/expression retargeting uses cached reference geometry. Crop boundaries visible; rotation, occlusions and drift unscored | Implemented in-process live/offline adapter; single subject, Haar alignment. Experimental on 4090 | Pinned June 2026 source; tested with host Torch 2.14.1; modest optional dependencies |
| PersonaLive | Verified official source and six weights; Apache-2.0 code/card metadata alongside academic-use notice. Required base checkpoint absent from audited official repository | Upstream claims low-memory offline streaming; **not measured here** | Expressive portrait and rolling temporal conditioning are upstream features, not validated integration outcomes | Offline worker implemented using official classes; installer blocked by missing base model. No working backend claim | Pinned August 2026 source; Python 3.10 / Torch 2.1 dependency stack isolated |
| EditaLive | Verified official inference/webcam code, Wan-Animate base plus compatible LoRAs; Apache-2.0 files/cards and academic-use notice reviewed independently | Upstream H100 low-memory results do not establish 4090 speed. Local selected bundle: 52.71 GiB disk; no measured VRAM/latency | Official conditioning preserves body/face motion; fidelity, pose and flicker not scored locally | Offline subprocess adapter and verified-download manifest implemented; genuine generation **unvalidated**. Webcam service not integrated | Pinned September 2026 source; Python 3.10 / Torch 2.6; source has FP8 and CPU offload options |
| PixReenact | Paper verified; no official runnable implementation/checkpoints located in checked paper/search | Paper reports 239 ms mean emission latency; not a 4090 measurement | Causal rollout/drift training described; no code/technique ported here | Unavailable as a reproducible backend; not registered | Too early to establish integration stability |
| DirectSwap | Verified official repository contains release plan/docs, no released inference or checkpoints | Not measured; no live claim | Whole-head silhouettes and background restoration motivate future evaluation only | Unavailable backend; no copied implementation | Track official release checklist |
| Wan-Animate / Wan-Animate-2 | Official models/source verified; Apache-2.0 cards. Animate-2 uses a different architecture | Not measured independently | Animation/replacement conditioning differs by generation | EditaLive uses Wan2.2-Animate-14B specifically; Animate-2 weights must not replace it | Animate-2 requires its own adapter/environment; no LoRA compatibility assumed |

The EditaLive loader constructs CPU T5, CLIP and the transformer before block
placement/FP8 conversion. Loading progressed to generation, but validation did
not complete. The user reported a host freeze requiring a forced reboot;
previous-boot logs show systemd-oomd killing terminal-session processes. Memory
exhaustion is suspected; exact process attribution is not established. GPU
offloading does not bound host RAM/swap consumption. The local backend and
validation script are disabled. Enforced RAM/swap containment and a separately
authorized validation are required before another run on this workstation.

Supporting components: existing OpenCV Haar detection and bounded queues are
reused. LivePortrait supplies implicit motion keypoints. EditaLive's worker uses
its official preprocessing/pose checkpoints. No new SMPL-X, SAM2, DWPose,
MediaPipe, optical-flow stabilizer, multi-person association or Gaussian avatar
implementation is claimed. A segmentation-aware compositor is needed before
claiming hand occlusion, neck or background preservation for the head adapter.

Official evidence inspected:

- [AlphaFace paper](https://arxiv.org/abs/2601.16429), [source](https://github.com/andrewyu90/Alphaface_Official).
- [LivePortrait source/license](https://github.com/KlingAIResearch/LivePortrait), [human weights](https://huggingface.co/KlingTeam/LivePortrait).
- [PersonaLive source](https://github.com/GVCLab/PersonaLive), [weights](https://huggingface.co/huaichang/PersonaLive), [required base repository](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).
- [EditaLive paper](https://arxiv.org/abs/2608.27123), [source and measured upstream configurations](https://github.com/GVCLab/EditaLive), [weights](https://huggingface.co/huaichang/EditaLive).
- [PixReenact paper](https://arxiv.org/html/2610.05233v1), [DirectSwap paper](https://arxiv.org/html/2512.09417v3), [DirectSwap release checklist](https://github.com/Yanan-Wang-cs/DirectSwap).
- [Wan-Animate weights](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B), [Wan-Animate-2 official source](https://github.com/Wan-Video/Wan-Animate-2).

Reproducibility: exact source revisions, individual checkpoint revisions, byte
sizes, SHA-256 digests and dependency notes are stored in
`src/deepfake/assets/experimental-models.json`. PersonaLive's base repository
revision `81226edde9d4f972f363f98e28dfe353d2b1e944` had only its initial files;
its revision history and refs did not expose the missing model files.
