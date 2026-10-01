# Signed disclosure and invisible watermark

## Export lifecycle

1. Resolve and validate the signer before work starts.
2. Process the video; optional TrustMark is applied **after** FrameGen, before compression.
3. Finish video encoding and mux original audio, bounded by actual video frames.
4. Decode ten evenly spaced frames of the encoded artifact and verify the expected TrustMark payload if enabled.
5. Embed a real C2PA manifest using the official SDK and ES256 signature.
6. Verify the signature, BMFF binding and failure statuses. Publish the verified file atomically to the requested pathname.

Failure leaves the prior requested output untouched and removes private staging files. A successful file has a cryptographically verified AI-modification statement, not merely arbitrary JSON metadata. The action uses the IPTC source type `compositeWithTrainedAlgorithmicMedia`; it records original input as a parent ingredient, model identification, face replacement, optional interpolation and transcoding. An invisible-disclosure result is a custom assertion with the actual implementation and decoded count; it is not falsely labeled a C2PA soft-binding identifier.

```bash
deepfake video -i input.mp4 -f face.png -o output.mp4
deepfake video -i input.mp4 -f face.png -o output.mp4 --provenance c2pa+watermark
deepfake video -i input.mp4 -f face.png -o output.mp4 --visible-watermark
deepfake provenance inspect output.mp4
```

`auto` signs when `[provenance]` is installed and optionally adds TrustMark when its package and verified assets are available. `c2pa+watermark` requires both, otherwise fails with installation instructions. Without C2PA, automatic mode retains visible disclosure. Webcam/virtual-camera pixel streams retain visible disclosure because they cannot carry the exported file's manifest. There is no new `--strip-provenance` feature; legacy hidden watermark toggles do not disable these disclosure rules.

## Signatures and trust are separate

A local root and leaf are generated once under `$XDG_CONFIG_HOME/deepfake/provenance`, using private-key mode 0600 and directory mode 0700. This proves the signer possesses the key and that signed video data is unchanged. It does **not** establish a public identity or membership in a platform's trust list. The inspector exposes `signature_valid`, `asset_binding_valid`, `valid`, `trusted` and the underlying SDK validation results; a local signature reports public trust false.

Configure your own reviewed credential via `DEEPFAKE_C2PA_CERT` and `DEEPFAKE_C2PA_KEY`. `--trust-cert` in inspection is an explicit local trust decision, not a declaration that the certificate is globally trusted. No external timestamp authority or automatic network credential fetch is configured. Changing signed media requires a new signature; tampering test flips an `mdat` byte and verification rejects it.

## Actual codec stress test

RTX 4090, TrustMark Q 0.9.2, a two-second fictional 640×640 face-swap export with H.264 NVENC, 60 FPS, original AAC audio. Ten equally spaced decoded frames; “present” requires at least half to decode the exact payload `DFv1AI` with upstream ECC. This small fixture does not establish a general detection probability or false-positive rate. One unmarked control did not decode the disclosure payload. Uncompressed fixture PSNR was 43.62 dB; encode cost 7.64 ms per frame.

| Transformation | TrustMark detections / 10 | C2PA still valid |
|---|---:|---|
| Original signed export | 10 | Yes, signature and content binding; local certificate untrusted |
| H.264 CRF 28 | 10 | No |
| AV1 CRF 35 (SVT-AV1) | 10 | No |
| H.264 300 kbps | 1 — failed presence threshold | No |
| Resize to 320×320, H.264 CRF 23 | 9 | No |
| Crop to 512×512, H.264 CRF 23 | 10 | No |
| Frame rate reduced to 30, H.264 CRF 23 | 10 | No |
| Stream-copy remux with metadata removed | 10 | No |

C2PA is removable metadata; it does not invisibly reappear after a remux. TrustMark survives some compression and edits but failed aggressive compression in this test. It is an image watermark applied to each frame, not SynthID, a signed identity claim or a universal video fingerprint. [Raw public measurement summary](benchmarks/2026-10-01.json) contains only fixture results, no private footage or signing material.

[SynthID's public documentation](https://deepmind.google/models/synthid/) does not provide a verified generic image/video embedder for this local swap pipeline. The public [SynthID text implementation](https://github.com/google-deepmind/synthid-text) does not change that. [X describes C2PA integration](https://help.x.com/en/business-and-advertising/brand-safety/industry-leadership-and-partnerships), but this tool cannot guarantee its UI will show an AI label, that it trusts the development signer, or that metadata survives a platform transcode. No X account upload was performed.
