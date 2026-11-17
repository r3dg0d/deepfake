# Rotoscoping and background replacement (LivePortrait head engine)

LivePortrait generates a square crop: the reference character *and the reference image's own
background*. Until 0.5 that whole square was blended over the camera frame. With the matte model
installed, the character is cut out instead.

```bash
deepfake models install pphumanseg --yes     # 5.9 MB, Apache-2.0, SHA-256 pinned
deepfake webcam --engine experimental --mode head --reference avatar.png
deepfake webcam --engine experimental --mode head --reference avatar.png --background blur
deepfake virtualcam --engine experimental --mode head --reference avatar.png --background green
deepfake video -i in.mp4 -o out.mp4 --engine experimental --mode head --reference avatar.png \
  --background ~/Pictures/office.jpg --matte-scope head
```

| Flag | Meaning |
|---|---|
| `--rotoscope / --no-rotoscope` | Default: on when `pphumanseg` is installed. `--no-rotoscope` restores the soft rectangle. |
| `--background keep` | Default. The camera background stays; only the character is matted in. |
| `--background blur` | Blurred camera image behind the subject. |
| `--background green\|blue\|black\|white\|#rrggbb` | Solid colour (`green` is a chroma key for OBS). |
| `--background IMAGE` | An image file, scaled to cover the frame. |
| `--matte-scope person` | With a replaced background: keep the generated head and the real body below the chin (default). |
| `--matte-scope head` | Keep only the generated head. |

The flags are rejected for AlphaFace and the offline engines, which have no head crop to matte.
Asking for a background without the weights is an error rather than a session that shows the room
you wanted hidden. In the control center the same settings are in **Studio → Matte**.

## How it works

1. **Segmentation.** PP-HumanSeg (192×192) runs through `cv2.dnn` on the CPU, so it does not compete
   with LivePortrait for the GPU. The generated crop is segmented after inference; the camera crop
   and, for background replacement, the whole camera frame are segmented on worker threads while
   the GPU works.
2. **Matte.** The probabilities are upsampled against the image with a guided filter so the edge
   follows the image edge, sharpened with a smoothstep, and smoothed over time in crop space
   (small changes are averaged, large ones followed immediately).
3. **Composite.** The head goes in through the matte, with a soft window at the crop border.
4. **Clean plate** (`--background keep`). The real head is often larger than the character's. Pixels
   seen as background are remembered; where the real head sticks out above the chin they are shown
   instead. A pixel the camera has never seen uncovered cannot be filled and keeps the real image.
5. **Background** (anything but `keep`). Foreground is the head matte plus, for `person`, the camera
   person matte below the chin. Above the chin only the character counts, so the real hairline does
   not show around it.

## Limits

- It is a small segmentation network, not a studio key: fine hair, motion blur and hands in front
  of the face are approximate.
- Real hair below the chin or outside the head crop stays in `person` scope.
- Video sinks carry no alpha channel; "removed" means black. Use `green` for keying.
- **Cost.** In a 4-core container without a GPU the compositing stage took 11–13 ms per frame at
  640×480, about 7.5 ms of it the segmentation of the generated crop. It has not been measured next
  to LivePortrait on a GPU workstation; the Performance page shows the real figure (`Matte`).
- OpenCV 5.0's new DNN graph engine returns a shifted matte for this network. The segmenter selects
  the classic engine, and `deepfake models verify pphumanseg --load` checks registration by
  mirroring the test image.
