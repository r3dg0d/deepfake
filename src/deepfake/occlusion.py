"""Occlusion-aware face mask. Visible face = face region minus occluder.

The live path calls ``OcclusionEngine``. When BiSeNet cannot run, the result
says the parser is unavailable and the composite keeps today's ellipse (the
visible mask is not applied). Nothing is downloaded.

An optional cached XSeg matte (``xseg_2.onnx``) can punch holes in BiSeNet skin
where the matte does not see a face (hands and mics BiSeNet calls skin). If that
file is missing or will not run, the mask stays BiSeNet-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .composite import _oval_soft_mask
from .detect import FaceBox
from .paths import cache_home, models_dir

# CelebAMask-HQ 19-class head used by face-parsing BiSeNet (zllrunning / the
# usual ResNet-18 ONNX export). This file does not store a label list; the
# order was checked on src/deepfake/assets/bench_face.jpg: hair is the top
# band, skin is the center, cloth is the bottom, brows sit above the eyes.
CELEBA_MASK_HQ_LABELS: tuple[str, ...] = (
    "background",  # 0
    "skin",  # 1
    "l_brow",  # 2
    "r_brow",  # 3
    "l_eye",  # 4
    "r_eye",  # 5
    "eye_g",  # 6 glasses
    "l_ear",  # 7
    "r_ear",  # 8
    "ear_r",  # 9 earring
    "nose",  # 10
    "mouth",  # 11
    "u_lip",  # 12
    "l_lip",  # 13
    "neck",  # 14
    "neck_l",  # 15 necklace
    "cloth",  # 16
    "hair",  # 17
    "hat",  # 18
)

# Swap these. Everything else stays the target frame.
SWAP_CLASS_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 11, 12, 13)
# Reported on the occluder channel. Ears, neck, and background also stay
# target, but they are not called occluders.
OCCLUDER_CLASS_IDS: tuple[int, ...] = (6, 9, 15, 16, 17, 18)

_WEIGHT_NAMES = ("bisenet_resnet_18.onnx",)
_XSEG_NAMES = ("xseg_2.onnx",)
_INPUT = 512
_XSEG_INPUT = 256
# Recompute the matte on this period. ~30 ms on CPU, so not every frame.
XSEG_INTERVAL = 4
# BiSeNet skin with almost no eyes/nose/mouth: a hand-sized flood. Forces a
# matte refresh between interval frames. A normal portrait stays on the interval.
XSEG_SKIN_FLOOD = 0.12
XSEG_FEATURE_FLOOR = 0.02
_FEATURE_CLASS_IDS = (2, 3, 4, 5, 10, 11, 12, 13)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OccluderSource(Protocol):
    """Something that marks occluding pixels. The test double implements this."""

    def occluder_mask(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        """Float32 mask, 1 where the swap must not paint. Same HxW as the frame."""


@dataclass
class OcclusionResult:
    face_mask: np.ndarray
    occluder_mask: np.ndarray
    visible_face_mask: np.ndarray
    confidence: float
    temporal_state: Any
    parser_available: bool
    detail: str
    # False on the ellipse fallback so paste_face is not given a second ellipse.
    apply_to_composite: bool = False


# Asymmetric temporal filter on the visible-face mask. See docs/architecture/occlusion.md.
# Weight on the new sample. Smaller mask (occlusion arriving) moves fast; a larger
# mask (occlusion leaving) moves slower. A near-empty frame is held once.
MASK_OCCLUDE_ALPHA = 0.9
MASK_REVEAL_ALPHA = 0.35
MASK_DROPOUT_RATIO = 0.05
MASK_DROPOUT_MIN_AREA = 32.0


def smooth_visible_mask(
    raw: np.ndarray,
    temporal_state: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Smooth one visible-face mask. ``temporal_state`` from the previous result, or None.

    Dropout (raw area < 5% of the previous area, and the previous area is real):
    keep the last mask for this frame only. A second empty frame is not held.

    Otherwise, per pixel ``out = prev + alpha * (raw - prev)`` with
    ``alpha = 0.9`` where ``raw < prev`` and ``alpha = 0.35`` where ``raw > prev``.
    """
    current = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    prev = _previous_visible(temporal_state, current.shape)
    if prev is None:
        out = current.copy()
        return out, {"visible": out.copy(), "dropout_holds": 0}

    prev_area = float(prev.sum())
    raw_area = float(current.sum())
    held = int(temporal_state.get("dropout_holds", 0)) if isinstance(temporal_state, dict) else 0
    dropout = prev_area >= MASK_DROPOUT_MIN_AREA and raw_area < MASK_DROPOUT_RATIO * prev_area
    if dropout and held < 1:
        out = prev.copy()
        return out, {"visible": out.copy(), "dropout_holds": held + 1}

    smaller = current < prev
    alpha = np.where(smaller, MASK_OCCLUDE_ALPHA, MASK_REVEAL_ALPHA).astype(np.float32)
    out = np.clip(prev + alpha * (current - prev), 0.0, 1.0)
    return out, {"visible": out.copy(), "dropout_holds": 0}


def _previous_visible(temporal_state: Any, shape: tuple[int, ...]) -> np.ndarray | None:
    if not isinstance(temporal_state, dict):
        return None
    prev = temporal_state.get("visible")
    if prev is None:
        return None
    prev = np.asarray(prev, dtype=np.float32)
    if prev.shape != shape:
        return None
    return prev


def combine_visible_mask(face_mask: np.ndarray, occluder_mask: np.ndarray) -> np.ndarray:
    """Face region with occluder pixels removed. Both masks are float HxW in 0..1."""
    face = np.clip(np.asarray(face_mask, dtype=np.float32), 0.0, 1.0)
    occ = np.clip(np.asarray(occluder_mask, dtype=np.float32), 0.0, 1.0)
    if face.shape != occ.shape:
        raise ValueError(f"mask shape mismatch: face {face.shape} vs occluder {occ.shape}")
    return np.clip(face * (1.0 - occ), 0.0, 1.0)


def masks_from_label_map(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a CelebAMask-HQ label image to face, occluder, and visible masks.

    Visible pixels are only skin, brows, eyes, nose, mouth, and lips.
    Hair, cloth, glasses, hat, jewelry, background, ears, and neck stay out.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2:
        raise ValueError(f"label map must be HxW, got {lab.shape}")
    face = np.isin(lab, SWAP_CLASS_IDS).astype(np.float32)
    occ = np.isin(lab, OCCLUDER_CLASS_IDS).astype(np.float32)
    return face, occ, combine_visible_mask(face, occ)


def ellipse_face_mask(frame_bgr: np.ndarray, box: FaceBox, *, feather: int = 24) -> np.ndarray:
    """The current paste ellipse, placed in the full frame (zeros outside the box)."""
    fh, fw = frame_bgr.shape[:2]
    mask = np.zeros((fh, fw), dtype=np.float32)
    x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    x, y = max(0, x), max(0, y)
    w, h = x1 - x, y1 - y
    if w <= 1 or h <= 1:
        return mask
    roi = _oval_soft_mask(h, w, feather)
    mask[y:y1, x:x1] = roi[:h, :w]
    return mask


class ColorKeyOccluder:
    """Deterministic test double: pixels near ``color_bgr`` are occluders.

    This is not a face parser and does not load weights. Tests use it for a
    synthetic hand (or hair) swatch inside the ellipse.
    """

    def __init__(self, color_bgr: tuple[int, int, int], *, tolerance: int = 8) -> None:
        self.color = np.asarray(color_bgr, dtype=np.int16)
        self.tolerance = int(tolerance)

    def occluder_mask(self, frame_bgr: np.ndarray, box: FaceBox) -> np.ndarray:
        frame = np.asarray(frame_bgr)
        diff = np.abs(frame.astype(np.int16) - self.color.reshape(1, 1, 3))
        hit = np.all(diff <= self.tolerance, axis=2)
        mask = np.zeros(frame.shape[:2], dtype=np.float32)
        x, y, w, h = int(box.x), int(box.y), int(box.w), int(box.h)
        fh, fw = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(fw, x + w), min(fh, y + h)
        if x1 <= x0 or y1 <= y0:
            return mask
        mask[y0:y1, x0:x1] = hit[y0:y1, x0:x1].astype(np.float32)
        return mask


def xseg_should_run(frame_index: int, skin_frac: float, feature_frac: float, *, interval: int = XSEG_INTERVAL) -> bool:
    """True on the interval, and when skin flooded the crop between those frames."""
    if interval <= 1 or frame_index % interval == 0:
        return True
    return skin_frac >= XSEG_SKIN_FLOOD and feature_frac < XSEG_FEATURE_FLOOR


def xseg_occluder_from_keep(skin: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Occluder is BiSeNet skin the XSeg face matte rejected. Both are HxW 0..1."""
    skin_m = np.clip(np.asarray(skin, dtype=np.float32), 0.0, 1.0)
    keep_m = np.clip(np.asarray(keep, dtype=np.float32), 0.0, 1.0)
    if skin_m.shape != keep_m.shape:
        raise ValueError(f"xseg shape mismatch: skin {skin_m.shape} vs keep {keep_m.shape}")
    return np.clip(skin_m * (1.0 - keep_m), 0.0, 1.0)


def _weight_candidates(explicit: Path | None) -> list[Path]:
    if explicit is not None:
        return [explicit] if explicit.is_file() else []
    roots = [
        models_dir() / "vision",
        models_dir(),
        cache_home() / "vision",
        Path(__file__).resolve().parents[2] / "models" / "vision",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in _WEIGHT_NAMES:
            path = root / name
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                found.append(path)
    return found


def _xseg_candidates(explicit: Path | None) -> list[Path]:
    roots: list[Path] = []
    if explicit is not None and explicit.is_file():
        roots.append(explicit.parent)
    roots.extend(
        [
            models_dir() / "vision",
            models_dir(),
            cache_home() / "vision",
            Path(__file__).resolve().parents[2] / "models" / "vision",
        ]
    )
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        for name in _XSEG_NAMES:
            path = root / name
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                found.append(path)
    return found


def _ort_module():
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return ort


class _BiseNetSession:
    """Lazy onnxruntime session. CUDA if that provider is already in the build."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.session: Any = None
        self.provider = ""
        self.error = ""
        self._shape_ok = False

    def load(self) -> bool:
        if self.session is not None and self._shape_ok:
            return True
        if self.error and self.session is None:
            return False
        ort = _ort_module()
        if ort is None:
            self.error = (
                f"parser unavailable: {self.path.name} is installed but onnxruntime is not"
                " (optional extra 'parse'; no download attempted)"
            )
            return False
        try:
            available = list(ort.get_available_providers())
            chosen: list[str] = []
            if "CUDAExecutionProvider" in available:
                chosen.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                chosen.append("CPUExecutionProvider")
            if not chosen:
                self.error = (
                    f"parser unavailable: onnxruntime has no CPU or CUDA provider ({available})"
                )
                return False
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self.session = ort.InferenceSession(str(self.path), options, providers=chosen)
            self.provider = str(self.session.get_providers()[0])
            names = [o.name for o in self.session.get_outputs()]
            if "output" not in names:
                self.session = None
                self.error = (
                    f"parser unavailable: {self.path.name} has outputs {names}, not 'output';"
                    " not using this file"
                )
                return False
        except Exception as exc:
            self.session = None
            self.error = (
                f"parser unavailable: {self.path.name} could not be executed"
                f" ({type(exc).__name__}; no download attempted)"
            )
            return False
        return self._check_channels()

    def _check_channels(self) -> bool:
        try:
            dummy = np.zeros((1, 3, _INPUT, _INPUT), dtype=np.float32)
            out = self.session.run(["output"], {"input": dummy})[0]
        except Exception as exc:
            self.session = None
            self.error = (
                f"parser unavailable: {self.path.name} forward failed"
                f" ({type(exc).__name__}; no download attempted)"
            )
            return False
        if getattr(out, "ndim", 0) != 4 or int(out.shape[1]) != len(CELEBA_MASK_HQ_LABELS):
            shape = getattr(out, "shape", None)
            self.session = None
            self.error = (
                f"parser unavailable: {self.path.name} output shape {shape} is not"
                f" {len(CELEBA_MASK_HQ_LABELS)}-class CelebAMask-HQ; not using this file"
            )
            return False
        self._shape_ok = True
        self.error = ""
        return True

    def labels(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if not self.load() or self.session is None:
            raise RuntimeError(self.error or "parser unavailable")
        resized = cv2.resize(crop_bgr, (_INPUT, _INPUT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        norm = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        blob = np.transpose(norm, (2, 0, 1))[None]
        logits = self.session.run(["output"], {"input": blob})[0]
        pred = np.argmax(logits, axis=1)[0].astype(np.uint8)
        h, w = crop_bgr.shape[:2]
        if pred.shape != (h, w):
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        return pred


class _XSegSession:
    """DeepFaceLab XSeg matte. Output is high on the face and low on foreign pixels.

    This is not a hand-class model. The occluder is the skin BiSeNet kept where
    this matte is low. A missing or unloadable file is ignored.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.session: Any = None
        self.provider = ""
        self.error = ""

    def load(self) -> bool:
        if self.session is not None:
            return True
        if self.error:
            return False
        ort = _ort_module()
        if ort is None:
            self.error = f"xseg occluder unavailable: onnxruntime is not installed ({self.path.name})"
            return False
        try:
            available = list(ort.get_available_providers())
            chosen: list[str] = []
            if "CUDAExecutionProvider" in available:
                chosen.append("CUDAExecutionProvider")
            if "CPUExecutionProvider" in available:
                chosen.append("CPUExecutionProvider")
            if not chosen:
                self.error = f"xseg occluder unavailable: no CPU or CUDA provider ({available})"
                return False
            options = ort.SessionOptions()
            options.log_severity_level = 3
            self.session = ort.InferenceSession(str(self.path), options, providers=chosen)
            self.provider = str(self.session.get_providers()[0])
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            if not inputs or not outputs or inputs[0].name != "input" or outputs[0].name != "output":
                self.session = None
                self.error = f"xseg occluder unavailable: {self.path.name} is not an XSeg input/output graph"
                return False
            dummy = np.zeros((1, _XSEG_INPUT, _XSEG_INPUT, 3), dtype=np.float32)
            out = self.session.run(["output"], {"input": dummy})[0]
            if getattr(out, "ndim", 0) != 4 or int(out.shape[-1]) != 1:
                self.session = None
                self.error = f"xseg occluder unavailable: {self.path.name} output shape {getattr(out, 'shape', None)}"
                return False
        except Exception as exc:
            self.session = None
            self.error = f"xseg occluder unavailable: {self.path.name} could not be executed ({type(exc).__name__})"
            return False
        self.error = ""
        return True

    def keep_mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if not self.load() or self.session is None:
            raise RuntimeError(self.error or "xseg occluder unavailable")
        resized = cv2.resize(crop_bgr, (_XSEG_INPUT, _XSEG_INPUT), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        keep = self.session.run(["output"], {"input": rgb[None]})[0]
        matte = np.clip(keep[0, ..., 0], 0.0, 1.0).astype(np.float32)
        h, w = crop_bgr.shape[:2]
        if matte.shape != (h, w):
            matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
        return matte


class OcclusionEngine:
    """Estimate the visible-face mask for one tracked face.

    ``temporal_state`` may be ``None``. A parser mask is smoothed with
    ``smooth_visible_mask`` and the new state is returned. The ellipse fallback
    does not invent a state: the object you passed is returned as-is.
    """

    def __init__(
        self,
        parser: OccluderSource | None = None,
        *,
        feather: int = 24,
        weights_path: Path | None = None,
    ) -> None:
        self._parser = parser
        self.feather = feather
        self._weights_path = weights_path
        self._probe_detail: str | None = None
        self._backend: _BiseNetSession | None = None
        self._backend_ready = False
        self._xseg: _XSegSession | None = None
        self._xseg_probed = False
        self._xseg_note = ""
        self._xseg_hold: np.ndarray | None = None
        self._xseg_calls = 0

    def status(self) -> tuple[bool, str]:
        """(parser_available, detail) for doctor. Does not claim a parse it cannot run."""
        if self._parser is not None:
            return False, "deterministic test double; BiSeNet parser unavailable"
        self._ensure_backend()
        return self._backend_ready, self._probe_detail or "parser unavailable"

    def estimate(
        self,
        frame: np.ndarray,
        face_track: FaceBox,
        landmarks: Any = None,
        face_mask: np.ndarray | None = None,
        temporal_state: Any = None,
    ) -> OcclusionResult:
        del landmarks  # detectors do not fill these yet
        box = face_track
        if self._parser is not None:
            return self._from_test_double(frame, box, face_mask, temporal_state)
        parsed = self._from_bisenet(frame, box)
        if parsed is not None:
            face, occ, visible, detail = parsed
            smoothed, new_state = smooth_visible_mask(visible, temporal_state)
            return OcclusionResult(
                face_mask=face,
                occluder_mask=occ,
                visible_face_mask=smoothed,
                confidence=float(visible.mean()) if visible.size else 0.0,
                temporal_state=new_state,
                parser_available=True,
                detail=detail,
                apply_to_composite=True,
            )
        if face_mask is None:
            face = ellipse_face_mask(frame, box, feather=self.feather)
        else:
            face = self._as_frame_mask(face_mask, frame.shape[:2])
        occ = np.zeros(face.shape, dtype=np.float32)
        detail = self._probe_detail or "parser unavailable"
        if self._backend_ready:
            # Session is fine; this crop was empty. Do not describe that as an active parse.
            detail = "parser unavailable: empty face crop; using ellipse"
        return OcclusionResult(
            face_mask=face,
            occluder_mask=occ,
            visible_face_mask=combine_visible_mask(face, occ),
            confidence=0.0,
            temporal_state=temporal_state,
            parser_available=False,
            detail=detail,
            apply_to_composite=False,
        )

    def _from_test_double(self, frame, box, face_mask, temporal_state) -> OcclusionResult:
        if face_mask is None:
            face = ellipse_face_mask(frame, box, feather=self.feather)
        else:
            face = self._as_frame_mask(face_mask, frame.shape[:2])
        occ = np.clip(np.asarray(self._parser.occluder_mask(frame, box), dtype=np.float32), 0.0, 1.0)
        if occ.shape != face.shape:
            raise ValueError(f"occluder shape {occ.shape} != face {face.shape}")
        raw = combine_visible_mask(face, occ)
        smoothed, new_state = smooth_visible_mask(raw, temporal_state)
        return OcclusionResult(
            face_mask=face,
            occluder_mask=occ,
            visible_face_mask=smoothed,
            confidence=float(raw.mean()) if raw.size else 0.0,
            temporal_state=new_state,
            parser_available=False,
            detail="deterministic test double; BiSeNet parser unavailable (no model executed)",
            apply_to_composite=True,
        )

    def _from_bisenet(self, frame, box):
        self._ensure_backend()
        if not self._backend_ready or self._backend is None:
            return None
        x, y, x1, y1, crop = _clamped_crop(frame, box)
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        try:
            labels = self._backend.labels(crop)
        except Exception as exc:
            self._backend_ready = False
            self._probe_detail = (
                f"parser unavailable: {self._backend.path.name} inference failed"
                f" ({type(exc).__name__}); using ellipse"
            )
            return None
        face_roi, occ_roi, vis_roi = masks_from_label_map(labels)
        fh, fw = frame.shape[:2]
        face = np.zeros((fh, fw), dtype=np.float32)
        occ = np.zeros((fh, fw), dtype=np.float32)
        visible = np.zeros((fh, fw), dtype=np.float32)
        face[y:y1, x:x1] = face_roi
        occ[y:y1, x:x1] = occ_roi
        visible[y:y1, x:x1] = vis_roi
        extra, xseg_note = self._xseg_extra(crop, labels, (x, y, x1, y1), (fh, fw))
        if extra is not None:
            occ = np.maximum(occ, extra)
            visible = combine_visible_mask(face, occ)
        provider = self._backend.provider or "CPUExecutionProvider"
        detail = (
            f"BiSeNet CelebAMask-HQ on {provider}; swap classes"
            " skin/brows/eyes/nose/mouth/lips; hair/cloth/background stay target"
        )
        if xseg_note:
            detail = f"{detail}; {xseg_note}"
        return face, occ, visible, detail

    def _note_xseg_file(self) -> None:
        if self._xseg_note:
            return
        found = _xseg_candidates(self._weights_path)
        if found:
            self._xseg_note = f"xseg occluder file {found[0].name} (interval {XSEG_INTERVAL})"

    def _xseg_extra(
        self,
        crop: np.ndarray,
        labels: np.ndarray,
        box: tuple[int, int, int, int],
        hw: tuple[int, int],
    ) -> tuple[np.ndarray | None, str]:
        """Full-frame occluder from the XSeg matte, or None when it is not in use."""
        session = self._xseg_session()
        if session is None:
            return None, ""
        skin = (np.asarray(labels) == 1).astype(np.float32)
        feature = np.isin(labels, _FEATURE_CLASS_IDS).mean() if labels.size else 0.0
        skin_frac = float(skin.mean()) if skin.size else 0.0
        run = xseg_should_run(self._xseg_calls, skin_frac, float(feature))
        self._xseg_calls += 1
        x, y, x1, y1 = box
        if not run and self._xseg_hold is not None and self._xseg_hold.shape == hw:
            return self._xseg_hold, f"XSeg matte held ({session.path.name}, every {XSEG_INTERVAL} frames)"
        try:
            keep = session.keep_mask(crop)
        except Exception:
            self._xseg = None
            return None, ""
        extra_roi = xseg_occluder_from_keep(skin, keep)
        extra = np.zeros(hw, dtype=np.float32)
        extra[y:y1, x:x1] = extra_roi
        if self._xseg_hold is not None and self._xseg_hold.shape == hw:
            # Keep holes on faces this call did not refresh.
            stale = self._xseg_hold.copy()
            stale[y:y1, x:x1] = 0.0
            extra = np.maximum(extra, stale)
        self._xseg_hold = extra
        return extra, (
            f"XSeg matte on {session.provider} ({session.path.name}); "
            "skin the matte rejected is an occluder"
        )

    def _xseg_session(self) -> _XSegSession | None:
        if self._xseg_probed:
            return self._xseg if self._xseg is not None and self._xseg.session is not None else None
        self._xseg_probed = True
        found = _xseg_candidates(self._weights_path)
        if not found:
            return None
        session = _XSegSession(found[0])
        if not session.load():
            self._xseg = None
            return None
        self._xseg = session
        return session

    def _ensure_backend(self) -> None:
        if self._probe_detail is not None:
            return
        found = _weight_candidates(self._weights_path)
        if not found:
            self._probe_detail = "parser unavailable: no face-parsing weights installed"
            self._backend = None
            self._backend_ready = False
            return
        if self._backend is None:
            self._backend = _BiseNetSession(found[0])
        if self._backend.load():
            self._backend_ready = True
            self._probe_detail = (
                f"BiSeNet {self._backend.path.name} on {self._backend.provider}"
                " (CelebAMask-HQ 19 classes)"
            )
        else:
            self._backend_ready = False
            self._probe_detail = self._backend.error or "parser unavailable"
        self._note_xseg_file()
        if self._xseg_note and self._probe_detail and self._xseg_note not in self._probe_detail:
            self._probe_detail = f"{self._probe_detail}; {self._xseg_note}"

    @staticmethod
    def _as_frame_mask(face_mask: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
        face = np.clip(np.asarray(face_mask, dtype=np.float32), 0.0, 1.0)
        if face.ndim == 3:
            face = face[:, :, 0]
        if face.shape != hw:
            raise ValueError(f"face_mask shape {face.shape} != frame {hw}")
        return face


def _clamped_crop(frame: np.ndarray, box: FaceBox):
    fh, fw = frame.shape[:2]
    x, y = max(0, int(box.x)), max(0, int(box.y))
    x1, y1 = min(fw, int(box.x) + int(box.w)), min(fh, int(box.y) + int(box.h))
    if x1 <= x or y1 <= y:
        return x, y, x1, y1, frame[0:0, 0:0]
    return x, y, x1, y1, frame[y:y1, x:x1]
