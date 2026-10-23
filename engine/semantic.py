# -*- coding: utf-8 -*-
"""Semantic portrait regions from MediaPipe selfie_multiclass.

A binary portrait matte answers one question ("is this the subject?"). A
*semantic* segmentation answers six: background, hair, body skin, face skin,
clothes, accessories. That difference is what makes region-specific retouching
possible instead of approximate:

* **skin operations** act on face skin and body skin, which replaces the
  chroma heuristic that fails on heavy make-up, coloured lighting and wigs
  (a chroma test cannot tell a red-lit cheek from a red dress);
* **hair** can be treated separately — previously it was defined only by
  exclusion (whatever the skin mask missed);
* **accessories** are kept during background replacement, so glasses, earrings
  and a necklace are not cut away with the backdrop;
* **the subject** is simply "everything that is not background", which is a
  better boundary than a padded face box and includes the accessories a matte
  may drop.

Model: MediaPipe `selfie_multiclass_256x256` (Apache-2.0). Runs through
MediaPipe's `ImageSegmenter`, so the tflite stays in its published form.
"""
from __future__ import annotations

from .paths import models_dir
import os
import threading
from typing import Dict, Optional

import numpy as np
import cv2

MODEL_NAME = "selfie_multiclass_256x256.tflite"

# published class order
BACKGROUND, HAIR, BODY_SKIN, FACE_SKIN, CLOTHES, OTHERS = range(6)
CLASS_NAMES = ("background", "hair", "body-skin", "face-skin", "clothes",
               "accessories")

_SEG = None
_LOCK = threading.Lock()


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


def available() -> bool:
    return os.path.exists(model_path())


def _segmenter():
    global _SEG
    if _SEG is not None:
        return _SEG
    with _LOCK:
        if _SEG is not None:
            return _SEG
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                "semantic model not found at %s. Download "
                "selfie_multiclass_256x256.tflite into models/." % path)
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True,
            output_confidence_masks=True)
        _SEG = vision.ImageSegmenter.create_from_options(opts)
    return _SEG


class Regions:
    """Per-class soft masks at the input's resolution, plus derived helpers.

    Masks are stored **as the model emitted them**. Their absolute scale is
    image-dependent: on a real photograph the skin classes reach ~0.95, but on
    a stylised or synthetic image the same model returns ~0.02 everywhere it
    is unsure. A fixed cut-off therefore fails silently — a 0.35 threshold
    turned a 0.043-confidence face into a zero gate, which switched skin
    processing off entirely without any error. Adaptive helpers below
    normalise against each mask's own distribution instead.
    """

    def __init__(self, masks: Dict[str, np.ndarray], category: np.ndarray):
        self.masks = masks
        self.category = category

    def __getitem__(self, key: str) -> np.ndarray:
        return self.masks[key]

    # ---- adaptive normalisation -----------------------------------------
    @staticmethod
    def pct_norm(m: np.ndarray, lo: float = 55.0, hi: float = 92.0) -> np.ndarray:
        """Rescale a confidence mask by its own percentiles.

        `lo` maps to 0 and `hi` maps to 1. Using percentiles rather than fixed
        values keeps the gate meaningful whatever the model's absolute
        confidence on this particular image happens to be.

        A flat mask (nothing detected) returns zeros rather than being
        stretched into noise, and a mask with real content keeps its shape.
        """
        a = np.clip(m, 0.0, 1.0)
        p_lo = float(np.percentile(a, lo))
        p_hi = float(np.percentile(a, hi))
        if p_hi - p_lo < 1e-3:
            # no dynamic range: either the class is absent or it covers
            # everything at one level; a hard decision on that is safer than
            # amplifying rounding error
            return (a >= max(p_hi, 0.5)).astype(np.float32)
        out = (a - p_lo) / (p_hi - p_lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def norm(self, name: str, lo: float = 55.0, hi: float = 92.0) -> np.ndarray:
        m = self.masks.get(name)
        if m is None:
            return np.zeros_like(self.category, np.float32)
        return self.pct_norm(m, lo, hi)

    def _union_norm(self, keys, lo: float, hi: float) -> np.ndarray:
        raw = None
        for k in keys:
            m = self.masks.get(k)
            if m is None:
                continue
            raw = m if raw is None else np.maximum(raw, m)
        if raw is None:
            return np.zeros_like(self.category, np.float32)
        # normalise once on the union so the classes keep their relative
        # strengths; normalising each class separately would make a class with
        # faint confidence read as strongly as a confident one
        return self.pct_norm(raw, lo, hi)

    # ---- derived region combinations ------------------------------------
    def subject(self, adaptive: bool = True) -> np.ndarray:
        """Everything that is not background: subject incl. accessories."""
        keys = ("hair", "body-skin", "face-skin", "clothes", "accessories")
        if adaptive:
            return self._union_norm(keys, 45.0, 88.0)
        return self.union(*keys)

    def skin(self, adaptive: bool = True) -> np.ndarray:
        """Both skin classes; the correct gate for smoothing and blemish work."""
        keys = ("face-skin", "body-skin")
        if adaptive:
            return self._union_norm(keys, 50.0, 90.0)
        return self.union(*keys)

    def upper_body(self, adaptive: bool = True) -> np.ndarray:
        """Skin + hair + clothes: the region a body reshape should act on."""
        keys = ("hair", "face-skin", "body-skin", "clothes")
        if adaptive:
            return self._union_norm(keys, 45.0, 88.0)
        return self.union(*keys)

    def union(self, *keys: str) -> np.ndarray:
        """Raw per-pixel maximum, no normalisation."""
        acc = None
        for k in keys:
            m = self.masks.get(k)
            if m is None:
                continue
            acc = m if acc is None else np.maximum(acc, m)
        if acc is None:
            return np.zeros_like(self.category, np.float32)
        return np.clip(acc, 0.0, 1.0)

    def confidence(self) -> float:
        """How sure the model is on this image, as the peak subject confidence.

        A caller can use this to decide whether the semantic gate is worth
        trusting or whether to fall back to a colour-based test.
        """
        keys = ("face-skin", "body-skin", "hair", "clothes")
        vals = [float(np.percentile(self.masks[k], 95))
                for k in keys if k in self.masks]
        return max(vals) if vals else 0.0

    def dominant(self) -> str:
        """Name of the class covering the most pixels (excluding background)."""
        best, best_frac = "background", 0.0
        for name, m in self.masks.items():
            if name == "background":
                continue
            f = float((m > 0.5).mean())
            if f > best_frac:
                best, best_frac = name, f
        return best

    def coverage(self) -> Dict[str, float]:
        return {k: round(float((v > 0.5).mean()), 4)
                for k, v in self.masks.items()}


def segment(img: np.ndarray, size: int = 0) -> Optional[Regions]:
    """Return semantic regions for a float32 RGB image.

    `size` optionally caps the long edge used for inference; the masks are
    returned at the input's resolution either way. The model's own input is
    256x256, so a larger internal size buys little.
    """
    if not available():
        return None
    import mediapipe as mp

    h, w = img.shape[:2]
    work = img
    if size and max(h, w) > size:
        s = size / float(max(h, w))
        work = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                          interpolation=cv2.INTER_AREA)

    u8 = (np.clip(work, 0, 1) * 255).astype(np.uint8)
    res = _segmenter().segment(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=u8))

    masks: Dict[str, np.ndarray] = {}
    conf = res.confidence_masks
    for i, name in enumerate(CLASS_NAMES):
        if conf is None or i >= len(conf):
            continue
        a = np.asarray(conf[i].numpy_view(), np.float32)
        if a.ndim == 3:
            a = a[..., 0]
        if a.shape[:2] != (h, w):
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        masks[name] = np.clip(a, 0.0, 1.0)

    cat = np.asarray(res.category_mask.numpy_view())
    if cat.ndim == 3:
        cat = cat[..., 0]
    if cat.shape[:2] != (h, w):
        cat = cv2.resize(cat, (w, h), interpolation=cv2.INTER_NEAREST)
    return Regions(masks, cat.astype(np.uint8))


def refine(mask: np.ndarray, img: np.ndarray, radius: int = 3) -> np.ndarray:
    """Snap a soft class mask onto the image's own edges.

    The network's masks are smooth and slightly soft at hair, which reads as a
    halo after a background replacement. Guided filtering with the luminance
    as guide keeps the transition on the real edge.
    """
    from .imgutil import guided_filter, luminance

    g = luminance(img)
    out = guided_filter(g, mask[..., None], radius=max(2, radius), eps=1e-4)
    if out.ndim == 3:
        out = out[..., 0]
    return np.clip(out, 0.0, 1.0).astype(np.float32)
