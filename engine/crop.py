# -*- coding: utf-8 -*-
"""Crop, straighten and rotate.

Kept out of the tone pipeline on purpose: geometry changes the image's *shape*,
and every later stage (face detection, masks, grain scale, sharpening radius) is
defined relative to the frame, so mixing a resize into the colour stages would
make each of those silently rescale. The pipeline takes an already-cropped
image; the session remembers the crop so it stays non-destructive and can be
reopened, changed, or removed at any time.

Aspect ratios are honoured exactly rather than approximately: a crop is stored
as floats in normalised coordinates and rounded to whole pixels at apply time,
so repeatedly opening and re-confirming a crop does not drift it inward.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# name -> width/height, or None for "free"
RATIOS: Dict[str, Optional[float]] = {
    "free": None,
    "original": None,          # resolved from the current shape
    "1:1": 1.0,
    "4:3": 4 / 3,
    "3:2": 3 / 2,
    "16:9": 16 / 9,
    "4:5": 4 / 5,
    "2:3": 2 / 3,
    "5:4": 5 / 4,
    "9:16": 9 / 16,
    "3:4": 3 / 4,
}


def ratio_value(name: str, shape: Tuple[int, int]) -> Optional[float]:
    if name == "original":
        h, w = shape[:2]
        return w / float(h)
    return RATIOS.get(name)


def clamp_box(box: Dict[str, float]) -> Dict[str, float]:
    """Force a normalised crop box inside the frame and non-degenerate."""
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 1.0))
    h = float(box.get("h", 1.0))
    w = min(max(w, 0.02), 1.0)
    h = min(max(h, 0.02), 1.0)
    x = min(max(x, 0.0), 1.0 - w)
    y = min(max(y, 0.0), 1.0 - h)
    return {"x": x, "y": y, "w": w, "h": h}


def fit_ratio(box: Dict[str, float], ratio: Optional[float],
              shape: Tuple[int, int]) -> Dict[str, float]:
    """Reshape a box to an aspect ratio, keeping its centre.

    Centre-preserving is the behaviour people expect when switching ratio: the
    subject stays put and the frame grows or shrinks around it, instead of the
    crop jumping to a corner.
    """
    box = clamp_box(box)
    if not ratio:
        return box
    h, w = shape[:2]
    # work in pixels so the ratio means real pixels, not normalised units
    px = box["w"] * w
    py = box["h"] * h
    cx = (box["x"] + box["w"] / 2) * w
    cy = (box["y"] + box["h"] / 2) * h
    # largest box with this ratio that fits the frame and the current box
    nw = min(px, w)
    nh = nw / ratio
    if nh > h:
        nh = h
        nw = nh * ratio
    if nw > w:
        nw = w
        nh = nw / ratio
    x = min(max(cx - nw / 2, 0.0), w - nw)
    y = min(max(cy - nh / 2, 0.0), h - nh)
    return clamp_box({"x": x / w, "y": y / h, "w": nw / w, "h": nh / h})


def pixel_box(box: Dict[str, float], shape: Tuple[int, int]
              ) -> Tuple[int, int, int, int]:
    """Normalised box -> integer (x0, y0, x1, y1) in pixels."""
    h, w = shape[:2]
    b = clamp_box(box)
    x0 = int(round(b["x"] * w))
    y0 = int(round(b["y"] * h))
    x1 = int(round((b["x"] + b["w"]) * w))
    y1 = int(round((b["y"] + b["h"]) * h))
    # never produce an empty region: a 0-pixel side would crash the encoder
    x0 = min(max(x0, 0), w - 2)
    y0 = min(max(y0, 0), h - 2)
    x1 = min(max(x1, x0 + 2), w)
    y1 = min(max(y1, y0 + 2), h)
    return x0, y0, x1, y1


def rotate(img: np.ndarray, degrees: float,
           expand: bool = True) -> np.ndarray:
    """Rotate by any angle.

    `expand=True` grows the canvas to hold the whole rotated image, which is
    what "straighten then crop" needs: expanding first and cropping second can
    always reach the corners, while rotating inside the original canvas clips
    them. `expand=False` keeps the frame and fills the wedges by reflection —
    useful for small levelling corrections where a hard edge would show.
    """
    d = float(degrees) % 360.0
    if abs(d) < 1e-6:
        return img.copy()
    h, w = img.shape[:2]
    if abs(d - 90) < 1e-6:
        return np.ascontiguousarray(np.rot90(img, 1))
    if abs(d - 180) < 1e-6:
        return np.ascontiguousarray(np.rot90(img, 2))
    if abs(d - 270) < 1e-6:
        return np.ascontiguousarray(np.rot90(img, 3))

    c = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(c, -d, 1.0)
    if expand:
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw = int(round(h * sin + w * cos))
        nh = int(round(h * cos + w * sin))
        m[0, 2] += nw / 2.0 - c[0]
        m[1, 2] += nh / 2.0 - c[1]
        border = cv2.BORDER_REFLECT_101
        return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                              borderMode=border)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT_101)


def apply(img: np.ndarray, crop: Optional[dict] = None,
          rotate_deg: float = 0.0, flip_h: bool = False,
          flip_v: bool = False) -> np.ndarray:
    """Apply the geometry in the order the UI presents it: rotate, flip, crop."""
    out = img
    if rot_deg := float(rotate_deg or 0.0):
        out = rotate(out, rot_deg, expand=True)
    if flip_h:
        out = np.ascontiguousarray(out[:, ::-1])
    if flip_v:
        out = np.ascontiguousarray(out[::-1])
    if crop:
        h, w = out.shape[:2]
        x0, y0, x1, y1 = pixel_box(crop, (h, w))
        out = np.ascontiguousarray(out[y0:y1, x0:x1])
    return out


def info() -> dict:
    return {"ratios": list(RATIOS.keys()),
            "supports": ["rotate", "straighten", "flip", "crop"]}
