# -*- coding: utf-8 -*-
"""Colour-wheel selection (SelectColor) and grey-point white balance.

SelectColor in the reference application is a colour wheel built on the six
CMYK primaries — red, green, blue, cyan, magenta, yellow. The user picks a
hue family and moves a slider; every pixel whose hue falls inside that
family's angular window shifts by the given amount. It is a *coarser* UI over
the same mechanism as the 8-band HSL (which uses named colour ranges), but
the family centres are the CMYK secondaries rather than photographic hues,
and the width is a single global control rather than per-band.

Grey-point white balance answers "make this pixel neutral": given a picker
position, estimate the illuminant from a small patch and correct the whole
frame. The estimate is the patch's channel ratios after removing the
luminance component, clamped so a colourless patch does not explode.

Both run in HSV/Lab respectively and reuse the project's existing colour
helpers, so results are consistent with the rest of the engine.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import cv2

from .imgutil import smoothstep

# CMYK wheel: hue centres in degrees (OpenCV hue is 0..360 in float images)
HUE_CENTRES = {
    "red": 0.0,
    "yellow": 60.0,
    "green": 120.0,
    "cyan": 180.0,
    "blue": 240.0,
    "magenta": 300.0,
}
# angular half-width of each family at width=0 (degrees)
BASE_HALF_WIDTH = 25.0


def _hue_weight(hue: np.ndarray, centre: float, width: float,
                feather: float) -> np.ndarray:
    """Angular window around `centre`, wrapping at 360.

    `width` (0..1) scales the half-width; `feather` is the fraction of the
    window spent falling off, so a wide window with a small feather has crisp
    edges and vice versa.
    """
    half = BASE_HALF_WIDTH * (0.35 + 2.2 * float(np.clip(width, 0.0, 1.0)))
    d = np.abs(np.mod(hue - centre + 180.0, 360.0) - 180.0)
    edge = half * float(np.clip(feather, 0.02, 1.0))
    return 1.0 - smoothstep(half - edge, half + edge, d)


def select_color(img: np.ndarray, family: str,
                 hue: float = 0.0, sat: float = 0.0, lum: float = 0.0,
                 width: float = 0.5, feather: float = 0.6) -> np.ndarray:
    """Shift one CMYK hue family.

    `family` is one of the six wheel keys. `hue` is a rotation in degrees,
    `sat` and `lum` percentage offsets, `width` how much of the wheel the
    family covers, `feather` how softly it blends into its neighbours.
    """
    if family not in HUE_CENTRES:
        raise ValueError("family must be one of %s" % ", ".join(HUE_CENTRES))
    if abs(hue) < 1e-6 and abs(sat) < 1e-6 and abs(lum) < 1e-6:
        return img.copy()

    src = np.clip(img, 0, 1).astype(np.float32)
    lum_before = src @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0] * 360.0, hsv[..., 1], hsv[..., 2]

    w = _hue_weight(h, HUE_CENTRES[family], width, feather)
    # luminance also gates the change slightly: pure blacks and whites have
    # no meaningful hue, and shifting them produces mud
    chroma_gate = smoothstep(0.03, 0.12, s)
    w = w * chroma_gate

    if abs(hue) > 1e-6:
        h = h + hue * w
    if abs(sat) > 1e-6:
        s = s * (1.0 + (sat / 100.0) * w)
    if abs(lum) > 1e-6:
        v = v * (1.0 + (lum / 100.0) * w * 0.7)

    hsv[..., 0] = np.mod(h, 360.0) / 360.0
    hsv[..., 1] = np.clip(s, 0.0, 1.0)
    hsv[..., 2] = np.clip(v, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # luminance compensation for the saturation edit, as in grade.py
    if abs(sat) > 1e-6 and abs(lum) <= 1e-6:
        after = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        gain = np.where(after > 1e-4, lum_before / np.maximum(after, 1e-4), 1.0)
        out = out * np.clip(gain, 0.5, 2.0)[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def grey_point(img: np.ndarray, x: float, y: float,
               radius: int = 6, strength: float = 1.0) -> Tuple[np.ndarray, dict]:
    """Neutralise the patch at (x, y) and correct the whole frame.

    The patch's channel ratios estimate the illuminant. Ratios are measured
    in linear light, normalised against their own mean so a mid-grey patch
    yields unity gains, and clamped so a near-neutral patch (the common case
    when the user clicks somewhere roughly grey) does not overcorrect.

    Returns (corrected image, info) where info carries the estimated gains
    and the patch statistics, so a UI can show what was assumed.
    """
    src = np.clip(img, 0.0, 1.0).astype(np.float32)
    h, w = src.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    r = max(1, int(radius))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    patch = src[y0:y1, x0:x1]
    if patch.size == 0:
        raise ValueError("picker position outside the image")

    # discard the extremes: a clipped highlight or a black pixel carries no
    # usable colour information for an illuminant estimate
    lum = patch @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    usable = (lum > 0.03) & (lum < 0.97)
    if usable.sum() < 8:
        usable = np.ones_like(lum, bool)
    means = patch[usable].reshape(-1, 3).mean(axis=0)

    grey = float(means.mean())
    gains = grey / np.maximum(means, 1e-5)
    gains = np.clip(gains, 0.6, 1.7)
    # normalise to keep exposure: gains are about colour, not level
    gains /= float(np.mean(gains))

    k = float(np.clip(strength, 0.0, 1.0))
    applied = 1.0 + k * (gains - 1.0)

    from .imgutil import srgb_to_linear, linear_to_srgb
    lin = srgb_to_linear(src) * applied[None, None, :]
    out = np.clip(linear_to_srgb(lin), 0.0, 1.0).astype(np.float32)

    # how far from neutral the patch was, for the caller to report
    cast = float(np.sqrt(((means / max(means.mean(), 1e-5) - 1.0) ** 2).sum()))
    return out, {"gains": [round(float(g), 4) for g in applied],
                 "patch_mean": [round(float(v), 4) for v in means],
                 "cast": round(cast, 4),
                 "patch_pixels": int(usable.sum())}
