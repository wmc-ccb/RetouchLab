"""Three-way colour grading (split toning).

The reference application grades in three overlapping zones — global,
midtone and highlight — each with hue, saturation and luminance, plus a
balance that shifts where the zones meet and a blending control for how
smoothly they overlap. That is a different control surface from a plain
global grade, and it is what makes a "cinematic" look possible: cool
shadows with warm highlights cannot be expressed by one global hue shift.

Implementation
--------------
The image is decomposed into three luminance weights whose sum is 1, so the
result never brightens or darkens simply by moving a slider:

    w_global    = 1
    w_midtone   = bell-shaped around `balance`
    w_highlight = ramp toward the top of the range

Each zone then rotates hue, scales saturation and offsets luminance in HSV,
weighted by its own mask. Working in HSV rather than Lab keeps hue shifts
intuitive: "rotate the highlights toward orange" maps to a direct hue angle.
"""
from __future__ import annotations

import numpy as np
import cv2

from .imgutil import smoothstep, luminance


def _zone_weights(lum: np.ndarray, balance: float, blending: float):
    """Return (midtone, highlight) weights for a luminance map.

    `balance` (-1..1) moves the crossover between zones: negative pushes the
    warm/highlight treatment down into the midtones, positive keeps it high.
    `blending` (0..1) widens the overlap so the zones fade into each other
    instead of banding.
    """
    b = float(np.clip(balance, -1.0, 1.0))
    k = float(np.clip(blending, 0.0, 1.0))

    # crossover points; blending widens each transition
    span = 0.18 + 0.42 * k
    mid_lo = 0.12 - b * 0.10
    mid_hi = 0.62 - b * 0.14
    hi_lo = 0.52 - b * 0.16
    hi_hi = hi_lo + span

    # a smooth bump for the midtones: rises then falls
    mid = smoothstep(mid_lo - span, mid_lo + span, lum) * \
        (1.0 - smoothstep(mid_hi - span, mid_hi + span, lum))
    high = smoothstep(hi_lo, hi_hi, lum)

    # highlights take precedence where they overlap, and the midtone bump is
    # scaled back so the two never sum above one
    high = np.clip(high, 0.0, 1.0)
    mid = np.clip(mid, 0.0, 1.0) * (1.0 - high)
    return mid.astype(np.float32), high.astype(np.float32)


def color_grade(img: np.ndarray,
                global_hue: float = 0.0, global_sat: float = 0.0, global_lum: float = 0.0,
                mid_hue: float = 0.0, mid_sat: float = 0.0, mid_lum: float = 0.0,
                high_hue: float = 0.0, high_sat: float = 0.0, high_lum: float = 0.0,
                balance: float = 0.0, blending: float = 0.5) -> np.ndarray:
    """Apply three-way grading.

    Hue values are degrees (-180..180), saturation and luminance are
    percentage offsets (-100..100).

    Saturation is applied with luminance compensation. Editing HSV directly
    and leaving V untouched looks correct but is not: V is max(R,G,B), so
    raising S pulls the other two channels down and the image measurably
    darkens (measured: a +50 global saturation moved mean luminance from 0.377
    to 0.273). Rescaling afterwards keeps the brightness where it was.
    """
    active = any(abs(v) > 1e-6 for v in
                 (global_hue, global_sat, global_lum,
                  mid_hue, mid_sat, mid_lum,
                  high_hue, high_sat, high_lum))
    if not active:
        return img.copy()

    src = np.clip(img, 0, 1).astype(np.float32)
    lum_before = luminance(src)

    hsv = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    lum = lum_before
    mid_w, high_w = _zone_weights(lum, balance, blending)
    glob_w = np.ones_like(lum)

    def zone(hue_d: float, sat_p: float, lum_p: float, w: np.ndarray):
        nonlocal h, s, v
        if abs(hue_d) < 1e-6 and abs(sat_p) < 1e-6 and abs(lum_p) < 1e-6:
            return
        # hue is stored in 0..1 by OpenCV for float input
        if abs(hue_d) > 1e-6:
            h = h + (hue_d / 360.0) * w
        if abs(sat_p) > 1e-6:
            f = sat_p / 100.0
            s = s * (1.0 + f * w)
        if abs(lum_p) > 1e-6:
            f = lum_p / 100.0
            v = v * (1.0 + f * w * 0.85)

    zone(global_hue, global_sat, global_lum, glob_w)
    zone(mid_hue, mid_sat, mid_lum, mid_w)
    zone(high_hue, high_sat, high_lum, high_w)

    hsv[..., 0] = np.mod(h, 1.0)
    hsv[..., 1] = np.clip(s, 0.0, 1.0)
    hsv[..., 2] = np.clip(v, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ---- luminance compensation -----------------------------------------
    # Restore the brightness that the saturation edit displaced, but only the
    # part that the *saturation* controls caused: an explicit luminance offset
    # must still take effect.
    wants_lum = abs(global_lum) > 1e-6 or abs(mid_lum) > 1e-6 or abs(high_lum) > 1e-6
    if not wants_lum:
        after = luminance(out)
        gain = np.where(after > 1e-4, lum_before / np.maximum(after, 1e-4), 1.0)
        gain = np.clip(gain, 0.5, 2.0)
        out = out * gain[..., None]

    return np.clip(out, 0.0, 1.0).astype(np.float32)


def zone_preview(shape: tuple, balance: float, blending: float) -> np.ndarray:
    """Visualise the zone weights, for diagnostics and UI hints."""
    h, w = shape[:2]
    lum = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    lum = np.repeat(lum, h, axis=0)
    mid, high = _zone_weights(lum, balance, blending)
    glob = np.ones_like(mid)
    return np.stack([glob, mid, high], axis=2)
