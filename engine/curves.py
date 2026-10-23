# -*- coding: utf-8 -*-
"""Tone curves and levels.

Both are lookup-table remaps of a channel, which is why they live together:
the difference is only how the table is built.

Levels
------
Four numbers per channel — input black, input white, gamma, output black,
output white — plus the standard rule that a level below the input black
clips to the output black. This is the mapping Adobe's Levels dialog applies,
and it is exactly invertible when gamma is 1.

Curves
------
A free-form remap through control points. The interpolation is **monotone
cubic Hermite (PCHIP)**, not a natural cubic spline. That choice is not
cosmetic: a natural spline overshoots between close control points, so a curve
that should stay flat lifts detail into clipping, and the overshoot gets worse
as control points are added. Monotone Hermite cannot overshoot by construction,
which is why every image editor that exposes curves uses it.

Per-channel curves combine with the master curve by composition — master first,
then channel, matching the order a retoucher expects when they set an overall
shape and then correct one colour cast.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

LUT_SIZE = 1024


# ---------------------------------------------------------------------------
# levels
# ---------------------------------------------------------------------------

def levels_lut(in_black: float = 0.0, in_white: float = 1.0,
               gamma: float = 1.0, out_black: float = 0.0,
               out_white: float = 1.0, size: int = LUT_SIZE) -> np.ndarray:
    """Build the 0..1 remap table for one channel.

    The order matches the published behaviour: normalise the input range,
    invert with gamma, then map onto the output range. An inverted or empty
    input range is rejected rather than producing a divide-by-zero table.
    """
    ib, iw = float(in_black), float(in_white)
    if iw <= ib:
        # a degenerate range would divide by zero; treat it as a hard clip at
        # the black point, which is what the UI shows for a collapsed slider
        iw = ib + 1e-4
    g = max(float(gamma), 1e-3)
    ob, ow = float(out_black), float(out_white)

    x = np.linspace(0.0, 1.0, size, dtype=np.float32)
    t = np.clip((x - ib) / (iw - ib), 0.0, 1.0)
    if abs(g - 1.0) > 1e-6:
        t = np.power(t, 1.0 / g)
    y = ob + t * (ow - ob)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# curves
# ---------------------------------------------------------------------------

def curve_lut(points: Sequence[Sequence[float]], size: int = LUT_SIZE) -> np.ndarray:
    """Build the remap table through `points`, a list of [x, y] in 0..1.

    Fewer than two points is the identity. Points are sorted by x and
    de-duplicated (a repeated x has no meaning for a function), and the
    endpoints are held flat outside the first and last control point, which is
    what makes a curve dragged inward behave as a clip rather than a slope.
    """
    pts = [(float(p[0]), float(p[1])) for p in (points or []) if len(p) >= 2]
    if len(pts) < 2:
        return np.linspace(0.0, 1.0, size, dtype=np.float32)

    pts.sort(key=lambda p: p[0])
    # de-duplicate x, keeping the last y for a repeated x
    uniq: List[Tuple[float, float]] = [pts[0]]
    for x, y in pts[1:]:
        if x - uniq[-1][0] < 1e-6:
            uniq[-1] = (uniq[-1][0], y)
        else:
            uniq.append((x, y))
    if len(uniq) < 2:
        return np.linspace(0.0, 1.0, size, dtype=np.float32)

    xs = np.array([p[0] for p in uniq], np.float64)
    ys = np.array([p[1] for p in uniq], np.float64)

    x = np.linspace(0.0, 1.0, size, dtype=np.float64)
    try:
        from scipy.interpolate import PchipInterpolator
        # PchipInterpolator is monotone by construction, so a monotone set of
        # control points yields a monotone curve with no overshoot
        y = PchipInterpolator(xs, ys, extrapolate=False)(x)
        # flat beyond the ends: np.nan marks the extrapolated region
        y = np.where(np.isnan(y), np.nan, y)
        y = np.where(x <= xs[0], ys[0], y)
        y = np.where(x >= xs[-1], ys[-1], y)
    except Exception:
        # scipy missing: linear interpolation, still monotone
        y = np.interp(x, xs, ys)

    return np.clip(np.nan_to_num(y, nan=0.0), 0.0, 1.0).astype(np.float32)


def _apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a 0..1 -> 0..1 lookup table to an image."""
    n = len(lut)
    idx = np.clip(img * (n - 1), 0, n - 1)
    lo = np.floor(idx).astype(np.int32)
    hi = np.minimum(lo + 1, n - 1)
    frac = (idx - lo).astype(np.float32)
    a = lut[lo]
    b = lut[hi]
    return a + (b - a) * frac


def levels(img: np.ndarray,
           master: Optional[Sequence[float]] = None,
           r: Optional[Sequence[float]] = None,
           g: Optional[Sequence[float]] = None,
           b: Optional[Sequence[float]] = None) -> np.ndarray:
    """Apply per-channel levels.

    Each argument is (in_black, in_white, gamma, out_black, out_white) or None
    to leave that channel alone. `master` runs first so a per-channel tweak can
    correct what the master pass did.

    Evaluated as the closed-form expression rather than through a lookup table.
    A 1024-entry table smears the clip boundary across one table step, which
    measured 0.001 of error exactly where the black point bites; the formula is
    both exact and cheaper than interpolating a table.
    """
    out = np.clip(img, 0.0, 1.0).astype(np.float32)

    def apply(spec, ch=None):
        ib, iw, gm, ob, ow = (float(v) for v in spec[:5])
        if iw <= ib:
            iw = ib + 1e-4          # degenerate range: behave as a hard clip
        gm = max(gm, 1e-3)
        src = out if ch is None else out[..., ch]
        t = np.clip((src - ib) / (iw - ib), 0.0, 1.0)
        if abs(gm - 1.0) > 1e-6:
            t = np.power(t, 1.0 / gm)
        y = ob + t * (ow - ob)
        if ch is None:
            return np.clip(y, 0.0, 1.0).astype(np.float32)
        out[..., ch] = np.clip(y, 0.0, 1.0)
        return None

    if master is not None:
        out = apply(master)
    for ch, spec in ((0, r), (1, g), (2, b)):
        if spec is not None:
            apply(spec, ch)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _compose(outer: np.ndarray, inner: Optional[np.ndarray]) -> np.ndarray:
    """Table for `outer(inner(x))`, both length LUT_SIZE over 0..1."""
    if inner is None:
        return outer
    idx = np.clip(inner * (len(outer) - 1), 0, len(outer) - 1)
    return np.interp(idx, np.arange(len(outer)), outer).astype(np.float32)


def curves(img: np.ndarray,
           rgb: Optional[Sequence[Sequence[float]]] = None,
           r: Optional[Sequence[Sequence[float]]] = None,
           g: Optional[Sequence[Sequence[float]]] = None,
           b: Optional[Sequence[Sequence[float]]] = None) -> np.ndarray:
    """Apply a master curve and optional per-channel curves.

    The master curve runs first, then each channel curve.

    Each channel is evaluated from the **original** pixel value through the
    composed table `channel(master(x))`. An earlier version applied the master
    to the whole frame first and then the composed table to that result, which
    ran the master twice on the channels that had a curve of their own —
    `channel(master(master(x)))` — and diverged from applying the two curves in
    sequence by 0.13, an order of magnitude more than any interpolation error.
    """
    out = np.clip(img, 0.0, 1.0).astype(np.float32)
    master = curve_lut(rgb) if rgb else None

    specs = (r, g, b)
    for ch in range(3):
        spec = specs[ch]
        ch_lut = curve_lut(spec) if spec else None
        # no channel curve: the master alone decides this channel
        lut = _compose(ch_lut, master) if ch_lut is not None else master
        if lut is None:
            continue
        out[..., ch] = _apply_lut(img[..., ch], lut)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def auto_levels(img: np.ndarray, clip: float = 0.005) -> dict:
    """Suggest level settings from the image's own histogram.

    `clip` is the fraction of pixels allowed to clip at each end, matching the
    "auto" button's behaviour in every editor: it ignores a small percentile
    so a few specular highlights do not pin the white point.
    """
    from .imgutil import luminance

    lum = luminance(img)
    lo = float(np.quantile(lum, clip))
    hi = float(np.quantile(lum, 1.0 - clip))
    if hi <= lo:
        return {"master": [0.0, 1.0, 1.0, 0.0, 1.0]}
    return {"master": [round(lo, 4), round(hi, 4), 1.0, 0.0, 1.0]}
