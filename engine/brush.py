# -*- coding: utf-8 -*-
"""Clone stamp and healing brush.

Two tools that look alike and behave very differently, which is why they are
implemented together — the difference is one term in the same equation.

Both sample a source patch and blend it into the destination, so with the
source offset fixed:

    clone(x) = P_src(x)

A clone is exact and that is the point: it is the tool for duplicating a
texture or extending a pattern. It is also why a clone is visible when the
source and destination sit under different lighting — the pasted patch carries
the source's brightness and the seam is a step in a smooth gradient. Every
retoucher has seen this and works around it by cloning from nearby.

    heal(x) = P_src(x) - blur(P_src)(x) + blur(P_dst)(x)

A heal keeps the source's **high frequency** (texture, pores, grain) and takes
the destination's **low frequency** (colour and illumination). That single
substitution removes the seam for the common case of retouching across a
gradient, which is why healing is the default for skin work and cloning is
reserved for texture.

The low-pass radius controls the split: too small and the patch keeps the
source's shading (seam returns), too large and the patch loses its own
modelling (looks flat). It is exposed as a fraction of the brush so one
setting survives a size change.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import cv2

from .imgutil import gaussian


def _brush_mask(size: int, hardness: float) -> np.ndarray:
    """Radial falloff for one dab.

    `hardness` 0 is a fully soft brush and 1 is a hard edge with only a
    single-pixel antialias ring. The soft default matters: a hard-edged dab
    leaves a visible disc, so nearly every real stroke is soft.
    """
    r = max(1, int(size))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy) / float(r)
    h = float(np.clip(hardness, 0.0, 1.0))
    # inside `inner` the brush is solid, then it ramps to zero at the rim;
    # np.where rather than an if, since `d` is an array
    inner = h * 0.92
    t = np.clip((d - inner) / max(1e-4, 1.0 - inner), 0.0, 1.0)
    ramp = 1.0 - (t * t * (3.0 - 2.0 * t))        # smoothstep, no banding
    m = np.where(d <= inner, 1.0, ramp)
    return np.clip(m, 0.0, 1.0).astype(np.float32)


def _patch(img: np.ndarray, cx: float, cy: float, r: int) -> Tuple[np.ndarray, int, int]:
    """Extract a (2r+1)² patch, zero-padding at the frame edge.

    Reflect padding would mirror content across the border, which for a clone
    duplicates the edge pixel outward and looks wrong; zeros at least keep the
    brush mask (which already fades to 0) doing the work.
    """
    h, w = img.shape[:2]
    x0, y0 = int(round(cx)) - r, int(round(cy)) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    out = np.zeros((2 * r + 1, 2 * r + 1, img.shape[2]), np.float32)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out, x0, y0


def dab(img: np.ndarray, cx: float, cy: float, sx: float, sy: float,
        radius: int, hardness: float = 0.65, opacity: float = 1.0,
        mode: str = "heal", low_freq: float = 0.5) -> np.ndarray:
    """Apply one dab; returns a new array (never mutates the input).

    `radius` is in pixels, `opacity` 0..1 scales the whole dab, and
    `low_freq` (only used by `heal`) is the split radius as a fraction of the
    brush radius.
    """
    r = max(1, int(radius))
    k = float(np.clip(opacity, 0.0, 1.0))
    if k <= 0:
        return img

    src, _sx0, _sy0 = _patch(img, sx, sy, r)
    m = _brush_mask(r, hardness)
    if k < 1.0:
        m = m * k
    m3 = m[..., None]

    if mode == "clone":
        sample = src
    else:
        # keep the source's texture, take the destination's colour/lighting
        sigma = max(1.0, r * float(np.clip(low_freq, 0.05, 1.5)))
        dst, _x0, _y0 = _patch(img, cx, cy, r)
        sample = src - gaussian(src, sigma) + gaussian(dst, sigma)

    out = img.copy()
    h, w = out.shape[:2]
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    x1, y1 = x0 + 2 * r + 1, y0 + 2 * r + 1
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return out
    px, py = sx0 - x0, sy0 - y0
    region = out[sy0:sy1, sx0:sx1]
    piece = sample[py:py + (sy1 - sy0), px:px + (sx1 - sx0)]
    mm = m3[py:py + (sy1 - sy0), px:px + (sx1 - sx0)]
    # Clip only what is actually written. Clipping the whole frame is tempting
    # but wrong: a decoded image can hold values a hair above 1.0 (measured
    # 1.0000001 at the top edge of a JPEG), and a full-frame clip would move
    # those pixels by one ULP — 29 of them in one test stroke — breaking the
    # rule that a masked operation leaves everything outside its mask alone.
    out[sy0:sy1, sx0:sx1] = np.clip(
        region * (1.0 - mm) + piece * mm, 0.0, 1.0)
    return out.astype(np.float32)


def _resample_path(points: Sequence[Sequence[float]],
                   spacing: float) -> List[Tuple[float, float]]:
    """Fill a polyline with evenly spaced dabs.

    Mouse events arrive far apart when the pointer moves quickly, so painting
    only at event positions leaves a dotted stroke. Interpolating at a fraction
    of the brush radius produces a continuous line at any speed.
    """
    pts = [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
    if not pts:
        return []
    if len(pts) == 1:
        return [pts[0]]
    step = max(0.5, float(spacing))
    out = [pts[0]]
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        seg = float(np.hypot(x1 - x0, y1 - y0))
        n = int(seg / step)
        for j in range(1, n + 1):
            t = j / float(n)
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return out


def stroke(img: np.ndarray, points: Sequence[Sequence[float]],
           source: Sequence[float], radius: int = 40,
           hardness: float = 0.65, opacity: float = 1.0,
           mode: str = "heal", low_freq: float = 0.5) -> np.ndarray:
    """Paint a whole stroke by dabbing along an interpolated path.

    The source offset is held fixed for the stroke, so a dragged brush keeps
    sampling the same relative area — the behaviour that makes a clone read as
    a continuous copy rather than a smear.
    """
    pts = _resample_path(points, spacing=max(1.0, radius * 0.25))
    if not pts:
        return img
    x0, y0 = float(points[0][0]), float(points[0][1])
    off_x = float(source[0]) - x0
    off_y = float(source[1]) - y0

    # Per-dab opacity is reduced as dabs get closer, otherwise a slow drag
    # stacks the same dab dozens of times and burns through to full opacity.
    spacing = max(1.0, radius * 0.25)
    per = float(np.clip(opacity, 0.0, 1.0))
    if mode == "heal":
        per = per * min(1.0, spacing / max(1.0, radius * 0.5))

    out = img
    for (px, py) in pts:
        out = dab(out, px, py, px + off_x, py + off_y, radius,
                  hardness=hardness, opacity=per, mode=mode,
                  low_freq=low_freq)
    return out


def stroke_inplace(img: np.ndarray, points: Sequence[Sequence[float]],
                   source: Sequence[float], **kw) -> np.ndarray:
    """Same as `stroke` but writes into `img` — used by the live preview."""
    result = stroke(img, points, source, **kw)
    img[...] = result
    return img
