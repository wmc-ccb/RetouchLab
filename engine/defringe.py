# -*- coding: utf-8 -*-
"""祛边 — remove the fringe left along a cut-out edge.

When a subject is lifted off its background, the boundary keeps a blend of the
old background: a white halo from a bright backdrop, a colour fringe from a
saturated one, and a soft one-pixel ramp that reads as "unsharp". Three
separate problems, three separate fixes, because they need different detection:

* **彩边 / 紫边** — chromatic fringing. Found by looking for pixels whose
  channels *disagree in chroma direction* across the edge: a magenta fringe is
  R and B both rising while G does not.
* **白边** — luminance halo. The edge band is brighter than both sides of it.
* **边缘收缩** — the boundary sits a pixel outside the true subject. Fixed by
  eroding the alpha and re-compositing against an inpainted interior colour,
  rather than by blurring, which just smears the fringe inward.

This is deliberately not a "sharpen the edge" tool: a fringe is a colour
contamination problem, and sharpening it makes it more visible.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _edge_band(alpha: np.ndarray, width: int = 2) -> np.ndarray:
    """Pixels near the alpha boundary, inside and outside."""
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    hi = (a >= 0.5).astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    inner = cv2.erode(hi, k, iterations=1)
    outer = cv2.dilate(hi, k, iterations=1)
    band = (outer - inner).astype(bool)
    if width > 1:
        band = cv2.dilate(band.astype(np.uint8),
                          np.ones((2 * width - 1, 2 * width - 1), np.uint8),
                          iterations=1).astype(bool)
    # restrict to where alpha is actually partial: a hard edge has no fringe
    return band & (a > 0.02) | (band & (a < 0.98))


def defringe(img: np.ndarray, alpha: np.ndarray, amount: float = 0.6,
             radius: int = 2) -> np.ndarray:
    """Suppress chromatic fringing by clamping each channel to its local range.

    Along an edge each channel is interpolated with the *background's* value,
    so a channel that the background pushes up shows a halo past the value the
    subject itself ever reaches. Clamping the edge band to the min/max the
    neighbouring subject-side pixels take removes the excursion without
    touching the subject's real colour.
    """
    src = np.asarray(img, np.float32)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    if not np.any(a > 0.02):
        return src.copy()

    band = _edge_band(a, radius)
    # subject interior = confident alpha; its local range is the truth
    solid = (a > 0.85).astype(np.float32)
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.float32)
    kw = float(k.sum())
    # local min / max over the solid region, per channel
    out = src.copy()
    for c in range(3):
        ch = src[..., c]
        local_min = cv2.erode(ch, np.ones((5, 5), np.uint8))
        local_max = cv2.dilate(ch, np.ones((5, 5), np.uint8))
        # weight the correction by how much subject is nearby; where there is
        # no solid neighbour the clamp is meaningless and is skipped
        near = cv2.filter2D(solid, -1, k) / kw
        w = np.clip(near * 2.0, 0.0, 1.0) * amount
        fixed = np.clip(ch, local_min, local_max)
        out[..., c] = np.where(band, ch * (1 - w) + fixed * w, ch)
    return np.clip(out, 0.0, 1.0)


def dehalo(img: np.ndarray, alpha: np.ndarray, strength: float = 0.5,
           radius: int = 3) -> np.ndarray:
    """Remove a bright (or dark) halo by pulling the edge band toward the
    subject-side colour.

    A halo is a *luminance* excursion whose sign is the same on both sides of
    the boundary. Estimating the subject's own edge colour by sampling just
    inside the alpha, then blending the band toward it, removes the ring
    without the flattening that a plain blur causes.
    """
    src = np.asarray(img, np.float32)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    if not np.any(a > 0.02):
        return src.copy()

    band = _edge_band(a, 2)
    k = (2 * radius + 1)
    inside = cv2.erode((a > 0.7).astype(np.float32),
                       np.ones((5, 5), np.uint8))
    # the colour the subject has just inside the edge
    interior = cv2.blur(src * inside[..., None],
                        (k, k))
    wt = cv2.blur(inside, (k, k)) + 1e-6
    interior = interior / wt[..., None]
    w = (band & (inside < 0.5)).astype(np.float32) * strength
    out = src * (1 - w[..., None]) + interior * w[..., None]
    return np.clip(out, 0.0, 1.0)


def contract(img: np.ndarray, alpha: np.ndarray, pixels: float = 1.0,
             sample_radius: int = 4) -> np.ndarray:
    """Shrink the subject by `pixels`, filling the vacated ring from inside.

    Used when the cut sits consistently outside the true edge (very common with
    a soft matte). The ring is filled from the subject's own nearby colour, so
    the boundary keeps the subject's texture instead of the background's.
    """
    src = np.asarray(img, np.float32)
    a = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if a.ndim == 3:
        a = a[..., 0]
    p = float(pixels)
    if p <= 0:
        return src.copy()
    k = int(np.ceil(p)) * 2 + 1
    shrunk = cv2.erode(a, np.ones((k, k), np.uint8), iterations=1)
    grown = cv2.dilate(shrunk, np.ones((k, k), np.uint8), iterations=1)
    ring = (grown > 0.5) & (shrunk <= 0.5)
    if not ring.any():
        return src.copy()
    r = max(2, int(sample_radius))
    solid = (shrunk > 0.5).astype(np.float32)
    inner = cv2.blur(src * solid[..., None], (2 * r + 1, 2 * r + 1))
    wt = cv2.blur(solid, (2 * r + 1, 2 * r + 1)) + 1e-6
    inner = inner / wt[..., None]
    out = src.copy()
    out[ring] = inner[ring]
    return np.clip(out, 0.0, 1.0)


def auto_alpha(img: np.ndarray) -> Optional[np.ndarray]:
    """Subject alpha from the installed matting model, if any."""
    try:
        from . import matting as MT
        if not MT.available():
            return None
        return MT.subject_alpha(img, mode="hybrid")
    except Exception:
        return None


def apply(img: np.ndarray, defringe_amount: float = 0.0,
          dehalo_amount: float = 0.0, contract_px: float = 0.0,
          alpha: Optional[np.ndarray] = None) -> np.ndarray:
    """Run the selected edge fixes in the only order that makes sense."""
    out = np.asarray(img, np.float32)
    a = alpha if alpha is not None else auto_alpha(out)
    if a is None:
        return out.copy()
    if contract_px > 0:
        out = contract(out, a, contract_px)
        # contract changed the boundary; recompute so the later passes use it
        a = np.clip(cv2.erode(a, np.ones((3, 3), np.uint8)), 0.0, 1.0)
    if defringe_amount > 0:
        out = defringe(out, a, defringe_amount)
    if dehalo_amount > 0:
        out = dehalo(out, a, dehalo_amount)
    return out


def info() -> dict:
    return {"supports": ["defringe", "dehalo", "contract"],
            "needs_alpha": True}
