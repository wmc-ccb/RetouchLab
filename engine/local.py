"""Local adjustments: masked edits to a region of the frame.

The reference application exposes gradient and radial local tools. Their
semantics are the same across every retoucher: build a soft mask, then apply
a *subset* of the global controls — exposure, contrast, saturation,
clarity, temperature, tint, dehaze — inside it. What makes the tool useful is
that the mask feathers smoothly and the same control can be stacked several
times over different regions.

Mask construction
-----------------
* **radial** — an elliptical falloff; `invert` makes it affect everything
  *outside* the ellipse, which is the usual "vignette the subject" shape.
* **linear** — a ramp along an angle, from full effect on one side to none on
  the other, with a soft band between.

Each entry carries its own parameters, so one image can have a brightened
face, a darkened corner and a warmed sky without the controls interfering.

Only masks are geometry; the parameter application reuses `engine.tone`, so
local results stay consistent with the global ones.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import cv2

from .face import _plateau
from .imgutil import smoothstep, meshgrid_f32


def radial_mask(shape: tuple, cx: float, cy: float, rx: float, ry: float,
                feather: float = 0.5, invert: bool = False,
                angle: float = 0.0) -> np.ndarray:
    """Elliptical falloff, 1 at the centre and 0 at the boundary.

    `feather` (0..1) is the fraction of the radius spent on the transition;
    0 is a hard edge, 1 falls off across the whole ellipse.
    `angle` rotates the ellipse in degrees.
    """
    h, w = shape[:2]
    yy, xx = meshgrid_f32(h, w)
    dx = xx - float(cx)
    dy = yy - float(cy)
    if abs(angle) > 1e-6:
        t = np.deg2rad(float(angle))
        c, s = np.cos(t), np.sin(t)
        dx, dy = dx * c + dy * s, -dx * s + dy * c
    rx = max(float(rx), 1.0)
    ry = max(float(ry), 1.0)
    d = np.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)

    f = float(np.clip(feather, 0.0, 1.0))
    # inside 1-f of the radius the weight is 1, then it falls to 0 at 1.0
    inner = max(1e-3, 1.0 - f)
    m = 1.0 - smoothstep(inner, 1.0, d)
    if invert:
        m = 1.0 - m
    return m.astype(np.float32)


def linear_mask(shape: tuple, angle: float, position: float,
                feather: float = 0.4, invert: bool = False) -> np.ndarray:
    """Ramp along `angle` (degrees, 0 = left→right, 90 = top→bottom).

    `position` (0..1) is where the ramp's centre sits along the gradient
    direction, measured on the *projected* extent of the frame, so 0.5 is the
    middle regardless of orientation. `feather` is the band width as a
    fraction of the projected extent.
    """
    h, w = shape[:2]
    yy, xx = meshgrid_f32(h, w)
    t = np.deg2rad(float(angle))
    proj = xx * np.cos(t) + yy * np.sin(t)

    # project the frame's corners to get the real extent along the axis
    corners = np.array([
        0 * np.cos(t) + 0 * np.sin(t),
        w * np.cos(t) + 0 * np.sin(t),
        0 * np.cos(t) + h * np.sin(t),
        w * np.cos(t) + h * np.sin(t),
    ], np.float32)
    lo, hi = float(corners.min()), float(corners.max())
    extent = max(hi - lo, 1.0)

    centre = lo + float(np.clip(position, 0.0, 1.0)) * extent
    band = max(extent * float(np.clip(feather, 0.01, 1.0)) * 0.5, 1.0)
    m = 1.0 - smoothstep(centre - band, centre + band, proj)
    if invert:
        m = 1.0 - m
    return m.astype(np.float32)


# The subset of global controls a local entry may carry. Reusing the tone
# module keeps local and global results consistent; adding a control here is
# the only change needed to expose it locally.
_LOCAL_CONTROLS = (
    "exposure", "contrast", "saturation", "clarity", "temp", "tint", "dehaze",
)


def _coord(v: float, extent: int, name: str = "coordinate") -> float:
    """Accept either a 0..1 fraction or an absolute pixel value.

    A JSON recipe reads better in fractions ({"cx": 0.5}), while the function
    API grew up on pixels; treating values <= 1 as fractions makes both work.
    The one ambiguous case, a genuine pixel value of 1, loses nothing: it is
    indistinguishable from "the left edge" either way.
    """
    v = float(v)
    if 0.0 <= v <= 1.0:
        return v * extent
    return v


def apply_local(img: np.ndarray, entries: List[dict],
                strength_scale: float = 1.0) -> np.ndarray:
    """Apply every local adjustment entry in order.

    Each entry is a dict::

        {"type": "radial"|"linear", ...geometry,
         "strength": 0..100, "invert": bool, "feather": 0..1,
         "exposure": -100..100, "contrast": ..., "saturation": ...,
         "clarity": ..., "temp": ..., "tint": ..., "dehaze": ...}

    Geometry may use 0..1 fractions or pixels; fractions are the natural
    spelling in a stored recipe, pixels in live code. Entries stack: the
    second sees the output of the first, which is what a retoucher expects
    when they add a face brightener and then a corner darkener.
    """
    from . import tone as T

    # No upfront full-frame clip: the input can legitimately sit a hair outside
    # [0, 1] after a decode, and clipping it here would move pixels that no
    # mask covers. Clipping happens once, at the end, restricted to what was
    # written.
    out = img.astype(np.float32).copy()
    if not entries:
        return out
    if not any(isinstance(e, dict) for e in entries):
        return out
    # every pixel any entry actually writes, so the closing clip can be limited
    # to those and the rest of the frame stays bit-identical
    touched = np.zeros(out.shape[:2], bool)

    h, w = out.shape[:2]
    for e in entries:
        if not isinstance(e, dict):
            continue
        etype = str(e.get("type", "radial")).lower()
        if etype == "radial":
            m = radial_mask(
                (h, w),
                cx=_coord(e.get("cx", 0.5), w), cy=_coord(e.get("cy", 0.5), h),
                rx=_coord(e.get("rx", 0.25), w), ry=_coord(e.get("ry", 0.25), h),
                feather=float(e.get("feather", 0.5)),
                invert=bool(e.get("invert", False)),
                angle=float(e.get("angle", 0.0)))
        elif etype == "linear":
            m = linear_mask(
                (h, w),
                angle=float(e.get("angle", 0.0)),
                position=float(e.get("position", 0.5)),
                feather=float(e.get("feather", 0.4)),
                invert=bool(e.get("invert", False)))
        else:
            continue

        strength = float(np.clip(e.get("strength", 100.0), 0.0, 100.0)) / 100.0
        strength *= float(np.clip(strength_scale, 0.0, 1.0))
        if strength <= 0:
            continue

        # resolve the controls this entry actually carries
        kw = {}
        for k in _LOCAL_CONTROLS:
            if e.get(k):
                kw[k] = float(e[k])
        if not kw:
            continue

        adj = _apply_controls(out, kw)

        a = (m * strength)[..., None]
        blended = out * (1.0 - a) + adj * a
        # Write back only inside the mask. A full-frame assignment would clip
        # pixels far outside the mask by one ULP when the input sits a hair
        # above 1.0 (measured 2.4e-07 over 93% of the frame), which is exactly
        # the leak a mask is supposed to prevent.
        inside = (m * strength) > 0.0
        out = np.where(inside[..., None], blended, out)
        touched |= inside

    # clip only what was written: the untouched frame must come back bit-exact
    if touched.any():
        out[touched] = np.clip(out[touched], 0.0, 1.0)
    return out.astype(np.float32)


def apply_through_mask(img: np.ndarray, mask: np.ndarray, controls: dict,
                       feather: float = 1.5,
                       invert: bool = False,
                       return_mask: bool = False):
    """Apply global tone controls inside a supplied mask.

    This is what a *piecewise* editor is built on: instead of authoring mask
    geometry, the caller hands over a mask that came from somewhere else — a
    semantic class, a brush, a luminance range — and the same controls the
    global panel exposes act only inside it.

    Reusing `_apply_controls` matters: a local exposure must produce the same
    pixels as the global exposure, or a face brightened locally and a frame
    brightened globally would not match and the two controls could not be
    balanced against each other.

    The untouched region is returned bit-identical, clipped only where the mask
    actually writes. `return_mask=True` also returns the **effective** mask
    (after inversion and feathering), because that — not the input mask — is
    the region the call is allowed to touch, and a caller cannot otherwise
    assert locality exactly: the feather legitimately reaches beyond every
    pixel where the input is non-zero.
    """
    m = np.asarray(mask, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    m = np.clip(m, 0.0, 1.0)
    if invert:
        m = 1.0 - m

    def done(result, effective):
        return (result, effective) if return_mask else result

    if m.max() <= 1e-6:
        return done(img.copy(), np.zeros_like(m, bool))

    kw = {k: float(v) for k, v in (controls or {}).items()
          if k in _LOCAL_CONTROLS and v}
    if not kw:
        return done(img.copy(), np.zeros_like(m, bool))

    adj = _apply_controls(img.astype(np.float32), kw)
    a = m
    if feather and feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(feather))
    a = np.clip(a, 0.0, 1.0)

    out = img.astype(np.float32).copy()
    touched = a > 0.0
    if touched.any():
        base = img.astype(np.float32)[touched]          # (N, 3)
        top = adj[touched]                              # (N, 3)
        w = a[touched][:, None]                         # (N, 1) — 2-D mask,
        # so it has to be indexed first and only then expanded; `a[touched,
        # None]` is a tuple index on the first axis of a 2-D array and raises.
        out[touched] = np.clip(base * (1.0 - w) + top * w, 0.0, 1.0)
    return done(out.astype(np.float32), touched)


def _apply_controls(img: np.ndarray, kw: dict) -> np.ndarray:
    """Run the global tone operators named in `kw`."""
    from . import tone as T

    out = img
    if "exposure" in kw:
        out = T.exposure(out, kw["exposure"])
    if "contrast" in kw:
        out = T.contrast(out, kw["contrast"])
    if "dehaze" in kw:
        out = T.dehaze(out, kw["dehaze"])
    if "clarity" in kw:
        out = T.clarity(out, kw["clarity"])
    if "saturation" in kw:
        out = T.saturation(out, kw["saturation"])
    if "temp" in kw or "tint" in kw:
        out = T.white_balance(out, kw.get("temp", 0.0), kw.get("tint", 0.0),
                              "as_shot")
    return out
