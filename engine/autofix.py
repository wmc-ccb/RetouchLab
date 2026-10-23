# -*- coding: utf-8 -*-
"""One-click auto correction: white balance, exposure, contrast.

Three estimates that each need to be robust before they are useful, because a
single bright object or one dark corner will otherwise drag the whole frame.

White balance
-------------
Three estimators, in increasing robustness:

* **max-rgb** (white patch) — assumes the brightest pixel is white. One
  specular highlight or a white shirt and the frame goes blue.
* **grey world** — assumes channel means are equal. A frame dominated by one
  colour (grass, a red background) fails badly.
* **shades of grey** (default) — a Minkowski norm over the pixels, which
  interpolates between the two: p=1 is grey world, p=∞ is white patch. It is
  the standard fix for both failure modes above, and the estimator used here.

Estimates are computed on a **downsampled, clipped** copy: downsampling removes
sensor noise (which biases a max-based estimator upward) and clipping to the
1st..99th percentile removes speculars and deep shadows.

Exposure and contrast
---------------------
Exposure comes from the median, not the mean, so a large bright or dark area
does not move it. Contrast is a levels remap between the 1st and 99th
percentiles of **luminance**, which is the same thing the auto button in every
editor does, plus a ceiling on how far it will move: a flat frame (fog, a
white background) would otherwise be stretched into noise.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .imgutil import luminance


def _sample(img: np.ndarray, long_edge: int = 256) -> np.ndarray:
    """Downsample for estimation.

    Downsampling is not just speed: a max-based estimator over full-resolution
    noise picks up hot pixels, and averaging many pixels into one removes that.
    """
    import cv2
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return img
    s = long_edge / float(m)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))),
                      interpolation=cv2.INTER_AREA)


def white_balance_gains(img: np.ndarray, method: str = "shades",
                        power: int = 6, clip: float = 0.01) -> np.ndarray:
    """Per-channel gains that neutralise the illuminant, estimated in **linear
    light**.

    The space matters and getting it wrong is a silent 2× error. An illuminant
    is a multiplicative gain on the scene's linear radiance, so a 1.35× gain in
    linear light appears in the gamma-encoded image as roughly
    ``1.35^(1/2.2) ≈ 1.15``. Estimating in the encoded space therefore finds a
    gain that is numerically correct for that space but only removes about half
    of the cast once applied as a linear correction — measured at 52% of the
    cast removed before this was fixed, against 93% after.

    Returns gains scaled so the green channel is 1.0, which keeps the overall
    brightness unchanged — a white balance should move colour, not exposure.
    """
    from .imgutil import srgb_to_linear

    s = _sample(img)
    flat = srgb_to_linear(s).reshape(-1, 3)
    # Drop the extreme tails: speculars and near-black pixels carry almost no
    # illuminant information but dominate both max and mean estimators.
    #
    # A per-channel saturation filter was tried here and removed: selecting on
    # `max(channel)` makes the excluded set depend on the cast's direction (a
    # warm frame saturates red first), which biases the estimate itself — it
    # moved the warm case from 43% of the cast corrected to 39%. An estimator
    # must not carry a bias that correlates with what it is measuring.
    lo = np.quantile(flat, clip, axis=0)
    hi = np.quantile(flat, 1.0 - clip, axis=0)
    flat = np.clip(flat, lo, hi)
    flat = flat[flat.max(axis=1) > 0.02]          # ignore pure black
    if flat.shape[0] < 64:
        return np.ones(3, np.float32)

    if method == "maxrgb":
        est = np.percentile(flat, 99.5, axis=0)
    elif method == "grey":
        est = flat.mean(axis=0)
    else:
        # Minkowski norm per channel; p=6 sits between grey world and white patch
        est = np.power(np.mean(np.power(flat, power), axis=0), 1.0 / power)
    est = np.maximum(est, 1e-4)

    gains = float(est[1]) / est                 # normalise on green
    return np.clip(gains, 0.4, 2.5).astype(np.float32)


def apply_gains(img: np.ndarray, gains: Sequence[float]) -> np.ndarray:
    """Apply gains in linear light, matching how they were estimated."""
    from .imgutil import srgb_to_linear, linear_to_srgb

    g = np.asarray(gains, np.float32).reshape(1, 1, 3)
    lin = srgb_to_linear(img) * g
    return np.clip(linear_to_srgb(np.clip(lin, 0.0, None)), 0.0, 1.0).astype(np.float32)


def gains_as_srgb(gains: Sequence[float]) -> np.ndarray:
    """The equivalent multiplicative gain in gamma-encoded space.

    Only for reporting: the temp/tint sliders are defined as gains on the
    encoded image, so a linear gain has to be converted before it can be shown
    as a slider position (``k**(1/2.2)``).
    """
    g = np.asarray(gains, np.float32).reshape(3)
    return np.power(np.maximum(g, 1e-6), 1.0 / 2.2)


def exposure_ev(img: np.ndarray, target: float = 0.46,
                clip: float = 0.02) -> float:
    """Exposure correction in EV that brings the median luminance to `target`.

    The median is used rather than the mean so a large uniform area — a white
    backdrop, a dark stage — does not decide the exposure for the subject.
    """
    lum = luminance(_sample(img))
    flat = lum[(lum > 0.01) & (lum < 0.99)]
    if flat.size < 64:
        return 0.0
    med = float(np.median(flat))
    if med <= 1e-4:
        return 0.0
    ev = float(np.log2(target / med))
    return float(np.clip(ev, -2.0, 2.0))


def contrast_spec(img: np.ndarray, clip: float = 0.01,
                  max_stretch: float = 0.30) -> Dict[str, list]:
    """Level settings that set the black and white points from the histogram.

    `max_stretch` caps how much of the range may be removed from each end.
    Without it a low-contrast frame (fog, a white sweep) is stretched until its
    noise fills the full range, which looks worse than leaving it flat.
    """
    lum = luminance(_sample(img))
    lo = float(np.quantile(lum, clip))
    hi = float(np.quantile(lum, 1.0 - clip))
    lo = float(np.clip(lo, 0.0, max_stretch))
    hi = float(np.clip(hi, 1.0 - max_stretch, 1.0))
    if hi <= lo + 1e-3:
        return {"master": [0.0, 1.0, 1.0, 0.0, 1.0]}
    return {"master": [round(lo, 4), round(hi, 4), 1.0, 0.0, 1.0]}


def suggest(img: np.ndarray, wb: bool = True, exposure: bool = True,
            contrast: bool = True, method: str = "shades",
            strength: float = 1.0) -> dict:
    """Build a parameter dict for the pipeline.

    `strength` scales every estimate toward neutral, so a user who finds the
    automatic result too strong can dial it back without giving up the tool —
    a 0.5 strength halves the gains' distance from 1.0 and the EV.
    """
    k = float(np.clip(strength, 0.0, 1.0))
    out: dict = {}

    if wb:
        g = white_balance_gains(img, method=method)
        g = 1.0 + (g - 1.0) * k
        out["wb_gains"] = [round(float(v), 4) for v in g]
        # temp/tint are reported for the sliders, but the pipeline applies the
        # linear gains: converting a linear gain to the equivalent encoded-space
        # gain is what the display needs (see gains_as_srgb)
        from .tone import gains_to_temp_tint
        tt = gains_to_temp_tint(gains_as_srgb(g))
        if abs(tt["temp"]) > 0.5 or abs(tt["tint"]) > 0.5:
            out["_ui_temp"] = round(tt["temp"], 1)
            out["_ui_tint"] = round(tt["tint"], 1)
        out["_gains"] = [round(float(v), 4) for v in g]

    if exposure:
        ev = exposure_ev(img) * k
        if abs(ev) > 0.05:
            # the pipeline's `exposure` is a -100..100 slider spanning ±2 stops
            out["exposure"] = round(ev / 2.0 * 100.0, 2)
            out["_ev"] = round(ev, 4)

    if contrast:
        lv = contrast_spec(img)
        m = lv["master"]
        if k < 1.0:
            m = [m[0] * k, 1.0 - (1.0 - m[1]) * k, 1.0, 0.0, 1.0]
        if m[0] > 1e-3 or m[1] < 1.0 - 1e-3:
            out["levels"] = {"master": [round(v, 4) for v in m]}

    return out


def apply(img: np.ndarray, params: dict) -> np.ndarray:
    """Apply a `suggest` result directly (used for the preview)."""
    from . import curves as CV
    from . import tone as T

    out = img
    if "_gains" in params:
        out = apply_gains(out, params["_gains"])
    if params.get("exposure"):
        # `exposure` is the slider value (-100..100 over ±2 stops)
        out = T.exposure(out, params["exposure"])
    if params.get("levels"):
        out = CV.levels(out, master=params["levels"]["master"])
    return np.clip(out, 0.0, 1.0).astype(np.float32)
