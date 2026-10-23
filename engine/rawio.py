# -*- coding: utf-8 -*-
"""RAW ingestion: decode camera files into the retouching pipeline.

A RAW file is not an image; it is a single-channel sensor readout plus
metadata that says how to turn that readout into a colour image. Decoding is
therefore a pipeline of its own, and the controls a retoucher expects map onto
LibRaw's postprocessing stages:

  raw CFA  ->  black/white scaling  ->  white balance  ->  demosaic
           ->  colour space / gamma ->  16- or 8-bit RGB

Only `rawpy` (a LibRaw wrapper) is used, which is why this module can support
the formats LibRaw supports — CR2/CR3/NEF/ARW/RAF/DNG/ORF/RW2 — without
implementing any format by hand.

Half-size decode exists because a 24 Mpx frame takes ~0.8 s to demosaic while
a retoucher tuning sliders wants a preview in well under 200 ms. LibRaw's
`half_size` option decodes every second pixel, which is exactly a 2x downscale
for free, and the embedded JPEG thumbnail is faster still when quality does
not matter.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Tuple

import numpy as np

RAW_EXTENSIONS = (".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng",
                  ".orf", ".rw2", ".srw", ".pef", ".nrw")


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def _normalize_exposure(img: np.ndarray,
                        target_p95: float = 0.72,
                        max_gain: float = 4096.0) -> np.ndarray:
    """Place a linearly decoded RAW frame where a retoucher expects it.

    With `no_auto_bright` the decode stays linear, which is what a RAW editor
    needs — but linear means an interior shot sits near mean 0.09 and looks
    black. Every RAW application therefore normalises exposure on open.

    The anchor is the 95th percentile rather than the median. Real frames are
    not uniformly lit: a dance-floor shot measured a linear median of 0.0018
    against a p99 of 0.30 — the *median* is floor, not subject, so anchoring
    on it demands a 165x gain and destroys the highlights. The bright tail is
    what the viewer reads as "the lit part", and it should sit just under
    diffuse white.

    The gain is clamped, and the result passes through a filmic shoulder: a
    linear gain on a high-contrast frame pushes highlights past 1.0, and hard
    clipping them reads as blown (measured 14% blowout on a frame with 0.08%
    before). Compressing the top two stops keeps highlight structure.
    """
    from .imgutil import srgb_to_linear, linear_to_srgb

    lin = srgb_to_linear(np.clip(img, 0.0, 1.0))
    p95 = float(np.percentile(lin, 95))
    if p95 < 1e-6:
        return img
    gain = float(np.clip(target_p95 / p95, 1.0, max_gain))

    lin = lin * gain
    # filmic shoulder: compress the top of the range instead of clipping it
    shoulder = 0.92
    k = 0.10
    hi = lin > shoulder
    if hi.any():
        over = lin[hi] - shoulder
        lin[hi] = shoulder + over / (1.0 + over / k)
    return np.clip(linear_to_srgb(lin), 0.0, 1.0).astype(np.float32)


def decode(path: str,
           half_size: bool = False,
           demosaic: str = "AHD",
           wb: str = "camera",
           gamma: Tuple[float, float] = (2.222, 4.5),
           brightness: float = 1.0,
           sixteen_bit: bool = False,
           no_auto_bright: bool = True,
           auto_exposure: bool = True,
           target_p95: float = 0.72) -> np.ndarray:
    """Decode a RAW file to float32 RGB in [0, 1].

    Parameters
    ----------
    half_size : decode every second pixel. A free 2x downscale; use it for
        previews, not for export.
    demosaic : "LINEAR" (fastest), "AHD" (good default), "DCB" (cleanest
        edges), "LMMSE"/"AMAZE" (slow, high quality), or "DHT"/"AAHD".
    wb : "camera" uses the shot's white balance, "daylight" the camera's
        daylight preset, or a list of four multipliers (R, G, B, G2) for a
        manual value.
    gamma : (gamma, slope); (1, 1) gives a linear dump, the default is sRGB.
    brightness : multiplier applied after the standard LibRaw curve.
    no_auto_bright : LibRaw's auto-bright is convenient but silently changes
        exposure, so a retouching pipeline keeps it off by default.
    auto_exposure : place the frame so its 95th-percentile luminance sits at
        `target_p95`, with a filmic shoulder protecting highlights. Leave off
        for a linear dump.
    target_p95 : the bright-tail anchor `auto_exposure` aims for.

    Returns a float32 RGB image in [0, 1] (16-bit input is scaled by 1/65535).
    """
    import rawpy

    algos = {"LINEAR": rawpy.DemosaicAlgorithm.LINEAR,
             "VNG": rawpy.DemosaicAlgorithm.VNG,
             "PPG": rawpy.DemosaicAlgorithm.PPG,
             "AHD": rawpy.DemosaicAlgorithm.AHD,
             "DCB": rawpy.DemosaicAlgorithm.DCB,
             "MOD": rawpy.DemosaicAlgorithm.MODIFIED_AHD,
             "LMMSE": rawpy.DemosaicAlgorithm.LMMSE,
             "AMAZE": rawpy.DemosaicAlgorithm.AMAZE,
             "DHT": rawpy.DemosaicAlgorithm.DHT,
             "AAHD": rawpy.DemosaicAlgorithm.AAHD}
    algo = algos.get(str(demosaic).upper())
    if algo is None:
        raise ValueError("unknown demosaic algorithm: %r (have %s)"
                         % (demosaic, ", ".join(algos)))

    with rawpy.imread(path) as raw:
        # rawpy.Params accepts `gamma` and `user_wb` at construction; the same
        # fields are spelled `gamm` and `user_mul` only after construction,
        # which is why passing those spellings as kwargs fails. `fbdd` gives
        # sensor-level noise reduction before demosaicing, and `exp_shift`
        # recovers highlights without moving the midtone — both are things a
        # RAW editor exposes that a JPEG pipeline cannot offer.
        params = rawpy.Params(
            demosaic_algorithm=algo,
            use_camera_wb=(wb == "camera"),
            use_auto_wb=(wb == "auto"),
            gamma=tuple(gamma),
            bright=float(brightness),
            no_auto_bright=bool(no_auto_bright),
            half_size=bool(half_size),
            output_bps=16 if sixteen_bit else 8)

        if isinstance(wb, (list, tuple)):
            params.user_wb = [float(v) for v in wb[:4]]
            params.use_camera_wb = False

        try:
            rgb = raw.postprocess(params=params)
        except rawpy.LibRawError:
            # some files need a specific demosaicer; fall back rather than fail
            params.demosaic_algorithm = rawpy.DemosaicAlgorithm.LINEAR
            rgb = raw.postprocess(params=params)

    out = rgb.astype(np.float32)
    out /= 65535.0 if sixteen_bit else 255.0
    out = np.clip(out, 0.0, 1.0)
    if auto_exposure:
        out = _normalize_exposure(out, target_p95=target_p95)
    return out


def thumbnail(path: str) -> Optional[np.ndarray]:
    """The camera-generated preview, decoded from the embedded JPEG.

    Order of magnitude faster than a full decode: a 4000x2672 NEF decodes in
    ~0.8 s, while its embedded thumbnail is a JPEG the camera already made.
    """
    import rawpy

    try:
        with rawpy.imread(path) as raw:
            thumb = raw.extract_thumb()
    except Exception:
        return None
    if thumb.format == rawpy.ThumbFormat.JPEG:
        import cv2
        data = np.frombuffer(bytes(thumb.data), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return thumb.data.astype(np.float32) / 255.0


def metadata(path: str) -> dict:
    """The facts a RAW editor shows before touching anything."""
    import rawpy

    with rawpy.imread(path) as raw:
        s = raw.sizes
        wb = raw.camera_whitebalance
        return {
            "file": os.path.basename(path),
            "bytes": os.path.getsize(path),
            "raw_type": str(raw.raw_type),
            "cfa_shape": [int(v) for v in raw.raw_image.shape],
            "width": int(s.width),
            "height": int(s.height),
            "color_desc": raw.color_desc.decode("latin1") if raw.color_desc else "",
            "camera_wb": ([float(v) for v in wb[:4]] if wb is not None else None),
            "daylight_wb": ([float(v) for v in raw.daylight_whitebalance[:4]]
                            if raw.daylight_whitebalance is not None else None),
        }


def timed_decode(path: str, **kw) -> Tuple[np.ndarray, float]:
    """Decode and report how long it took, for the preview/export split."""
    t0 = time.perf_counter()
    out = decode(path, **kw)
    return out, (time.perf_counter() - t0) * 1000.0
