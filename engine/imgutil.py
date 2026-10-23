"""Core image utilities: colour-space conversion, filtering, masks, blending.

All engine modules work on float32 RGB arrays in [0, 1] with shape (H, W, 3),
except where a function documents otherwise.
"""
from __future__ import annotations

import numpy as np
import cv2


# --------------------------------------------------------------------------
# colour space
# --------------------------------------------------------------------------

def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """Decode sRGB-encoded values to scene-linear."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """Encode scene-linear values back to sRGB."""
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.709 relative luminance, returns (H, W) float32."""
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def hsv_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


# OpenCV's float32 Lab conversion uses L in [0, 100] and a/b in [-127, 127],
# unlike the 8-bit convention. These helpers normalise to L in [0, 1] and
# a/b in [-1, 1] so the rest of the engine can assume one scale everywhere.

def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2LAB)
    lab[..., 0] /= 100.0
    lab[..., 1] /= 127.0
    lab[..., 2] /= 127.0
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    out = lab.copy()
    out[..., 0] *= 100.0
    out[..., 1] *= 127.0
    out[..., 2] *= 127.0
    return cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB)


# --------------------------------------------------------------------------
# filtering
# --------------------------------------------------------------------------

def gaussian(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that preserves the input's dimensionality.

    OpenCV squeezes a trailing single-channel axis, so it is restored
    explicitly: callers rely on broadcasting against the original shape.
    """
    if sigma <= 0:
        return img.copy()
    a = np.ascontiguousarray(img.astype(np.float32))
    squeezed = a.ndim == 3 and a.shape[2] == 1
    if squeezed:
        a = a[..., 0]
    k = int(max(3, round(sigma * 6) | 1))
    out = cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REFLECT)
    if squeezed:
        out = out[..., None]
    return out


def gaussian2d(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur for a single-channel 2-D array."""
    if sigma <= 0:
        return img.copy()
    a = np.ascontiguousarray(img.astype(np.float32))
    k = int(max(3, round(sigma * 6) | 1))
    return cv2.GaussianBlur(a, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


def box(img: np.ndarray, radius: int) -> np.ndarray:
    """Box filter that preserves the input's dimensionality.

    OpenCV squeezes a trailing single-channel axis, so restore it explicitly:
    downstream broadcasting depends on that axis being present.
    """
    r = max(1, int(radius))
    a = np.ascontiguousarray(img.astype(np.float32))
    squeezed = a.ndim == 3 and a.shape[2] == 1
    if squeezed:
        a = a[..., 0]
    out = cv2.boxFilter(a, -1, (r, r), borderType=cv2.BORDER_REFLECT)
    if squeezed:
        out = out[..., None]
    return out


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Edge-preserving smoothing (He et al. 2010).

    `guide` is a single-channel image, `src` may have any channel count.
    """
    g = np.ascontiguousarray(guide.astype(np.float32))
    p = np.ascontiguousarray(src.astype(np.float32))
    if p.ndim == 2:
        p = p[..., None]
    g3 = g[..., None] if p.ndim == 3 else g

    mean_g = box(g3, radius)
    mean_p = box(p, radius)
    corr_gp = box(g3 * p, radius)
    corr_gg = box(g3 * g3, radius)

    cov_gp = corr_gp - mean_g * mean_p
    var_g = corr_gg - mean_g * mean_g

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g

    mean_a = box(a, radius)
    mean_b = box(b, radius)
    out = mean_a * g3 + mean_b
    return out.astype(np.float32)


def unsharp(img: np.ndarray, radius: float, amount: float, threshold: float = 0.0) -> np.ndarray:
    """Add back a scaled high-frequency band, preserving input dimensionality."""
    if amount == 0.0:
        return img.copy()
    a = np.ascontiguousarray(img.astype(np.float32))
    squeezed = a.ndim == 3 and a.shape[2] == 1
    if squeezed:
        a = a[..., 0]
    blurred = gaussian2d(a, radius) if a.ndim == 2 else gaussian(a, radius)
    detail = a - blurred
    if threshold > 0:
        # only amplify detail above the threshold (noise gate)
        mag = np.abs(detail) if detail.ndim == 2 else np.abs(detail).max(axis=2, keepdims=True)
        if detail.ndim == 2:
            gate = np.clip((mag - threshold) / max(threshold, 1e-6), 0.0, 1.0)
        else:
            gate = np.clip((mag - threshold) / max(threshold, 1e-6), 0.0, 1.0)
        detail = detail * gate
    out = a + detail * amount
    if squeezed:
        out = out[..., None]
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------

# Coordinate grids are rebuilt constantly by the mask builders, and np.mgrid
# produces int64 arrays that then have to be cast to float32. On a 1600x1066
# preview this dominated face-region construction (measured: build_regions
# took 2.0 s of a 2.4 s detection stage, almost all of it grid allocation).
# Cache one float32 grid per shape instead.
_GRID_CACHE: dict = {}
_GRID_CACHE_LIMIT = 6


def meshgrid_f32(h: int, w: int):
    """Cached float32 (h, w) coordinate grids, returned as (yy, xx)."""
    key = (int(h), int(w))
    hit = _GRID_CACHE.get(key)
    if hit is not None:
        return hit
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if len(_GRID_CACHE) >= _GRID_CACHE_LIMIT:
        _GRID_CACHE.clear()
    _GRID_CACHE[key] = (yy, xx)
    return yy, xx


def feather(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur a 2-D mask to soften its edge."""
    if sigma <= 0:
        return mask
    m = np.ascontiguousarray(mask.astype(np.float32))
    k = int(max(3, round(sigma * 6) | 1))
    return cv2.GaussianBlur(m, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Hermite smoothstep; supports descending edges (edge1 < edge0)."""
    denom = float(edge1) - float(edge0)
    if abs(denom) < 1e-12:
        return (x >= float(edge1)).astype(np.float32)
    t = np.clip((x - float(edge0)) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def radial_falloff(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Elliptical falloff: 1 at the centre, 0 at and beyond the ellipse boundary."""
    yy, xx = meshgrid_f32(h, w)
    d = np.sqrt(((xx - cx) / max(rx, 1e-3)) ** 2 + ((yy - cy) / max(ry, 1e-3)) ** 2)
    return (1.0 - smoothstep(0.0, 1.0, d)).astype(np.float32)


# --------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------

def blend(base: np.ndarray, top: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-pixel linear blend; `mask` broadcasts over channels."""
    m = mask[..., None] if mask.ndim == 2 else mask
    return (base * (1.0 - m) + top * m).astype(np.float32)


def blend_amount(base: np.ndarray, top: np.ndarray, amount: float) -> np.ndarray:
    """Uniform blend by a scalar in [0, 1]; amount may exceed 1 to extrapolate."""
    return (base + (top - base) * amount).astype(np.float32)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1], honouring EXIF orientation."""
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0)


def save_image(path: str, img: np.ndarray, quality: int = 95) -> None:
    """Write an RGB float image to disk, choosing the codec from the extension."""
    ext = path.rsplit(".", 1)[-1].lower()
    out = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in ("jpg", "jpeg") else []
    ok, buf = cv2.imencode("." + ext, bgr, params)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buf.tofile(path)


def encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
    out = (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("cannot encode jpeg")
    return buf.tobytes()


def fit_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    """Downscale so the longer side is at most `long_edge`."""
    h, w = img.shape[:2]
    scale = long_edge / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
