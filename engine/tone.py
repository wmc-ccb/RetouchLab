"""Tone and colour engine.

Parameter names mirror the ones recovered from the analysed application's
UI definition (the "zt", "aa", "Ca", "_a", "ut" parameter groups) so the
behaviour can be compared feature by feature. The implementations here are
original and built on standard, published image-processing operations.

Convention: every function takes and returns float32 RGB in [0, 1].
Sliders use the same ranges the reference UI declares (mostly -100..100,
0..100 for magnitude-only controls).
"""
from __future__ import annotations

import numpy as np
import cv2

from .imgutil import (
    srgb_to_linear, linear_to_srgb, luminance, gaussian, gaussian2d, unsharp,
    rgb_to_hsv, hsv_to_rgb, blend, smoothstep,
)


# --------------------------------------------------------------------------
# basic tone  (group "zt": exposure, contrast, brightness, highlights, ...)
# --------------------------------------------------------------------------

def exposure(img: np.ndarray, amount: float) -> np.ndarray:
    """Exposure compensation in stops, scaled from a -100..100 slider."""
    if amount == 0:
        return img.copy()
    stops = amount / 100.0 * 2.0          # +/- 2 stops at the extremes
    lin = srgb_to_linear(img)
    lin = lin * (2.0 ** stops)
    return linear_to_srgb(lin)


def brightness(img: np.ndarray, amount: float) -> np.ndarray:
    """Simple additive lift applied in linear light."""
    if amount == 0:
        return img.copy()
    lin = srgb_to_linear(img)
    lin = lin + (amount / 100.0) * 0.25
    return linear_to_srgb(np.clip(lin, 0.0, None))


def contrast(img: np.ndarray, amount: float) -> np.ndarray:
    """S-curve contrast around mid-grey, pivoted in linear light."""
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    lin = srgb_to_linear(img)
    pivot = 0.1845                         # linear value of 18% grey
    # smooth, monotonic curve; k>0 steepens, k<0 flattens
    f = 1.0 + k
    out = np.power(np.clip(lin / pivot, 1e-6, None), f) * pivot
    return linear_to_srgb(out)


def _tone_band(lum: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Weight for a luminance band with smooth shoulders."""
    up = smoothstep(lo, hi, lum)
    down = 1.0 - smoothstep(hi, min(1.0, hi + (hi - lo)), lum) if hi < 1.0 else np.ones_like(lum)
    return (up * down).astype(np.float32)


def highlights(img: np.ndarray, amount: float) -> np.ndarray:
    """Recover or lift the upper tonal range."""
    if amount == 0:
        return img.copy()
    lum = luminance(img)
    w = smoothstep(0.5, 1.0, lum)[..., None]
    delta = (amount / 100.0) * 0.35
    lin = srgb_to_linear(img)
    out = np.where(delta >= 0, lin + delta * w, lin * (1.0 + delta * w))
    return linear_to_srgb(np.clip(out, 0.0, None))


def shadows(img: np.ndarray, amount: float) -> np.ndarray:
    """Lift or deepen the lower tonal range."""
    if amount == 0:
        return img.copy()
    lum = luminance(img)
    w = 1.0 - smoothstep(0.0, 0.5, lum)
    w = (w * w)[..., None]
    delta = (amount / 100.0) * 0.35
    lin = srgb_to_linear(img)
    out = np.where(delta >= 0, lin + delta * w, lin * (1.0 + delta * w))
    return linear_to_srgb(np.clip(out, 0.0, None))


def whites(img: np.ndarray, amount: float) -> np.ndarray:
    """Move the white point."""
    if amount == 0:
        return img.copy()
    lum = luminance(img)
    w = smoothstep(0.7, 1.0, lum)[..., None]
    lin = srgb_to_linear(img)
    out = lin * (1.0 + (amount / 100.0) * 0.3 * w)
    return linear_to_srgb(np.clip(out, 0.0, None))


def blacks(img: np.ndarray, amount: float) -> np.ndarray:
    """Move the black point."""
    if amount == 0:
        return img.copy()
    lum = luminance(img)
    w = (1.0 - smoothstep(0.0, 0.3, lum))[..., None]
    lin = srgb_to_linear(img)
    out = lin + (amount / 100.0) * 0.12 * w
    return linear_to_srgb(np.clip(out, 0.0, None))


# --------------------------------------------------------------------------
# presence  (texture, clarity, dehaze, vibrance, saturation)
# --------------------------------------------------------------------------

def texture(img: np.ndarray, amount: float) -> np.ndarray:
    """Mid-frequency local contrast, applied to luminance only.

    Per-channel local contrast would shift hue at edges (see `sharpening`),
    so the mid-frequency band is extracted from luminance and added back to
    all three channels equally.
    """
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    lum = luminance(img)
    base = gaussian(lum[..., None], 2.0)[..., 0]
    mid = lum - base
    return np.clip(img + (mid * k * 1.2)[..., None], 0.0, 1.0).astype(np.float32)


def clarity(img: np.ndarray, amount: float) -> np.ndarray:
    """Large-radius local contrast with mid-tone weighting, on luminance only."""
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    lum = luminance(img)
    sigma = max(4.0, min(img.shape[:2]) * 0.012)
    base = gaussian(lum[..., None], sigma)[..., 0]
    detail = lum - base
    # strongest in the mid-tones, tapering off at both ends
    weight = (1.0 - np.abs(lum - 0.5) * 2.0).clip(0.0, 1.0)
    return np.clip(img + (detail * k * 1.5 * weight)[..., None], 0.0, 1.0).astype(np.float32)


def dehaze(img: np.ndarray, amount: float) -> np.ndarray:
    """Contrast-stretch in a haze-attenuated space.

    Positive values remove haze; negative values add it.
    """
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    if k > 0:
        # estimate the atmospheric veil from the darkest-channel minimum
        dark = img.min(axis=2)
        atmos = np.percentile(dark, 99.5)
        atmos = float(np.clip(atmos, 0.05, 1.0))
        trans = 1.0 - 0.85 * k * (dark / max(atmos, 1e-6))
        trans = np.clip(trans, 0.15, 1.0)[..., None]
        out = (img - atmos) / trans + atmos
    else:
        veil = -k * 0.35
        out = img * (1.0 - veil) + veil
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    """Saturation boost that protects already-saturated colours and skin.

    Skin protection matters: a naive vibrance pass pushes the transition
    between skin and background toward magenta/cyan, which reads as a halo
    around the face.
    """
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    hsv = rgb_to_hsv(img)
    s = hsv[..., 1]
    hue = hsv[..., 0]           # OpenCV hue is 0..179, i.e. degrees / 2

    # protect saturated pixels
    protect = 1.0 - s

    # protect skin hues. Skin spans roughly 10-40 degrees, i.e. 5-20 in
    # OpenCV's 0..179 hue; widen the band and require some saturation so that
    # neutral pixels are not accidentally treated as skin.
    skin_hue = ((hue >= 3) & (hue <= 24)).astype(np.float32)
    skin = skin_hue * smoothstep(0.10, 0.30, s)

    weight = protect * (1.0 - 0.6 * skin)

    # Keep the effect out of the deep shadows. Anti-aliased edge pixels where
    # dark hair meets a cool background are a *blend* of two colours, so they
    # carry an intermediate hue (brown + blue-grey reads as purple). Scaling
    # saturation there amplifies that blend into a visible colour rim, which is
    # the classic vibrance fringe. Vibrance is a mid-tone and skin control, so
    # excluding the shadows costs nothing and removes the artefact.
    lum = luminance(img)
    shadow_guard = smoothstep(0.10, 0.45, lum)

    hsv[..., 1] = np.clip(s + k * weight * shadow_guard * (1.0 - s), 0.0, 1.0)
    return hsv_to_rgb(hsv)


def saturation(img: np.ndarray, amount: float) -> np.ndarray:
    """Uniform saturation change, guarded in the deep shadows.

    Like `vibrance`, saturation work in HSV is unstable on near-black pixels:
    a small hue error there becomes a visible colour rim on dark edges.
    """
    if amount == 0:
        return img.copy()
    k = 1.0 + amount / 100.0
    hsv = rgb_to_hsv(img)
    lum = luminance(img)
    guard = smoothstep(0.05, 0.25, lum)
    s = hsv[..., 1]
    hsv[..., 1] = np.clip(s * (1.0 + (k - 1.0) * guard), 0.0, 1.0)
    return hsv_to_rgb(hsv)


# --------------------------------------------------------------------------
# colour grading  (group "Ca": eight hue bands; group "_a": camera primaries)
# --------------------------------------------------------------------------

HSL_BANDS = {
    "red":     (-15, 15),
    "orange":  (15, 45),
    "yellow":  (45, 75),
    "green":   (75, 165),
    "aqua":    (165, 195),
    "blue":    (195, 255),
    "purple":  (255, 285),
    "magenta": (285, 345),
}


def hsl_band(img: np.ndarray, band: str, hue: float = 0.0,
             sat: float = 0.0, lum: float = 0.0) -> np.ndarray:
    """Adjust one hue band.

    `hue` uses the PS/ACR unit -100..+100, which maps to a ±30° rotation of
    the affected band (the slider a photographer knows; the old ±60 degrees
    read as a different tool). `sat`/`lum` are -100..100.
    """
    if band not in HSL_BANDS:
        raise ValueError(f"unknown HSL band: {band}")
    if hue == 0 and sat == 0 and lum == 0:
        return img.copy()

    lo, hi = HSL_BANDS[band]
    hsv = rgb_to_hsv(img)
    # OpenCV float32 HSV: H 已经是 0..360 度（纯红=0、纯黄=60，实测）。
    # 旧代码写 `hsv[...,0] * 2` 把它当成 0..180 又翻倍，8 个色带的中心
    # 全部错位一倍——只是 span 同步翻倍后仍盖满整环，视觉上"能用"。
    h_deg = hsv[..., 0].astype(np.float32)              # 0..360

    # wrap-aware band membership with a soft 15-degree shoulder
    span = hi - lo
    d = np.abs(((h_deg - (lo + hi) / 2.0 + 180.0) % 360.0) - 180.0)
    mask = 1.0 - smoothstep(span / 2.0, span / 2.0 + 15.0, d)
    mask = (mask * (hsv[..., 1] > 0.05)).astype(np.float32)

    if hue != 0:
        # PS unit -> degrees: ±100 = ±30°.
        deg = max(-30.0, min(30.0, hue * 0.30))
        hsv[..., 0] = (hsv[..., 0] + deg * mask) % 360.0
    if sat != 0:
        s = hsv[..., 1]
        hsv[..., 1] = np.clip(s * (1.0 + (sat / 100.0) * mask), 0.0, 1.0)
    out = hsv_to_rgb(hsv)

    if lum != 0:
        lin = srgb_to_linear(out)
        lin = lin * (1.0 + (lum / 100.0) * 0.35 * mask[..., None])
        out = linear_to_srgb(np.clip(lin, 0.0, None))
    return out.astype(np.float32)


def camera_primaries(img: np.ndarray, r_hue: float = 0.0, r_sat: float = 0.0,
                     g_hue: float = 0.0, g_sat: float = 0.0,
                     b_hue: float = 0.0, b_sat: float = 0.0) -> np.ndarray:
    """Per-primary hue/saturation tweaks, the classic camera-calibration control."""
    out = img
    if r_hue or r_sat:
        out = hsl_band(out, "red", r_hue, r_sat, 0.0)
    if g_hue or g_sat:
        out = hsl_band(out, "green", g_hue, g_sat, 0.0)
    if b_hue or b_sat:
        out = hsl_band(out, "blue", b_hue, b_sat, 0.0)
    return out


# --------------------------------------------------------------------------
# detail: sharpening and noise reduction  (group "ut")
# --------------------------------------------------------------------------

def sharpening(img: np.ndarray, amount: float = 0.0, radius: float = 1.0,
               detail: float = 25.0) -> np.ndarray:
    """Unsharp mask applied to luminance only.

    Sharpening each RGB channel independently makes the three channels
    overshoot by different amounts at a strong edge, which shows up as a
    magenta/cyan fringe along high-contrast boundaries (hair against
    background, for example). Sharpening the luminance channel and leaving
    chroma untouched avoids that artefact entirely.
    """
    if amount == 0:
        return img.copy()
    k = amount / 100.0
    gate = (detail / 100.0) * 0.02

    lum = luminance(img)
    lum_sharp = unsharp(lum[..., None], max(0.3, radius), k * 1.5, gate)[..., 0]
    # apply the luminance delta to every channel, so hue and saturation hold
    delta = (lum_sharp - lum)[..., None]
    return np.clip(img + delta, 0.0, 1.0).astype(np.float32)


def noise_reduction(img: np.ndarray, luminance: float = 0.0,
                    color: float = 0.0, detail: float = 50.0,
                    smoothness: float = 50.0) -> np.ndarray:
    """Edge-aware chroma/luma denoise built on a guided filter."""
    if luminance == 0 and color == 0:
        return img.copy()

    out = img
    if luminance > 0:
        k = luminance / 100.0
        guide = luminance_channel = (0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2])
        radius = int(2 + smoothness / 100.0 * 6)
        eps = (0.004 + (1.0 - detail / 100.0) * 0.03) ** 2
        smooth = np.stack([guide] * 3, axis=2)
        smoothed = _guided_rgb(guide, out, radius, eps)
        out = blend(out, smoothed, np.full(out.shape[:2], k, np.float32))

    if color > 0:
        k = color / 100.0
        # denoise chroma only, preserving luma detail
        ycc = cv2.cvtColor(out, cv2.COLOR_RGB2YCrCb)
        radius = int(2 + smoothness / 100.0 * 6)
        cb = cv2.medianBlur((ycc[..., 1] * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        cr = cv2.medianBlur((ycc[..., 2] * 255).astype(np.uint8), 5).astype(np.float32) / 255.0
        ycc[..., 1] = ycc[..., 1] * (1 - k) + cb * k
        ycc[..., 2] = ycc[..., 2] * (1 - k) + cr * k
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _guided_rgb(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    from .imgutil import box
    g = guide.astype(np.float32)
    if g.ndim == 2:
        # 亮度引导是 (H,W)：升成 (H,W,1) 才能与 (H,W,3) 的源图广播。
        # 此前 2D 引导 × 3D 源在 cov 一步直接崩——只在无 torch 的机器上
        # 触发（有 torch 时走神经网络降噪，保底路径从未被执行过）。
        g = g[..., None]
    p = src.astype(np.float32)
    mean_g = box(g, radius)
    mean_p = box(p, radius)
    cov = box(g * p, radius) - mean_g * mean_p
    var = box(g * g, radius) - mean_g * mean_g
    a = cov / (var + eps)
    b = mean_p - a * mean_g
    return (box(a, radius) * g + box(b, radius)).reshape(p.shape).astype(np.float32)


def ai_denoise(img: np.ndarray, enabled: bool, strength: float = 50.0) -> np.ndarray:
    """Neural denoising with a graceful fallback.

    A trained checkpoint (`models/denoise.pt`) is used when present. The
    network was trained on synthesised read + shot noise across a range of
    levels, so `strength` selects the level it assumes and blends the result
    with the input. Without the checkpoint the strong edge-aware filter below
    still runs, so the control does something honest rather than erroring.
    """
    if not enabled:
        return img.copy()
    k = strength / 100.0

    try:
        from .nndenoise import available, denoise
        if available():
            return denoise(img, strength=strength)
    except Exception:
        # a missing/incompatible torch install must not break the pipeline
        pass

    guide = luminance(img)
    radius = 4
    eps = 0.012 ** 2
    smoothed = _guided_rgb(guide, img, radius, eps)
    smoothed = _guided_rgb(luminance(smoothed), smoothed, radius, eps)
    return blend(img, smoothed, np.full(img.shape[:2], k, np.float32))


# --------------------------------------------------------------------------
# grain  (group "aa")
# --------------------------------------------------------------------------

def grain(img: np.ndarray, amount: float = 0.0, size: float = 25.0,
          roughness: float = 50.0, shadows: float = 0.0,
          midtones: float = 0.0, highlights: float = 0.0,
          seed: int = 12345) -> np.ndarray:
    """Film-grain emulation with per-tonal-range weighting."""
    if amount == 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]

    sigma = max(0.4, size / 100.0 * 3.0)
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = gaussian2d(noise, sigma)
    # normalise so `size` does not change perceived strength
    std = float(noise.std())
    if std > 1e-6:
        noise /= std

    # roughness mixes in a finer-grained component
    rough = roughness / 100.0
    fine = rng.standard_normal((h, w)).astype(np.float32)
    noise = noise * (1.0 - rough) + fine * rough

    lum = luminance(img)
    w_sh = (1.0 - smoothstep(0.0, 0.5, lum)) * (shadows / 100.0 if shadows else 1.0)
    w_mid = (1.0 - np.abs(lum - 0.5) * 2.0).clip(0, 1) * (midtones / 100.0 if midtones else 1.0)
    w_hi = smoothstep(0.5, 1.0, lum) * (highlights / 100.0 if highlights else 1.0)
    weight = (w_sh + w_mid + w_hi)
    peak = max(weight.max(), 1e-6)
    weight = weight / peak

    k = amount / 100.0 * 0.09
    out = img + (noise * weight * k)[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# optics  (group "Ot": lens correction; group "Xt": geometry)
# --------------------------------------------------------------------------

def lens_correction(img: np.ndarray, distortion: float = 0.0,
                    vignetting: float = 0.0, midpoint: float = 50.0) -> np.ndarray:
    """Barrel/pincushion correction plus vignette add or removal."""
    out = img
    h, w = out.shape[:2]

    if distortion != 0:
        k = distortion / 100.0
        K = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]], np.float32)
        # positive slider removes barrel distortion (k1 < 0 in OpenCV terms)
        fx = fy = float(max(h, w))
        cam = np.array([[fx, 0, w / 2.0], [0, fy, h / 2.0], [0, 0, 1.0]], np.float32)
        dist = np.array([-k * 0.35, 0.0, 0.0, 0.0], np.float32)
        out = cv2.undistort(out, cam, dist, None, cam)

    if vignetting != 0:
        k = vignetting / 100.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
        m = np.clip(midpoint / 100.0, 0.05, 0.95)
        fall = 1.0 - k * 0.9 * np.clip((r - m) / max(1.0 - m, 1e-3), 0.0, 1.0) ** 2
        out = out * fall[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def geometry(img: np.ndarray, vertical: float = 0.0, horizontal: float = 0.0,
             rotate: float = 0.0, scale: float = 100.0,
             offset_x: float = 0.0, offset_y: float = 0.0) -> np.ndarray:
    """Perspective, rotation, scale and offset in one warp."""
    if not any([vertical, horizontal, rotate, offset_x, offset_y]) and scale == 100.0:
        return img.copy()
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    s = scale / 100.0

    M = np.array([[s, 0.0, offset_x / 100.0 * w], [0.0, s, offset_y / 100.0 * h]], np.float32)
    M[0, 2] += cx * (1 - s)
    M[1, 2] += cy * (1 - s)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                         borderMode=cv2.BORDER_REFLECT)

    if rotate:
        R = cv2.getRotationMatrix2D((cx, cy), -rotate, 1.0)
        out = cv2.warpAffine(out, R, (w, h), flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT)

    if vertical or horizontal:
        kx = -horizontal / 100.0 * 0.35
        ky = vertical / 100.0 * 0.35
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([
            [0 + kx * w * 0.5, 0 + ky * h * 0.5],
            [w - kx * w * 0.5, 0 + ky * h * 0.5],
            [w, h],
            [0, h],
        ])
        P = cv2.getPerspectiveTransform(src, dst)
        out = cv2.warpPerspective(out, P, (w, h), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# white balance
# --------------------------------------------------------------------------

def white_balance(img: np.ndarray, temp: float = 0.0, tint: float = 0.0,
                  mode: str = "as_shot") -> np.ndarray:
    """Temperature/tint adjustment; `temp` warm-positive, `tint` magenta-positive."""
    out = img
    if mode == "auto":
        # grey-world estimate, applied gently
        means = out.reshape(-1, 3).mean(axis=0)
        grey = float(means.mean())
        gains = np.clip(grey / np.maximum(means, 1e-6), 0.85, 1.18)
        out = np.clip(out * gains, 0.0, 1.0).astype(np.float32)
    if temp == 0 and tint == 0:
        return out
    gains = np.ones(3, np.float32)
    gains[0] *= 1.0 + temp / 100.0 * 0.22          # red up = warmer
    gains[2] *= 1.0 - temp / 100.0 * 0.22          # blue down = warmer
    gains[1] *= 1.0 - tint / 100.0 * 0.10
    gains[0] *= 1.0 + tint / 100.0 * 0.05
    gains[2] *= 1.0 + tint / 100.0 * 0.05
    return np.clip(out * gains, 0.0, 1.0).astype(np.float32)


def gains_to_temp_tint(gains) -> dict:
    """Invert the gain model above, so an estimated correction can be shown as
    temperature/tint instead of an opaque multiplier.

    Derivation — the model is

        R = (1 + t·0.22)(1 + n·0.05)
        G = (1 - n·0.10)
        B = (1 - t·0.22)(1 + n·0.05)

    with t = temp/100 and n = tint/100. The tint factor is common to R and B,
    so the R:B ratio isolates the temperature term, and G alone gives tint.
    """
    import numpy as _np

    g = _np.asarray(gains, _np.float64).reshape(3)
    g = _np.maximum(g, 1e-6)
    # tint from the green channel
    n = (1.0 - g[1]) / 0.10
    # temperature from the R:B ratio, independent of the common tint factor
    r = g[0] / max(g[2], 1e-6)
    a = (r - 1.0) / (r + 1.0)
    t = a / 0.22
    return {"temp": float(_np.clip(t * 100.0, -100.0, 100.0)),
            "tint": float(_np.clip(n * 100.0, -100.0, 100.0))}


def chromatic_aberration(img: np.ndarray, amount: float = 0.0) -> np.ndarray:
    """轴向色差校正：R/B 通道绕光心做反向径向缩放。

    amount > 0 校正"红边"（R 通道外扩补偿），< 0 校正"蓝边"。
    色差本质是各波长焦距不同 → 通道间径向放大率差，按线性径向
    缩放建模已能消掉大部分紫边/绿边，足够手动滑杆使用。
    """
    if not amount:
        return img
    h, w = img.shape[:2]
    k = amount * 0.0022
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = (xs - (w - 1) * 0.5) / (w * 0.5)
    ys = (ys - (h - 1) * 0.5) / (w * 0.5)      # 与宽同尺度，各向同性
    out = img.copy()
    for ch, scale in ((0, 1.0 + k), (2, 1.0 - k)):
        mx = xs * scale * (w * 0.5) + (w - 1) * 0.5
        my = ys * scale * (w * 0.5) + (h - 1) * 0.5
        out[..., ch] = cv2.remap(img[..., ch], mx, my,
                                 cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    return out
