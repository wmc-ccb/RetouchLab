# -*- coding: utf-8 -*-
"""后调色 / 配置文件 / 滤镜 / 可选颜色 / 色彩编辑器.

These five share one idea: a colour decision made **per hue** rather than per
pixel. That is why they live together — each one is a different way of asking
"which colours am I talking about" and then moving them:

* **可选颜色** — a CMYK-style pull on one hue band (cyan/magenta/yellow/black).
  A red lipstick is `Reds`, a blue sky is `Cyans|Blues`; adjusting "Reds" must
  not move the blues, which is what separates this from a saturation slider.
* **色彩编辑器** — the same machinery with HSL-style targets (hue shift,
  saturation, luminance) and a pick-from-image entry point.
* **配置文件/滤镜** — a fixed transform: a camera profile is a measured
  deviation to *correct*, a filter is a look to *apply*. Both are 3D LUTs here
  so a chain of them composes into one resample instead of one per style.
* **后调色** — split toning, colour balance, vignette and a halo/glow pass,
  applied after the tone edits because they are grading, not correction.

Hue selection uses a **raised-cosine window** in Lab hue rather than a hard
range: a hard cut makes the boundary of a colour range visible as a seam in a
gradient (a sunset sky shows a band). The window sums to 1 across neighbouring
bands so overlapping selections compose smoothly.

**Units.** `imgutil.rgb_to_lab` returns a *normalised* Lab — L in 0..1 and a/b
in about -1..1 — not OpenCV's native 0..100 / ±127. Every constant below is in
those normalised units. Writing them in native units compiles and runs, and
silently does almost nothing: a chroma test of `chroma / 42` against a chroma
that tops out near 0.8 leaves a weight around 0.02, and a ±26 a/b push against a
range of ±1 is entirely clipped away.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# hue machinery
# ---------------------------------------------------------------------------

# names used by both 可选颜色 and 色彩编辑器; degrees are the band centres in Lab
HUE_BANDS: Dict[str, float] = {
    "reds": 30.0,
    "oranges": 55.0,
    "yellows": 85.0,
    "greens": 135.0,
    "cyans": 190.0,
    "blues": 245.0,
    "purples": 285.0,
    "magentas": 320.0,
}

BAND_LABELS = {
    "reds": "红", "oranges": "橙", "yellows": "黄", "greens": "绿",
    "cyans": "青", "blues": "蓝", "purples": "紫", "magentas": "洋红",
}


def _lab_hue(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lab and its hue angle in degrees, plus chroma."""
    from .imgutil import rgb_to_lab

    lab = rgb_to_lab(img)
    a = lab[..., 1]
    b = lab[..., 2]
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    chroma = np.sqrt(a * a + b * b)
    return lab, hue, chroma


def hue_weight(hue: np.ndarray, centre: float, width: float = 34.0
               ) -> np.ndarray:
    """Raised-cosine window around a hue, wrapping at 360.

    A hard threshold produces a visible seam where a gradient crosses the band
    edge. Cosine keeps the derivative zero at both ends, so the weight fades in
    and out with no step.
    """
    d = np.abs(((hue - centre + 180.0) % 360.0) - 180.0)
    return np.clip(0.5 + 0.5 * np.cos(np.pi * np.clip(d / width, 0.0, 1.0)),
                   0.0, 1.0)


# ---------------------------------------------------------------------------
# 可选颜色
# ---------------------------------------------------------------------------

def selective_color(img: np.ndarray, bands: Dict[str, Dict[str, float]],
                    width: float = 34.0,
                    protect_neutral: float = 0.35) -> np.ndarray:
    """CMYK-style adjustment per hue band.

    `bands` maps a band name to {"cyan","magenta","yellow","black"} in -100..100.
    The amount is scaled by the pixel's **chroma** so a nearly-grey pixel is
    barely touched — otherwise "increase cyan in Blues" would tint the whole
    shadow range of a grey wall.

    All offsets are in the normalised Lab units of `rgb_to_lab`: a/b span about
    ±1, so a full-scale push is 0.22, not 26.
    """
    if not bands:
        return img.copy()
    from .imgutil import lab_to_rgb

    lab, hue, chroma = _lab_hue(img)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    # 彩度满量程约 0.8（纯红/蓝），除以 0.35 让中等饱和的颜色也能吃到大部分权重
    cw = np.clip(chroma / 0.35, 0.0, 1.0)
    # 中性保护：低彩度像素的权重再压一次
    cw = cw * (1.0 - protect_neutral) + cw * protect_neutral * cw
    dA = np.zeros_like(A)
    dB = np.zeros_like(B)
    dL = np.zeros_like(L)
    for name, spec in (bands or {}).items():
        if name not in HUE_BANDS:
            continue
        if not spec:
            continue
        w = hue_weight(hue, HUE_BANDS[name], width) * cw
        if w.max() <= 1e-6:
            continue
        # CMYK 到归一化 Lab 的符号：加青 = -a（偏绿蓝），加洋红 = +a，
        # 加黄 = +b，黑 = 压 L
        cy = float(spec.get("cyan", 0.0)) / 100.0
        mg = float(spec.get("magenta", 0.0)) / 100.0
        ye = float(spec.get("yellow", 0.0)) / 100.0
        bk = float(spec.get("black", 0.0)) / 100.0
        dA += w * (-cy * 0.22 + mg * 0.22)
        dB += w * (ye * 0.22 - mg * 0.06)
        dL += w * (-bk * 0.18)
    lab2 = np.stack([np.clip(L + dL, 0.0, 1.0), A + dA, B + dB], axis=-1)
    return np.clip(lab_to_rgb(lab2.astype(np.float32)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 色彩编辑器（HSL 按色相）
# ---------------------------------------------------------------------------

def color_editor(img: np.ndarray, bands: Dict[str, Dict[str, float]],
                 width: float = 30.0) -> np.ndarray:
    """Per-band hue shift / saturation / luminance, HSL-style.

    `bands` maps band -> {"hue": -180..180, "saturation": -100..100,
    "luminance": -100..100}.
    """
    if not bands:
        return img.copy()
    from .imgutil import lab_to_rgb

    lab, hue, chroma = _lab_hue(img)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    cw = np.clip(chroma / 0.30, 0.0, 1.0)
    dh = np.zeros_like(hue)
    ds = np.zeros_like(chroma)
    dl = np.zeros_like(L)
    for name, spec in (bands or {}).items():
        if name not in HUE_BANDS or not spec:
            continue
        w = hue_weight(hue, HUE_BANDS[name], width) * cw
        if w.max() <= 1e-6:
            continue
        dh += w * float(spec.get("hue", 0.0))
        ds += w * float(spec.get("saturation", 0.0)) / 100.0
        dl += w * float(spec.get("luminance", 0.0)) / 100.0

    new_hue = np.radians((hue + dh) % 360.0)
    new_c = np.clip(chroma * (1.0 + ds), 0.0, 1.6)
    lab2 = np.stack([np.clip(L + dl * 0.30, 0.0, 1.0),
                     new_c * np.cos(new_hue),
                     new_c * np.sin(new_hue)], axis=-1)
    return np.clip(lab_to_rgb(lab2.astype(np.float32)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 配置文件 / 滤镜（3D LUT）
# ---------------------------------------------------------------------------

def _build_lut(fn, size: int = 17) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, size, dtype=np.float32)
    r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
    flat = np.stack([r.ravel(), g.ravel(), b.ravel()], axis=-1)
    out = fn(flat).reshape(size, size, size, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_lut(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Trilinear lookup. One resample for a whole chain if the chain is baked."""
    import torch

    n = lut.shape[0]
    x = np.clip(img, 0.0, 1.0) * (n - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, n - 1)
    f = (x - i0).astype(np.float32)
    i0c = np.clip(i0, 0, n - 1)
    out = np.zeros_like(img, dtype=np.float32)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                wr = f[..., 0] if dr else 1 - f[..., 0]
                wg = f[..., 1] if dg else 1 - f[..., 1]
                wb = f[..., 2] if db else 1 - f[..., 2]
                rr = i1[..., 0] if dr else i0c[..., 0]
                gg = i1[..., 1] if dg else i0c[..., 1]
                bb = i1[..., 2] if db else i0c[..., 2]
                out += (wr * wg * wb)[..., None] * lut[rr, gg, bb]
    return np.clip(out, 0.0, 1.0)


# 机内配置文件：把它当作需要"校正"的偏差来模拟。名称是通用的影调描述，
# 不复制任何厂商的私有数据；曲线形状为公开的影调响应常识。
PROFILES = {
    "neutral": {"desc": "中性：不做任何偏移",
                "contrast": 0.0, "sat": 1.0, "hue_shift": 0.0,
                "lift": 0.0},
    "standard": {"desc": "标准：轻微对比与饱和",
                 "contrast": 0.10, "sat": 1.08, "hue_shift": 0.0,
                 "lift": 0.0},
    "portrait": {"desc": "人像：肤色偏暖、对比柔和",
                 "contrast": -0.04, "sat": 1.04, "hue_shift": 3.0,
                 "lift": 0.012},
    "landscape": {"desc": "风光：绿蓝更浓、对比更强",
                  "contrast": 0.16, "sat": 1.16, "hue_shift": -2.0,
                  "lift": 0.0},
    "flat": {"desc": "平坦：低对比，便于后续调色",
             "contrast": -0.22, "sat": 1.02, "hue_shift": 0.0,
             "lift": 0.02},
    "mono": {"desc": "黑白：去色保留明度",
             "contrast": 0.14, "sat": 0.0, "hue_shift": 0.0,
             "lift": 0.0},
}


def profile_lut(name: str) -> Optional[np.ndarray]:
    p = PROFILES.get(name)
    if not p or name == "neutral":
        return None

    def fn(flat: np.ndarray) -> np.ndarray:
        out = flat.astype(np.float32)
        c = p["contrast"]
        if c:
            out = np.clip((out - 0.5) * (1.0 + c) + 0.5, 0, 1)
        if p["lift"]:
            out = np.clip(out * (1.0 - p["lift"]) + p["lift"], 0, 1)
        s = p["sat"]
        if abs(s - 1.0) > 1e-6:
            lum = (out * np.array([0.2126, 0.7152, 0.0722],
                                  np.float32)).sum(axis=-1, keepdims=True)
            out = np.clip(lum + (out - lum) * s, 0, 1)
        hs = p["hue_shift"]
        if abs(hs) > 1e-6:
            hr = np.radians(hs)
            ca, sa = np.cos(hr), np.sin(hr)
            y = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
            i = 0.596 * out[..., 0] - 0.274 * out[..., 1] - 0.322 * out[..., 2]
            q = 0.211 * out[..., 0] - 0.523 * out[..., 1] + 0.312 * out[..., 2]
            ni = i * ca - q * sa
            nq = i * sa + q * ca
            r = y + 0.956 * ni + 0.621 * nq
            g = y - 0.272 * ni - 0.647 * nq
            b = y - 1.106 * ni + 1.703 * nq
            out = np.stack([r, g, b], axis=-1)
        return np.clip(out, 0, 1)

    return _build_lut(fn)


# 滤镜：风格化外观。刻意做成参数化曲线而不是复制任何具体厂牌的 LUT 文件，
# 这些是公开的影调/色彩关系。
FILTERS = {
    "none": {"desc": "无"},
    "clean": {"desc": "干净：提亮暗部，略微降饱和"},
    "warm": {"desc": "暖调：高光偏暖，阴影略压青"},
    "cool": {"desc": "冷调：整体偏蓝，红色收敛"},
    "teal_orange": {"desc": "青橙：暗部青、高光橙"},
    "matte": {"desc": "褪色：黑场抬高，饱和降低"},
    "punch": {"desc": "浓郁：对比与饱和同时提升"},
    "faded_film": {"desc": "胶片感：柔和高光，绿色偏移"},
    "bw_contrast": {"desc": "黑白高反差"},
}


def filter_lut(name: str) -> Optional[np.ndarray]:
    if name in (None, "", "none"):
        return None
    # 未知名字必须返回 None，而不是恒等 LUT：调用方靠 None 判断"没有可应用的
    # 效果"，返回一个恒等表会让 API 报告 applied 了某个不存在的滤镜。
    if name not in FILTERS:
        return None

    def fn(flat: np.ndarray) -> np.ndarray:
        x = flat.astype(np.float32)
        lum = (x * np.array([0.2126, 0.7152, 0.0722],
                            np.float32)).sum(axis=-1, keepdims=True)
        if name == "clean":
            x = np.clip(x ** 0.96, 0, 1)
            x = np.clip(lum + (x - lum) * 0.93, 0, 1)
        elif name == "warm":
            x = np.clip(x * np.array([1.04, 1.0, 0.95], np.float32), 0, 1)
            x = np.clip(x + (0.012 * (1 - lum)), 0, 1)
        elif name == "cool":
            x = np.clip(x * np.array([0.96, 0.99, 1.05], np.float32), 0, 1)
        elif name == "teal_orange":
            sh = np.clip(1.0 - lum * 1.6, 0, 1)
            hi = np.clip((lum - 0.5) * 1.8, 0, 1)
            x = np.clip(x + sh * np.array([-0.02, 0.015, 0.035], np.float32) +
                        hi * np.array([0.035, 0.012, -0.03], np.float32), 0, 1)
            x = np.clip(lum + (x - lum) * 1.06, 0, 1)
        elif name == "matte":
            x = np.clip(x * 0.88 + 0.055, 0, 1)
            x = np.clip(lum + (x - lum) * 0.86, 0, 1)
        elif name == "punch":
            x = np.clip((x - 0.5) * 1.18 + 0.5, 0, 1)
            x = np.clip(lum + (x - lum) * 1.22, 0, 1)
        elif name == "faded_film":
            x = np.clip((x - 0.5) * 0.94 + 0.52, 0, 1)
            x = np.clip(x + np.array([0.012, 0.006, -0.01], np.float32), 0, 1)
            x = np.clip(lum + (x - lum) * 0.90, 0, 1)
        elif name == "bw_contrast":
            g = np.clip((lum - 0.5) * 1.32 + 0.5, 0, 1)
            x = np.repeat(g, 3, axis=-1)
        return np.clip(x, 0, 1)

    return _build_lut(fn)


# ---------------------------------------------------------------------------
# 后调色
# ---------------------------------------------------------------------------

def post_grade(img: np.ndarray, shadow_tint: Sequence[float] = (0, 0, 0),
               highlight_tint: Sequence[float] = (0, 0, 0),
               balance: float = 0.0, vignette: float = 0.0,
               glow: float = 0.0, temperature: float = 0.0) -> np.ndarray:
    """Split toning + colour balance + vignette + bloom.

    Split toning weights by **luminance** with a smooth crossover so the
    midpoint is not a hard boundary: a two-tone image with a hard split shows a
    contour wherever mid-grey sits.
    """
    out = np.asarray(img, np.float32).copy()
    from .imgutil import luminance

    lum = luminance(out)[..., None]
    if any(abs(v) > 1e-6 for v in shadow_tint) or \
            any(abs(v) > 1e-6 for v in highlight_tint):
        # balance 移动交叉点：正值把更多画面归入高光
        mid = 0.5 + float(balance) / 100.0 * 0.3
        lo = np.clip(1.0 - lum / max(mid, 1e-3), 0, 1) ** 1.4
        hi = np.clip((lum - mid) / max(1.0 - mid, 1e-3), 0, 1) ** 1.4
        st = np.array(shadow_tint, np.float32).reshape(1, 1, 3)
        ht = np.array(highlight_tint, np.float32).reshape(1, 1, 3)
        out = np.clip(out + lo * st * 0.10 + hi * ht * 0.10, 0, 1)

    if abs(temperature) > 1e-6:
        t = float(temperature) / 100.0
        out = np.clip(out * np.array([1 + 0.08 * t, 1.0, 1 - 0.08 * t],
                                     np.float32), 0, 1)

    if vignette > 0:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2.0, h / 2.0
        r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
        v = 1.0 - float(vignette) / 100.0 * np.clip(r, 0, 1) ** 2.2
        out = np.clip(out * v[..., None], 0, 1)

    if glow > 0:
        k = max(3, int(min(out.shape[:2]) * 0.02)) * 2 + 1
        blurred = cv2.GaussianBlur(out, (k, k), 0)
        band = np.clip((lum - 0.5) * 2.0, 0, 1)
        out = np.clip(out + (blurred - out) * band *
                      (float(glow) / 100.0 * 1.6), 0, 1)
    return out


def info() -> dict:
    return {"hue_bands": BAND_LABELS,
            "profiles": {k: v["desc"] for k, v in PROFILES.items()},
            "filters": {k: v["desc"] for k, v in FILTERS.items()},
            "selective_color": ["cyan", "magenta", "yellow", "black"],
            "color_editor": ["hue", "saturation", "luminance"],
            "post_grade": ["shadow_tint", "highlight_tint", "balance",
                           "vignette", "glow", "temperature"]}
