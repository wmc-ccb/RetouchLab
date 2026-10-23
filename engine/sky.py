# -*- coding: utf-8 -*-
"""换天空：把画面上部的天空区域换成程序化生成的新天空.

像素蛋糕「图像美化 → 换天空」的确定性实现（不需要生成模型）：

  1. **天空判据**：背景分割 ∩（蓝色色相带 / 高亮度低饱和的过曝灰）。
     只用颜色会在白色衣服上误判，所以必须先用 selfie_multiclass 的
     background 类把人物/前景排除。
  2. **地平线约束**：天空在画面上部。对逐行天空似然做累计分析，从上往
     下找到"天空性"衰减到 50% 的行，该行以下强制清零——地面上的蓝天
     反光（玻璃幕墙、水面）不该被替换。
  3. **程序化新天空**：竖直渐变 + 太阳光晕 + Perlin 风格噪声云。亮度
     与原天空匹配（地平线处最亮），避免合成感。
  4. **羽化合成**：蒙版先腐蚀再高斯，树梢/天线这类细结构不被硬边切断。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# 展示顺序即 UI 顺序；每个 preset 是 (顶色, 地平线色, 太阳位置, 云量)
PRESETS = [
    ("clear_blue", "晴空", (0.30, 0.55, 0.95), (0.72, 0.85, 1.00),
     (0.50, 0.18), 0.15),
    ("deep_blue", "深邃蓝天", (0.12, 0.28, 0.75), (0.55, 0.72, 0.98),
     (0.50, 0.15), 0.08),
    ("sunset", "夕阳金橙", (0.35, 0.45, 0.85), (1.00, 0.72, 0.42),
     (0.62, 0.55), 0.30),
    ("dusk", "暮色紫", (0.22, 0.18, 0.45), (0.92, 0.62, 0.68),
     (0.45, 0.50), 0.25),
    ("overcast", "柔灰阴天", (0.55, 0.58, 0.64), (0.92, 0.93, 0.95),
     (0.50, 0.30), 0.55),
    ("dawn", "晨曦青粉", (0.45, 0.62, 0.88), (1.00, 0.86, 0.78),
     (0.38, 0.48), 0.20),
]


def _noise(h: int, w: int, seed: int = 0, octaves: int = 4) -> np.ndarray:
    """Cheap fractal value noise for clouds (numpy only, tileable-ish)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(h / (2 ** (octaves - o))))
        gw = max(2, int(w / (2 ** (octaves - o))))
        g = rng.random((gh, gw)).astype(np.float32)
        # 双线性放大到全尺寸
        yi = np.linspace(0, gh - 1, h).astype(np.int32)
        xi = np.linspace(0, gw - 1, w).astype(np.int32)
        y0 = np.clip(yi, 0, gh - 2); x0 = np.clip(xi, 0, gw - 2)
        fy = (yi - y0)[:, None]; fx = (xi - x0)[None, :]
        g00 = g[np.ix_(y0, x0)]
        g01 = g[np.ix_(y0, x0 + 1)]
        g10 = g[np.ix_(y0 + 1, x0)]
        g11 = g[np.ix_(y0 + 1, x0 + 1)]
        acc += amp * ((g00 * (1 - fy) * (1 - fx) + g01 * (1 - fy) * fx +
                       g10 * fy * (1 - fx) + g11 * fy * fx))
        total += amp
        amp *= 0.5
    return acc / max(total, 1e-6)


def _sky_likelihood(img: np.ndarray) -> np.ndarray:
    """Per-pixel sky-likeness in 0..1 (colour only, no position)."""
    from .imgutil import rgb_to_hsv

    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    h_deg = hsv[..., 0].astype(np.float32)      # 0..360 (measured)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)

    # 蓝天：190..250°，饱和适中
    blue = 1.0 - np.clip(np.abs(h_deg - 220.0) / 40.0, 0, 1)
    blue *= np.clip((s - 0.08) / 0.25, 0, 1) * np.clip(v / 0.5, 0, 1)
    # 过曝/阴天：极亮 + 极低饱和
    bright = np.clip((v - 0.72) / 0.2, 0, 1) * np.clip((0.18 - s) / 0.18, 0, 1)
    return np.clip(np.maximum(blue, bright), 0, 1).astype(np.float32)


def detect_sky(img: np.ndarray, bg_mask: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, dict]:
    """Return (sky mask 0..1, info).

    `bg_mask` 是 selfie_multiclass 的 background 类；给了它就把人物/
    前景直接从候选里排除，没有就退化为纯颜色+位置判据（白墙会把整面
    墙判成天空，实测过，所以有分割时优先用分割）。
    """
    import cv2

    h, w = img.shape[:2]
    like = _sky_likelihood(img)
    if bg_mask is not None:
        like = like * np.clip(np.asarray(bg_mask, np.float32), 0, 1)

    # 地平线：逐行平均似然，从上向下累计；衰减到峰值 45% 的行以下清零
    row = like.mean(axis=1)
    peak = float(row[: max(2, h // 4)].max()) if h > 8 else float(row.max())
    horizon = int(h * 0.85)
    if peak > 0.05:
        thr = peak * 0.45
        below = np.where(row < thr)[0]
        if len(below):
            horizon = int(below[0])
    rows = np.arange(h, dtype=np.float32)[:, None]
    fade = np.clip((horizon + h * 0.05 - rows) / (h * 0.05), 0, 1)
    like = like * fade

    # 上部强先验：越靠上越可能是天空（避免把地面亮斑算进去）
    prior = np.clip(1.0 - rows / (h * 0.75), 0.25, 1.0)
    like = like * prior

    m = np.clip((like - 0.25) / 0.30, 0, 1).astype(np.float32)
    # 形态学收边 + 羽化：先腐蚀掉细碎误判，再高斯羽化
    u8 = (m > 0.5).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    m = cv2.GaussianBlur(np.maximum(m * 0.5, u8.astype(np.float32)),
                         (0, 0), max(2.0, w / 300.0))
    m = np.clip(m, 0, 1)
    info = {"horizon_row": horizon, "coverage": round(float((m > 0.5).mean()), 4),
            "used_segmentation": bg_mask is not None}
    return m, info


def _sky_texture(h: int, w: int, top, low, sun, clouds: float,
                 seed: int) -> np.ndarray:
    """Procedural sky: vertical gradient + sun glow + fractal clouds."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top = np.array(top, np.float32)[None, None, :]
    low = np.array(low, np.float32)[None, None, :]
    # gamma>1 让地平线更亮（大气散射）
    t = (yy ** 1.6)[..., None]
    img = top * (1 - t) + low * t

    sx, sy = sun
    xs = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    d = np.sqrt((xs - sx) ** 2 + (yy - sy) ** 2)
    glow = np.clip(1.0 - d / 0.75, 0, 1) ** 2.2
    glow = np.broadcast_to(glow, (h, w)).astype(np.float32)
    img = img + (glow * 0.35)[..., None] * \
        np.array([1.0, 0.9, 0.7], np.float32)

    if clouds > 0.01:
        n = _noise(h, w, seed=seed)
        band = np.clip(1.0 - np.abs(yy - 0.32) / 0.5, 0.25, 1.0)  # 云集中在上部
        c = np.clip((n - 0.55) / 0.25, 0, 1) * band * clouds
        img = img * (1 - c[..., None]) + \
            np.array([1.0, 1.0, 1.0], np.float32) * c[..., None]
    return np.clip(img, 0, 1)


def replace_sky(img: np.ndarray, preset: str = "clear_blue",
                bg_mask: Optional[np.ndarray] = None,
                strength: float = 1.0,
                match_brightness: bool = True) -> Tuple[np.ndarray, dict]:
    """把天空换成 `preset`；strength 0..1 控制混合强度."""
    import cv2

    from .imgutil import luminance

    p = next((x for x in PRESETS if x[0] == preset), None)
    if p is None:
        raise ValueError(f"unknown sky preset: {preset}")
    _, label, top, low, sun, clouds = p
    h, w = img.shape[:2]

    mask, dinfo = detect_sky(img, bg_mask)
    if match_brightness and (mask > 0.5).sum() > 50:
        # 原天空的亮度分布（上亮下暗），把新天空逐段缩放过去
        lum = luminance(img)
        ys = np.arange(h) / max(h - 1, 1)
        src_hi = float(np.median(lum[: h // 4][mask[: h // 4] > 0.5])) \
            if (mask[: h // 4] > 0.5).sum() > 20 else None
        src_lo = float(np.median(lum[-h // 4:][mask[-h // 4:] > 0.5])) \
            if (mask[-h // 4:] > 0.5).sum() > 20 else None
        tex = _sky_texture(h, w, top, low, sun, clouds, seed=int(w))
        if src_hi is not None:
            tex_hi = float(tex[: h // 4].mean())
            tex_lo = float(tex[-h // 4:].mean())
            gain = (src_hi / max(tex_hi, 1e-3))
            gain = float(np.clip(gain, 0.55, 1.8))
            tex = np.clip(tex * gain, 0, 1)
            dinfo["brightness_gain"] = round(gain, 3)
    else:
        tex = _sky_texture(h, w, top, low, sun, clouds, seed=int(w))

    a = (np.clip(mask, 0, 1) * float(np.clip(strength, 0, 1)))[..., None]
    out = np.clip(img * (1 - a) + tex * a, 0, 1).astype(np.float32)
    dinfo["preset"] = preset
    dinfo["label"] = label
    return out, dinfo


def info() -> dict:
    return {"presets": [{"key": k, "label": v} for k, _, v, _, _, _ in
                        PRESETS],
            "pipeline": "segmentation ∩ colour-likelihood → horizon fade "
                        "→ procedural sky → feathered blend"}
