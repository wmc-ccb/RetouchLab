# -*- coding: utf-8 -*-
"""四季背景：把人物背后的场景换成季节主题（确定性，无生成模型）.

像素蛋糕「图像美化 → 春日樱花/夏日晴空/秋日落叶/冬日雪景」是生成式的
整幅背景替换；这里用**背景分割 + 程序化场景**做确定性等价：

  1. 分割给出人物（前景），其余全部算背景——与换天空不同，这里不挑
     "哪些像素是天空"，整片背景一起换。
  2. 每个季节是一个程序化场景：底色渐变（上冷下暖或相反）+ 粒子层
     （樱花瓣/云/落叶/雪花）+ 柔光。
  3. 粒子有**景深感**：远的小而淡、近的大而略糊，避免贴纸感。
  4. 合成时保留**背景的结构明暗**（可选）：把原背景的亮度以低权重
     叠回场景色上，墙面纹理/地面透视还在，只是换了季节色彩——直接
     涂平会假。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

PRESETS = [
    ("sakura", "春日樱花"),
    ("summer", "夏日晴空"),
    ("autumn", "秋日落叶"),
    ("winter", "冬日雪景"),
]


def _particles(h: int, w: int, n: int, seed: int,
               colour, size_range, near_frac: float = 0.35,
               blur_near: float = 2.2, fall_angle: float = 0.0
               ) -> np.ndarray:
    """撒粒子。远的小而淡（占 1-near_frac），近的大而糊。"""
    import cv2

    rng = np.random.default_rng(seed)
    layer = np.zeros((h, w, 3), np.float32)
    alpha = np.zeros((h, w), np.float32)
    colour = np.array(colour, np.float32)
    n_far = max(1, int(n * (1 - near_frac)))
    for i in range(n):
        near = i >= n_far
        x = int(rng.integers(0, w))
        y = int(rng.integers(0, h))
        s = rng.integers(size_range[0], size_range[1] + 1)
        if near:
            s = int(s * 1.8)
        a = float(rng.uniform(0.5, 0.95)) * (1.0 if near else 0.55)
        # 沿 fall_angle 方向拉长（花瓣/落叶有方向感；雪没有）
        ax = int(max(1, s * abs(np.cos(fall_angle))))
        ay = int(max(1, s * max(0.35, abs(np.sin(fall_angle)))))
        # OpenCV 5.0 的 ellipse 不收 np.float32 数组当颜色，转 Python 元组
        col = (float(colour[0]), float(colour[1]), float(colour[2]))
        cv2.ellipse(layer, (x, y), (ax, ay), np.degrees(fall_angle),
                    0, 360, col, -1)
        cv2.ellipse(alpha, (x, y), (ax, ay), np.degrees(fall_angle),
                    0, 360, a, -1)
    if blur_near > 0:
        near_m = np.zeros((h, w), np.uint8)
        # 近似：对整层做轻糊即可，近粒子本来更大更显
        layer = cv2.GaussianBlur(layer, (0, 0), blur_near * 0.6)
        alpha = cv2.GaussianBlur(alpha, (0, 0), blur_near * 0.6)
        _ = near_m
    return layer, np.clip(alpha, 0, 1)


def _noise(h: int, w: int, seed: int = 0, octaves: int = 4) -> np.ndarray:
    """Cheap fractal value noise (same approach as sky.py)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        gh = max(2, int(h / (2 ** (octaves - o))))
        gw = max(2, int(w / (2 ** (octaves - o))))
        g = rng.random((gh, gw)).astype(np.float32)
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


def _gradient(h: int, w: int, top, bottom, gamma: float = 1.4):
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    t = (yy ** gamma)[..., None]
    top = np.array(top, np.float32)[None, None, :]
    bottom = np.array(bottom, np.float32)[None, None, :]
    g = top * (1 - t) + bottom * t
    return np.broadcast_to(g, (h, w, 3)).astype(np.float32)


def _scene(h: int, w: int, preset: str, seed: int = 0):
    """程序化季节场景 (背景层, 粒子层, 粒子alpha, 光效)."""
    import cv2

    if preset == "sakura":
        base = _gradient(h, w, (1.00, 0.88, 0.93), (0.98, 0.80, 0.86), 1.2)
        # 远处樱花树影：顶部粉团
        n = _noise(h, w, seed=seed + 1)
        tuft = np.clip((n - 0.52) / 0.2, 0, 1) * \
            np.clip(1.0 - np.linspace(0, 1, h)[:, None] / 0.6, 0, 1)
        base = base * (1 - tuft[..., None] * 0.35) + \
            np.array([1.0, 0.72, 0.82], np.float32) * tuft[..., None] * 0.35
        layer, alpha = _particles(h, w, int(w * h / 2600), seed,
                                  (1.0, 0.80, 0.88), (3, 7), fall_angle=0.5)
        glow = np.full((h, w, 3), 0.03, np.float32)
    elif preset == "summer":
        base = _gradient(h, w, (0.25, 0.50, 0.95), (0.72, 0.85, 1.00), 1.6)
        n = _noise(h, w, seed=seed + 2)
        cloud = np.clip((n - 0.58) / 0.2, 0, 1) * \
            np.clip(1.0 - np.linspace(0, 1, h)[:, None] / 0.5, 0, 1)
        base = base * (1 - cloud[..., None] * 0.5) + \
            np.array([0.98, 0.99, 1.0], np.float32) * cloud[..., None] * 0.5
        layer, alpha = _particles(h, w, 0, seed, (1, 1, 1), (1, 1))
        glow = np.array([0.06, 0.05, 0.0], np.float32)
    elif preset == "autumn":
        base = _gradient(h, w, (0.92, 0.62, 0.30), (0.80, 0.45, 0.20), 1.3)
        n = _noise(h, w, seed=seed + 3)
        canopy = np.clip((n - 0.5) / 0.22, 0, 1) * \
            np.clip(1.0 - np.linspace(0, 1, h)[:, None] / 0.55, 0, 1)
        base = base * (1 - canopy[..., None] * 0.4) + \
            np.array([0.75, 0.35, 0.12], np.float32) * canopy[..., None] * 0.4
        layer, alpha = _particles(h, w, int(w * h / 3200), seed,
                                  (0.92, 0.45, 0.12), (4, 8), fall_angle=0.9)
        glow = np.array([0.05, 0.02, 0.0], np.float32)
    elif preset == "winter":
        base = _gradient(h, w, (0.82, 0.88, 0.96), (0.95, 0.97, 1.00), 1.3)
        layer, alpha = _particles(h, w, int(w * h / 1800), seed,
                                  (1.0, 1.0, 1.0), (2, 5), fall_angle=0.15,
                                  blur_near=1.4)
        glow = np.array([0.0, 0.01, 0.03], np.float32)
    else:
        raise ValueError(f"unknown scene preset: {preset}")
    glow = np.broadcast_to(glow, (h, w, 3)).astype(np.float32)
    return base, layer, alpha, glow


def apply_scene(img: np.ndarray, preset: str,
                subject_mask: Optional[np.ndarray] = None,
                strength: float = 1.0,
                keep_structure: float = 0.25
                ) -> Tuple[np.ndarray, dict]:
    """换季节背景。`subject_mask` 是人物前景；不给就用分割现算."""
    import cv2

    from .imgutil import luminance

    if preset not in dict(PRESETS):
        raise ValueError(f"unknown scene preset: {preset}")
    h, w = img.shape[:2]

    if subject_mask is None:
        try:
            from . import semantic as SEM

            reg = SEM.segment(img)
            subject_mask = reg.subject() if reg is not None else None
        except Exception:
            subject_mask = None
    if subject_mask is None:
        raise RuntimeError("主体分割不可用，无法安全换背景")

    subj = np.clip(np.asarray(subject_mask, np.float32), 0, 1)
    if subj.ndim == 3:
        subj = subj[..., 0]
    if subj.shape[:2] != (h, w):
        subj = cv2.resize(subj, (w, h), interpolation=cv2.INTER_LINEAR)
    # 人物保护：轻微收缩前景（dilate 后再软边），避免发丝边缘被背景色
    # 包进去产生粉/橙描边
    fg = cv2.dilate((subj > 0.5).astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)))
    fg = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.2)
    fg = np.clip(fg, 0, 1)

    base, layer, palpha, glow = _scene(h, w, preset, seed=int(w))
    scene = base + glow + layer * palpha[..., None]

    if keep_structure > 0:
        # 保留背景的结构明暗（低权重）：墙面透视/地面阴影还在
        lum = luminance(img)
        bg_lum = lum * (1 - fg)
        hi = float(np.percentile(bg_lum[bg_lum > 0], 85)) if \
            (bg_lum > 0).any() else 0.6
        lo = float(np.percentile(bg_lum[bg_lum > 0], 15)) if \
            (bg_lum > 0).any() else 0.2
        rel = np.clip((lum - lo) / max(hi - lo, 1e-3), 0, 1)
        rel = cv2.GaussianBlur(rel, (0, 0), 2.0)
        shade = ((rel - 0.5) * 2 * keep_structure)[..., None]
        scene = np.clip(scene + shade * 0.35, 0, 1)

    a = ((1 - fg) * float(np.clip(strength, 0, 1)))[..., None]
    out = np.clip(img * (1 - a) + scene * a, 0, 1).astype(np.float32)
    return out, {"preset": preset,
                 "label": dict(PRESETS)[preset],
                 "bg_frac": round(float((1 - fg).mean()), 4),
                 "keep_structure": keep_structure}


def info() -> dict:
    return {"presets": [{"key": k, "label": v} for k, v in PRESETS],
            "pipeline": "segmentation → procedural seasonal scene → "
                        "structure-preserving blend",
            "deterministic": True}
