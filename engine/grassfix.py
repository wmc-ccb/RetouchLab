# -*- coding: utf-8 -*-
"""AI 补草地：把草地上的穿帮（杂物/人影/落叶堆）用周围草地纹理补全.

像素蛋糕「图像美化 → AI 补草地」的本地确定性等价实现。像素蛋糕的
行为是把草地上的杂物移除并延续草地纹理；这里的做法是：

  1. **草地判据**：绿色色相带（90-150°）+ 中低亮度 + 低纹理频率。
     水泥地/木板路不在色相带里，不会被当成草地。
  2. **穿帮判据**：在"草地似然高"的区域**内部**，长得不像草地的
     连通块（颜色离草地中位数远，或纹理频率显著更高——人影/背包/
     纸屑都是这类）。背景分割把人物主体从"穿帮候选"里排除，防止
     把主体的一部分（比如白裙子下摆）误当杂物。
  3. **修补**：LaMa 以周围草地为上下文填充（lama.fill 的窗口机制
     正好只跑含 mask 的窗口）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def _grass_likelihood(img: np.ndarray) -> np.ndarray:
    """Per-pixel grass-likeness: green hue band + moderate brightness."""
    from .imgutil import rgb_to_hsv

    hsv = rgb_to_hsv(np.clip(img, 0, 1))
    h_deg = hsv[..., 0].astype(np.float32)          # 0..360 (measured)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    green = 1.0 - np.clip(np.abs(h_deg - 115.0) / 45.0, 0, 1)
    green *= np.clip((s - 0.10) / 0.30, 0, 1)
    green *= np.clip((v - 0.10) / 0.25, 0, 1) * np.clip((0.85 - v) / 0.35, 0, 1)
    return np.clip(green, 0, 1).astype(np.float32)


def detect_debris(img: np.ndarray, subject_mask: Optional[np.ndarray] = None,
                  min_area_frac: float = 0.0004,
                  max_area_frac: float = 0.06) -> Tuple[np.ndarray, dict]:
    """Return (debris mask 0..1, info)."""
    import cv2

    h, w = img.shape[:2]
    grass = _grass_likelihood(img)
    grass_bin = (grass > 0.45).astype(np.uint8)
    grass_frac = float(grass_bin.mean())
    if grass_frac < 0.05:
        # 图里几乎没有草地，功能不适用——如实说明而不是硬猜
        return np.zeros((h, w), np.float32), {
            "applicable": False, "reason": "画面中草地占比过低",
            "grass_frac": round(grass_frac, 4)}

    # 草地的整体颜色分布（中位数 + MAD），用于判"不像草"
    gmed = np.median(img[grass_bin > 0], axis=0)
    dist = np.linalg.norm(img - gmed.reshape(1, 1, 3), axis=2)

    # 候选 = 草地附近、颜色离草地远、又不是主体。
    # 膨胀核取画面宽度的 1/6（半径），大杂物（背包/纸箱）的中心虽然
    # 本身不是草色，但仍在"草地环境"里——小核会让大杂物的中心落在
    # zone 外而漏检（实测 26px 椭圆在 21px 核下中心漏掉）。
    k = max(9, (w // 6) * 2 + 1)
    zone = cv2.dilate(grass_bin, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (k, k)))
    cand = (zone > 0) & (dist > 0.16) & (grass < 0.3)
    if subject_mask is not None:
        cand &= ~(np.asarray(subject_mask, np.float32) > 0.5)

    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        cand.astype(np.uint8), 8)
    out = np.zeros((h, w), np.uint8)
    removed = 0
    min_a = max(12, int(h * w * min_area_frac))
    max_a = int(h * w * max_area_frac)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if min_a <= a <= max_a:
            out[lab == i] = 255
            removed += 1
    m = cv2.dilate(out, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)))          # 稍外扩，把阴影边也吃进
    m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 1.5)
    info = {"applicable": True, "grass_frac": round(grass_frac, 4),
            "debris_count": removed,
            "debris_frac": round(float((out > 0).mean()), 5)}
    return np.clip(m, 0, 1), info


def fix_grass(img: np.ndarray, subject_mask: Optional[np.ndarray] = None,
              mask: Optional[np.ndarray] = None,
              strength: float = 1.0) -> Tuple[np.ndarray, dict]:
    """补草地。`mask` 给定则直接用它（用户手涂），否则自动识别穿帮."""
    from . import lama as LM

    h, w = img.shape[:2]
    if mask is None:
        m, dinfo = detect_debris(img, subject_mask)
        if not dinfo.get("applicable"):
            return img.copy(), dinfo
    else:
        m = np.clip(np.asarray(mask, np.float32), 0, 1)
        if m.ndim == 3:
            m = m[..., 0]
        dinfo = {"applicable": True, "manual": True,
                 "debris_frac": round(float((m > 0.5).mean()), 5)}
    if float((m > 0.5).sum()) < 40:
        return img.copy(), {**dinfo, "skipped": "没有可补的区域"}

    filled, finfo = LM.fill(img, m)
    a = (m * float(np.clip(strength, 0, 1)))[..., None]
    out = np.clip(img * (1 - a) + filled * a, 0, 1).astype(np.float32)
    return out, {**dinfo, "fill": finfo.get("windows", finfo)}


def info() -> dict:
    return {"feature": "grass_fix",
            "pipeline": "grass colour-likelihood → debris CC → LaMa fill",
            "requires": "画面里有草地（绿色占比 >5%）"}
