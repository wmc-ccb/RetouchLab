# -*- coding: utf-8 -*-
"""点击式选物：一个种子点 → 物体掩膜（智能消除的"点一下"路径）。

cv2.grabCut 以点击点为前景种子、外扩矩形为混合边界，适合"点一下
路灯/杂物/路人"这类目标与背景对比尚可的场景；失败模式（点了低对
比区域）返回空掩膜，由调用方提示用户手涂。
"""
from __future__ import annotations

import cv2
import numpy as np


def pick(img: np.ndarray, x: int, y: int,
         grow: float = 0.16) -> np.ndarray:
    """以 (x, y) 为种子的物体掩膜（0-1 float32，同图尺寸）。

    grow：种子矩形相对图宽的外扩比例——框住物体的松弛边界，
    grabCut 在框内做分割；种子本身标记为确定前景。
    返回空图表示该点无法可靠分割（覆盖率异常小/大）。
    """
    h, w = img.shape[:2]
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), np.float32)
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)

    pad = int(max(w, h) * grow)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + pad), min(h, y + pad)
    rect = (x0, y0, max(8, x1 - x0), max(8, y1 - y0))

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    mask = np.zeros((h, w), np.uint8)            # 全 GC_BGD
    # 框内先给 GC_PR_FGD（可能前景），框外保持背景
    mask[y0:y1, x0:x1] = cv2.GC_PR_FGD
    # 种子点小邻域 = 确定前景，锚住用户意图
    r = max(3, int(min(w, h) * 0.012))
    cx0, cy0 = max(0, x - r), max(0, y - r)
    cx1, cy1 = min(w, x + r + 1), min(h, y + r + 1)
    mask[cy0:cy1, cx0:cx1] = cv2.GC_FGD

    try:
        cv2.grabCut(bgr, mask, rect, bg_model, fg_model,
                    3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return np.zeros((h, w), np.float32)

    seg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                   1.0, 0.0).astype(np.float32)
    cov = seg.mean()
    if cov < 0.002 or cov > 0.6:
        # 覆盖率异常：种子点周围对比不足，grabCut 跑飞了
        return np.zeros((h, w), np.float32)
    # 收紧 + 羽化，让消除接缝更自然
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    seg = cv2.GaussianBlur(seg, (0, 0), 1.2)
    return seg
