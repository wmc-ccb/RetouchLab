# -*- coding: utf-8 -*-
"""沙滩美化：旅拍海景的确定性清理管线。

三步，全部局部于"沙面"掩膜（掩膜外逐像素不变）：

    1. 沙面识别 —— HSV 先验（暖低饱和亮色）+ 画面下半权重 +
       最大连通域。海面/天空/人物都被排除。
    2. 祛杂物 —— 沙面上与中值滤波差异显著的小斑块（烟头、瓶盖、
       石子、脚印边缘）→ 连通域过滤 → inpaint 修复。
    3. 沙面平整 —— 掩膜内保边平滑（沙纹保留大部分，细碎凹凸压平）。

与「祛杂物」（AI 布景面板）的分工：那个面向任意物体手涂/点选；
这个是场景预设，一键处理整片沙面，不用用户逐个点。
"""
from __future__ import annotations

import cv2
import numpy as np


def sand_mask(img: np.ndarray) -> np.ndarray:
    """沙面软掩膜（0-1）。暖色低饱和亮区 + 下半帧权重 + 最大连通域。"""
    h, w = img.shape[:2]
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    # 沙色的 HSV 带宽（H 8-40 度域、中低饱和、偏亮）
    m = ((hsv[..., 0] >= 4) & (hsv[..., 0] <= 42)
         & (hsv[..., 1] >= 8) & (hsv[..., 1] <= 130)
         & (hsv[..., 2] >= 90)).astype(np.float32)
    # 下半帧权重（沙滩通常在下面）；顶部 1/4 完全排除（天空/海）
    rows = np.arange(h, dtype=np.float32)[:, None]
    weight = np.clip((rows - h * 0.25) / (h * 0.35), 0, 1)
    m = m * weight
    # 最大连通域：只保留最大的那片沙（去掉木栈道等零散误检）
    u8m = (m > 0.5).astype(np.uint8)
    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(u8m, 8)
    if ncc > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        keep = (labels == biggest).astype(np.float32)
        m = m * keep
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    m = cv2.GaussianBlur(m, (0, 0), max(2.0, w / 350.0))
    return np.clip(m, 0, 1)


def remove_debris(img: np.ndarray, mask: np.ndarray,
                  strength: float = 0.7) -> tuple:
    """沙面祛杂物：中值差分找斑块 → 连通域过滤（只清小的）→ inpaint。

    strength 0-1：越高阈值越低、清得越多。
    返回 (图, 清除斑块数)。
    """
    h, w = img.shape[:2]
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    med = cv2.medianBlur(gray, 9)
    diff = cv2.absdiff(gray, med)
    thr = max(10.0, float(diff[mask > 0.5].std() if (mask > 0.5).any()
                         else diff.std()) * 1.8) * (1.5 - strength * 0.5)
    spots = ((diff > thr) & (mask > 0.5)).astype(np.uint8)
    # 连通域过滤：斑块面积限制在"杂物"尺度（太大的是阴影/物体，不清）
    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(spots, 8)
    keep = np.zeros_like(spots)
    count = 0
    for i in range(1, ncc):
        area = stats[i, cv2.CC_STAT_AREA]
        if 3 <= area <= max(120, int(w * h * 0.00025)):
            keep[labels == i] = 255
            count += 1
    if count == 0:
        return img, 0
    patch = cv2.inpaint(u8, keep, 4, cv2.INPAINT_TELEA)
    patch = patch.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(keep.astype(np.float32) / 255.0, (0, 0), 1.0)
    out = img * (1 - m[..., None]) + patch * m[..., None]
    return out, count


def smooth_sand(img: np.ndarray, mask: np.ndarray,
                amount: float = 0.5) -> np.ndarray:
    """沙面平整：掩膜内保边平滑，压细碎凹凸、留大纹理起伏。"""
    if amount <= 0:
        return img
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    smooth = cv2.bilateralFilter(u8, 11, 30, 11)
    down = cv2.resize(smooth, (max(1, u8.shape[1] // 4),
                               max(1, u8.shape[0] // 4)),
                      interpolation=cv2.INTER_AREA)
    smooth = cv2.resize(down, (u8.shape[1], u8.shape[0]),
                        interpolation=cv2.INTER_CUBIC)   # 只留大起伏
    a = float(np.clip(amount, 0, 1))
    mixed = (u8.astype(np.float32) * (1 - a * 0.7)
             + smooth.astype(np.float32) * (a * 0.7))
    m = cv2.GaussianBlur(mask, (0, 0), 2.0)[..., None]
    out = img * (1 - m) + (mixed / 255.0) * m
    return out


def beautify(img: np.ndarray, debris: float = 0.7,
             flatten: float = 0.5, warm: float = 0.0) -> tuple:
    """一键沙滩美化。返回 (图, 信息 dict)。"""
    mask = sand_mask(img)
    cov = float((mask > 0.5).mean())
    info = {"sand_coverage": round(cov, 3), "debris": 0}
    if cov < 0.02:
        info["note"] = "未识别到大片沙面"
        return img, info
    out, n = remove_debris(img, mask, debris)
    info["debris"] = n
    out = smooth_sand(out, mask, flatten)
    if warm > 0:
        # 轻微暖化沙面（日落感），只在掩膜内
        warm_a = np.array([0.04, 0.015, -0.03], np.float32) * warm
        out = np.clip(out + warm_a * mask[..., None], 0, 1)
    return out, info
