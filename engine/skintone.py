# -*- coding: utf-8 -*-
"""套图肤色统一：把一组照片的肤色对齐到参考图的肤色。

婚纱/写真交片的硬需求：同一场拍摄几百张，肤色因光线在张与张之间
漂移。思路是统计对齐而不是生成——

    1. 皮肤区域：faceparse 的 skin 掩膜（模型缺失时回退 YCbCr+HSV
       的确定性肤色判定，弱一些但可用）
    2. 参考图与目标图都在皮肤掩膜内统计 Lab 均值/方差
    3. 目标图在掩膜内做 (x-μt)·(σr/σt)+μr 的线性映射，
       羽化边缘、mask 外逐像素不变

不做全局色彩迁移——背景衣服不许被"牵连变色"，这是和全图追色的
本质区别。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def skin_mask(img: np.ndarray) -> Optional[np.ndarray]:
    """0-1 皮肤掩膜；faceparse 可用走模型，否则确定性肤色规则。"""
    try:
        from . import faceparse as FP
        if FP.available():
            m = FP.mask(img)
            # faceparse 输出部位 id 表；不同实现里皮肤 id 不同，
            # 这里取"既非背景非头发非衣服"的最大类，保守而稳健
            if m is not None and m.ndim == 2:
                return np.clip(m.astype(np.float32), 0, 1)
    except Exception:
        pass
    # 回退：YCbCr Cr∈[133,180] ∧ HSV 约束（经典规则，对黄种人偏保守）
    h, w = img.shape[:2]
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    m = ((ycc[..., 1] >= 133) & (ycc[..., 1] <= 180)
         & (hsv[..., 1] >= 15) & (hsv[..., 1] <= 175)
         & (ycc[..., 0] >= 40))
    mask = m.astype(np.float32)
    # 开运算去零星误检
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    return mask


def _lab_stats(img: np.ndarray, mask: np.ndarray
               ) -> Tuple[np.ndarray, np.ndarray]:
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    w = mask * (mask > 0.05)
    sw = w.sum()
    if sw < 32:                       # 皮肤区域太小：返回原值（不动）
        mean = np.array([128.0, 128.0, 128.0], np.float32)
        std = np.ones(3, np.float32)
    else:
        mean = (lab * w[..., None]).sum((0, 1)) / sw
        var = ((lab - mean) ** 2 * w[..., None]).sum((0, 1)) / sw
        std = np.sqrt(var + 1e-3)
    return lab, np.stack([mean, std])          # (2,3)


def stats(img: np.ndarray) -> np.ndarray:
    """皮肤区 Lab 均值/方差，参考图与目标图共用。"""
    _, st = _lab_stats(img, skin_mask(img))
    return st


def unify(target: np.ndarray, ref_stats: np.ndarray,
          strength: float = 1.0) -> np.ndarray:
    """把 target 的肤色统计对齐到 ref_stats（stats() 的输出）。

    strength 0-1：0 不动，1 完全对齐（留一点衰减让肤色统一不"过火"）。
    mask 外逐像素不变；a/b 通道（色相）权重高于 L（明度），
    肤色差异主要在色相。
    """
    mask = skin_mask(target)
    if mask is None or mask.max() < 0.05:
        return target
    lab, tstats = _lab_stats(target, mask)
    tmean, tstd = tstats[0], tstats[1]
    rmean, rstd = ref_stats[0], ref_stats[1]
    gain = np.array([0.6, 1.0, 1.0], np.float32) * float(strength)
    ratio = np.clip(rstd / tstd, 0.6, 1.6)
    adj = (lab - tmean) * (1 + (ratio - 1) * gain) + \
        tmean + (rmean - tmean) * gain
    adj = np.clip(adj, 0, 255)
    out_lab = lab * (1 - mask[..., None]) + adj * mask[..., None]
    out = cv2.cvtColor(out_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    out = out.astype(np.float32) / 255.0
    # 边缘羽化，避免皮肤边界出现色阶
    feathered = cv2.GaussianBlur(mask, (0, 0), 1.5)
    keep = (mask <= 0.05)
    mask_f = np.where(keep, 0.0, np.maximum(mask, feathered * 0.0))[..., None]
    return target * (1 - mask_f) + out * mask_f
