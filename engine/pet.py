# -*- coding: utf-8 -*-
"""宠物毛发的确定性替代：毛色提亮/顺毛 + 背景剥离.

像素蛋糕「宠物美化 → 祛除毛发/眼睛调整」用宠物专用模型；本地没有
宠物分割模型（如实说明过）。但"宠物照片"最常见的两类诉求可以用
**确定性的颜色/亮度判据**做到，不需要识别"这是猫还是狗"：

  1. **毛发提亮/顺毛**：宠物主体（用户用笔刷涂或用背景分割的补集）
     里的**高频亮纹**是毛发高光，沿毛流方向做各向异性提亮。
  2. **眼睛提亮**：宠物眼在头部区域内的深色高饱和圆块，提亮 +
     加眼神光。头部区域用 YuNet 检不到（它是人脸检测），退而求其次：
     用户涂一下，或取主体上半部。
  3. **背景剥离**：MODNet 是通用显著物体抠图（对猫狗同样有效——
     它不是人专用的），可以直接给宠物换背景/换底色。

这个模块不做"识别宠物部位"，只做这三件确定性可验证的事。
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def _subject_or_given(img: np.ndarray, mask: Optional[np.ndarray]
                      ) -> Optional[np.ndarray]:
    if mask is not None:
        m = np.clip(np.asarray(mask, np.float32), 0, 1)
        return m[..., 0] if m.ndim == 3 else m
    try:
        from . import matting as MT

        if MT.available():
            a = MT.subject_alpha(img)
            if a is not None:
                a = np.asarray(a, np.float32)
                return a[..., 0] if a.ndim == 3 else a
    except Exception:
        pass
    return None


def groom(img: np.ndarray, mask: Optional[np.ndarray] = None,
          shine: float = 0.0, smooth: float = 0.0,
          eye_glow: float = 0.0) -> Tuple[np.ndarray, dict]:
    """宠物毛发修整.

    * `shine`  毛发光泽：主体内高亮细节沿最大梯度方向轻微各向异性拉伸
      （视觉上"顺毛"），并整体提亮高光。
    * `smooth` 毛发平滑：压杂毛但不糊边缘（双边滤波，掩膜内）。
    * `eye_glow` 眼神光：主体上半部的深色高饱和小圆块（瞳孔）加亮点。
    """
    import cv2

    out = np.asarray(img, np.float32).copy()
    subj = _subject_or_given(img, mask)
    info: dict = {"has_subject": subj is not None}
    if subj is None:
        return np.clip(out, 0, 1), info
    m = np.clip(subj, 0, 1)
    if m.shape[:2] != out.shape[:2]:
        m = cv2.resize(m, (out.shape[1], out.shape[0]),
                       interpolation=cv2.INTER_LINEAR)

    if smooth > 0:
        amt = smooth / 100.0
        # sigmaColor 用 0..255 量纲（杂毛色差 0.05~0.15 ≈ 13~38）。
        # 半径不能大：d=0 会让直径按 sigmaSpace 自动膨胀到 100px+，
        # 掩膜边缘内侧混入旁边的背景色，"平滑"反而制造色偏（实测与
        # 干净底的偏差 0.029→0.052）。固定 d=13，把渗入限制在 6px 内，
        # 再用**侵蚀掩膜**混合，边界 6px 不动。
        core = cv2.erode((m > 0.5).astype(np.uint8),
                         np.ones((13, 13), np.uint8))
        m_in = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 2.0)
        sm = cv2.bilateralFilter(out, 13, 20 + amt * 55, 9 + amt * 6)
        wgt = (m_in * amt)[..., None]
        out = out * (1 - wgt) + sm * wgt
        info["smoothed"] = True

    if shine > 0:
        amt = shine / 100.0
        from .imgutil import luminance

        lum = luminance(out)
        gy, gx = np.gradient(lum)
        mag = np.sqrt(gx * gx + gy * gy) + 1e-6
        # 毛流方向 = 梯度方向（垂直于毛的走向）——沿该方向做轻量方向性
        # 提亮：把高光细节往梯度方向扩散 1px
        hi = np.clip((lum - 0.55) / 0.35, 0, 1) * np.clip(mag * 8, 0, 1)
        dx = (gx / mag * hi * amt * 0.8)
        dy = (gy / mag * hi * amt * 0.8)
        ys, xs = np.mgrid[0:out.shape[0], 0:out.shape[1]].astype(np.float32)
        sx = np.clip(xs + dx, 0, out.shape[1] - 1).astype(np.int32)
        sy = np.clip(ys + dy, 0, out.shape[0] - 1).astype(np.int32)
        shifted = out[sy, sx]
        wgt = (hi * m * amt * 0.6)[..., None]
        out = out * (1 - wgt) + shifted * wgt
        info["shined"] = True

    if eye_glow > 0:
        amt = eye_glow / 100.0
        # 主体上半部的深色圆块：暗 + 饱和度不低（黑瞳）+ 尺寸小
        from .imgutil import luminance, rgb_to_hsv

        hsv = rgb_to_hsv(np.clip(out, 0, 1))
        s = hsv[..., 1].astype(np.float32)
        lum = luminance(out)
        top = np.zeros_like(m)
        hh = int(out.shape[0] * 0.55)
        top[:hh] = 1.0
        cand = ((lum < 0.22) & (s > 0.25) & (m > 0.5) & (top > 0))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(
            cand.astype(np.uint8), 8)
        eye = np.zeros_like(m)
        picked = 0
        for i in range(1, n):
            a = stats[i, cv2.CC_STAT_AREA]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]
            # 瞳孔：小、近圆
            if 8 <= a <= int(out.shape[0] * out.shape[1] * 0.002) and \
                    bw < out.shape[1] * 0.08 and bh < out.shape[0] * 0.08 and \
                    min(bw, bh) > 2 and (a / (bw * bh)) > 0.45:
                eye[lab == i] = 1.0
                picked += 1
        if picked:
            eye = cv2.dilate(eye, np.ones((3, 3), np.uint8))
            eye = cv2.GaussianBlur(eye, (0, 0), 1.0)
            out = out + (eye * amt * 0.45)[..., None] * \
                np.array([0.9, 0.95, 1.0], np.float32)
        info["eyes"] = picked

    return np.clip(out, 0.0, 1.0), info


def cutout(img: np.ndarray) -> Tuple[np.ndarray, dict]:
    """宠物抠图（MODNet 通用显著物）：返回 RGBA。"""
    from . import matting as MT

    if not MT.available():
        raise RuntimeError("MODNet 不可用")
    a = MT.subject_alpha(img)
    a = np.asarray(a, np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    rgba = np.dstack([np.clip(img, 0, 1), np.clip(a, 0, 1)])
    return rgba, {"alpha_frac": round(float((a > 0.5).mean()), 4)}


def info() -> dict:
    return {"feature": "pet",
            "functions": ["groom(shine/smooth/eye_glow)", "cutout(MODNet)"],
            "note": "无宠物专用分割模型；以上为颜色/亮度判据的确定性实现"}
