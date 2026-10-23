# -*- coding: utf-8 -*-
"""闭眼检测：选片刚需，用于图库「闭眼」分组.

连拍几十张里总有几张闭眼，人工逐张看很费时间。参考软件的图库分组里
有「闭眼」这一类。

判据用**眼睛纵横比（EAR）**的变体。标准 EAR 需要每只眼的 6 个轮廓点
（上/下/左/右/左上/右上），YuNet 只给 5 点（眼睛中心、鼻尖、两嘴角），
没有眼皮轮廓。所以这里用**可测量的替代量**：

  1. 眼睛区域的**暗像素占比**：睁眼时瞳孔暴露，暗像素多；闭眼时眼皮
     覆盖，暗像素显著减少。在 5 点给出的眼中心周围取椭圆区域统计。
  2. 眼睛区域的**纵向梯度能量**：睁眼时有明显的上/下眼睑边缘，闭眼时
     该区域趋于平滑。

两项都低于同批次分布的下界 -> 判为闭眼。阈值取**这批照片自身的分位数**
（和抖动/欠曝一样），因为不同人、不同妆、不同光线的"正常眼睛"差异很大，
固定阈值会整批误判。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

L_EYE_R, L_EYE_L = 0, 1


def _eye_stats(img: np.ndarray, centre, iod: float) -> Tuple[float, float]:
    """Return (dark ratio, vertical gradient energy) around one eye."""
    import cv2

    from .imgutil import luminance

    h, w = img.shape[:2]
    rx = max(3, int(iod * 0.19))
    ry = max(2, int(iod * 0.12))
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0 = max(0, cx - rx); x1 = min(w, cx + rx + 1)
    y0 = max(0, cy - ry); y1 = min(h, cy + ry + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return 0.0, 0.0
    patch = luminance(img[y0:y1, x0:x1])
    # 椭圆掩膜，去掉眼角外的眉/皮肤
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]].astype(np.float32)
    m = (((xx - (patch.shape[1] - 1) / 2) / rx) ** 2 +
         ((yy - (patch.shape[0] - 1) / 2) / ry) ** 2) <= 1.0
    if m.sum() < 6:
        return 0.0, 0.0
    px = patch[m]
    # 暗像素占比：相对该 patch 的亮度分布
    lo = float(np.percentile(px, 20))
    dark = float((px <= lo + 0.02).mean())
    g = (np.clip(patch, 0, 1) * 255).astype(np.uint8)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    energy = float(np.abs(gy)[m].mean()) / 255.0
    return dark, energy


def eye_scores(img: np.ndarray, landmarks: np.ndarray) -> Optional[dict]:
    """Both eyes' openness indicators. `landmarks` is YuNet's 5 points."""
    lm = np.asarray(landmarks, np.float32)
    if lm.shape[0] < 5:
        return None
    iod = float(np.linalg.norm(lm[L_EYE_L] - lm[L_EYE_R]))
    if iod < 6:
        return None
    out = {}
    for tag, i in (("r", L_EYE_R), ("l", L_EYE_L)):
        dark, energy = _eye_stats(img, lm[i], iod)
        out["dark_" + tag] = dark
        out["energy_" + tag] = energy
    # 闭眼分：暗像素少 + 纵向边缘弱 => 分数低
    out["open"] = float((out["dark_r"] + out["dark_l"]) / 2)
    out["edge"] = float((out["energy_r"] + out["energy_l"]) / 2)
    return out


def detect_closed(images: List[np.ndarray], landmarks_list: List[Optional[np.ndarray]]
                  ) -> Dict[str, list]:
    """Classify which frames look like closed eyes, using batch percentiles.

    Returns {"closed": [idx...], "scores": [...]}.
    """
    n = len(images)
    res = {"closed": [], "scores": []}
    if n == 0:
        return res
    rows = []
    for i in range(n):
        lm = landmarks_list[i] if i < len(landmarks_list) else None
        if lm is None:
            rows.append(None)
            continue
        s = eye_scores(images[i], lm)
        rows.append(s)
        res["scores"].append(s)

    valid = [(i, r) for i, r in enumerate(rows) if r is not None]
    if len(valid) < 3:
        return res
    open_vals = np.array([r["open"] for _, r in valid], np.float32)
    edge_vals = np.array([r["edge"] for _, r in valid], np.float32)
    # 闭眼：两项都低于本批次 20 分位
    o_thr = float(np.percentile(open_vals, 20))
    e_thr = float(np.percentile(edge_vals, 20))
    for idx, (i, r) in enumerate(valid):
        if r["open"] <= o_thr and r["edge"] <= e_thr:
            res["closed"].append(i)
    return res


def info() -> dict:
    return {"feature": "closed_eye",
            "proxies": ["dark ratio in eye ellipse", "vertical gradient"],
            "thresholds": "20th percentile of the batch"}
