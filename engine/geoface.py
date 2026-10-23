# -*- coding: utf-8 -*-
"""几何遮罩：用 5 点解算出五官区域，不依赖任何分割模型.

为什么需要这个
--------------
`2d106det.onnx` 实测输出的 106 个点**不是标准语义关键点**：按 insightface
索引取「嘴唇内侧」拿到的中心落在脸的上半部，「左眼」的 y 跨了半个脸高。
按索引取值等于在散点里乱抓，妆容和表情因此作用在错误位置。

而 YuNet 自带的 5 点（左右眼中心、鼻尖、左右嘴角）语义是确定的，实测
眼距 / 脸宽 = 0.46，符合人脸解剖比例。所以这里改用 5 点 + 解剖比例反推
各区域，得到的是**位置可信、边界近似**的遮罩。

精度边界（要诚实）
------------------
这是椭圆近似，不是分割。它能保证「唇彩落在嘴上、腮红落在颧骨」，
但精度不如真正的解析模型——唇线不会贴着真实唇形走，眼窝边界是椭圆。
在可用的分割模型缺失时，这是正确性（位置对）优于精度（边界准）的取舍。

所有尺寸都以**眼距**为单位表达，所以同一套系数在任何脸大小上都成立。
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

# YuNet 5 点顺序
L_EYE_R, L_EYE_L, L_NOSE, L_MOUTH_R, L_MOUTH_L = 0, 1, 2, 3, 4

# 各区域相对眼距的比例。数值经实测校准（眼距 = 1.0）：
#   眉毛 -0.42、鼻尖 +0.54、嘴唇 +0.95、瞳孔与 5 点差 1px
# 半径比初版收窄，否则相邻椭圆会粘连：眉毛下缘与眼睛上缘原本只隔 0.17
# 眼距，视觉上糊成一片，看着像"遮罩位置错了"。
SCALE = {
    "eye_w": 0.58,        # 单眼宽（横向）
    "eye_h": 0.26,        # 单眼高（纵向）
    "brow_w": 0.72,       # 眉宽
    "brow_h": 0.12,       # 眉高
    "brow_lift": 0.42,    # 眉心在眼线上方多少（实测值）
    "lip_w": 1.30,        # 唇宽
    "lip_h": 0.40,        # 唇高（含上下唇）
    "philtrum": 0.41,     # 鼻尖到嘴角中点（实测 0.41 眼距）
    "cheek_w": 0.62,      # 腮红横半径
    "cheek_h": 0.46,      # 腮红纵半径
    "cheek_out": 0.42,    # 腮红中心在眼外侧多少（解剖值 0.3~0.5）
    "cheek_down": 0.22,   # 腮红中心在眼线下方多少（解剖值 0.15~0.25）
}


def _iod(lm: np.ndarray) -> float:
    return float(max(np.linalg.norm(lm[L_EYE_L] - lm[L_EYE_R]), 1.0))


def _roll_angle(lm: np.ndarray) -> float:
    """两眼连线的倾角（弧度）。人物歪头时所有区域要跟着转。"""
    d = lm[L_EYE_L] - lm[L_EYE_R]
    return float(np.arctan2(d[1], d[0]))


def _ellipse_mask(shape: Tuple[int, int], centre: Tuple[float, float],
                  radii: Tuple[float, float], angle: float = 0.0,
                  softness: float = 2.0) -> np.ndarray:
    """旋转椭圆，边缘软过渡。

    用椭圆方程做距离场而不是画实心多边形：实心多边形的硬边在任何后续
    模糊里都会露出棱角，而柔光遮罩（腮红、眼影）需要中心到边缘连续衰减。
    """
    h, w = shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - centre[0]
    dy = yy - centre[1]
    ca, sa = np.cos(angle), np.sin(angle)
    xr = dx * ca + dy * sa
    yr = -dx * sa + dy * ca
    rx = max(float(radii[0]), 1.0)
    ry = max(float(radii[1]), 1.0)
    d = (xr / rx) ** 2 + (yr / ry) ** 2
    # 内部 1、边界 0，过渡宽度按 softness 比例
    m = np.clip((1.0 - d) * float(softness) + 0.5, 0.0, 1.0)
    return m.astype(np.float32)


def region_masks(img: np.ndarray, landmarks: np.ndarray,
                 softness: float = 2.0) -> Dict[str, np.ndarray]:
    """从 5 点反推各区域遮罩。

    返回的每个遮罩都是 float32、0..1、与图同尺寸，可直接喂给
    `portrait._tint` / `portrait._local_warp`。
    """
    lm = np.asarray(landmarks, np.float32)
    if lm.shape[0] < 5:
        return {}
    h, w = img.shape[:2]
    iod = _iod(lm)
    ang = _roll_angle(lm)

    eye_r, eye_l = lm[L_EYE_R], lm[L_EYE_L]
    nose = lm[L_NOSE]
    mouth_r, mouth_l = lm[L_MOUTH_R], lm[L_MOUTH_L]

    # 与两眼连线垂直的「向下」方向，用来在歪头时正确偏移
    down = np.array([-np.sin(ang), np.cos(ang)], np.float32)
    right = np.array([np.cos(ang), np.sin(ang)], np.float32)

    masks: Dict[str, np.ndarray] = {}

    # ---- 眼睛（左右各一个椭圆）----
    rx, ry = SCALE["eye_w"] * iod / 2, SCALE["eye_h"] * iod / 2
    for tag, c in (("eye_r", eye_r), ("eye_l", eye_l)):
        masks[tag] = _ellipse_mask((h, w), c, (rx, ry), ang, softness)

    # 双眼合并（眼影、眼睛增强用）
    masks["eyes"] = np.maximum(masks["eye_r"], masks["eye_l"])

    # ---- 眉毛：眼线向上偏移，横向略宽 ----
    lift = SCALE["brow_lift"] * iod
    brx = SCALE["brow_w"] * iod / 2
    bry = SCALE["brow_h"] * iod / 2
    for tag, c in (("brow_r", eye_r), ("brow_l", eye_l)):
        # 眉毛比眼睛更靠外、更靠上
        cc = c + down * (-lift) + right * ((-1 if tag == "brow_r" else 1) * iod * 0.06)
        masks[tag] = _ellipse_mask((h, w), cc, (brx, bry), ang, softness)
    masks["brows"] = np.maximum(masks["brow_r"], masks["brow_l"])

    # ---- 嘴唇：用两个嘴角的中点定位，宽度取实际嘴宽 ----
    mouth_c = (mouth_r + mouth_l) / 2.0
    lip_w = float(np.linalg.norm(mouth_l - mouth_r))
    lip_h = SCALE["lip_h"] * iod
    masks["lips"] = _ellipse_mask((h, w), mouth_c,
                                  (max(lip_w, iod * 0.8) / 2, lip_h / 2),
                                  ang, softness)

    # ---- 腮红：眼外角外侧偏下，两个 ----
    for tag, c, side in (("blush_r", eye_r, -1), ("blush_l", eye_l, 1)):
        cc = (c + right * (side * SCALE["cheek_out"] * iod)
              + down * (SCALE["cheek_down"] * iod))
        masks[tag] = _ellipse_mask(
            (h, w), cc,
            (SCALE["cheek_w"] * iod / 2, SCALE["cheek_h"] * iod / 2),
            ang, softness * 0.8)
    masks["blush"] = np.maximum(masks["blush_r"], masks["blush_l"])

    # ---- 鼻子（鼻尖周围，用于高光/阴影）----
    # 纵向半径必须小于「鼻尖到嘴角中点」的距离。实测该距离只有约 0.41 个
    # 眼距，原来取 0.40 的半径会让鼻遮罩下缘压到嘴唇上（重叠 0.23），
    # 两个区域同时上色就会互相污染。
    masks["nose"] = _ellipse_mask((h, w), nose,
                                  (iod * 0.30, iod * 0.26), ang, softness)

    # ---- 人脸椭圆（整脸范围，用于限制一切面部操作）----
    # 脸中心略高于嘴、略低于眼；尺寸按眼距估算
    face_c = (eye_r + eye_l) / 2.0 + down * (iod * 0.55)
    masks["face"] = _ellipse_mask((h, w), face_c,
                                  (iod * 1.55, iod * 1.95), ang, softness)

    return masks


def apply_rotation(lm: np.ndarray) -> bool:
    """两眼连线的倾角是否大到需要做旋转补偿（超过 8 度）。"""
    return abs(_roll_angle(np.asarray(lm, np.float32))) > np.radians(8.0)


def info() -> dict:
    return {"source": "5-point landmarks + anthropometric ratios",
            "regions": ["eye_r", "eye_l", "eyes", "brow_r", "brow_l", "brows",
                        "lips", "blush_r", "blush_l", "blush", "nose", "face"],
            "scale_unit": "interocular distance",
            "precision": "position reliable, boundary approximate (ellipse)"}
