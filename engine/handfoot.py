# -*- coding: utf-8 -*-
"""手足美化：手部检测 → 皮肤管线复用；脚部/鞋子 → 姿态关节点定区域。

婚纱/写真交付里手是第二张脸——捧花、牵手、托腮特写让手部皮肤问题
（干燥纹路、色沉、青筋）在 100% 放大下无处可藏。做法与脸部一致：
    掩膜内（手/脚/鞋）→ 复用磨皮与均匀肤色的既有管线；
    掩膜外 → 逐像素不变。

检测分工：
- 手：MediaPipe HandLandmarker（21 关节点，float16 模型 7.5MB，
  models/hand_landmarker.task，与 pose 同一套 tasks API）
- 脚：姿态关节点（ANKLE/HEEL/FOOT_INDEX 已有）——脚没有独立模型，
  用踝-踵-脚尖三角构建区域
- 鞋：脚区域向下延伸的包围盒（穿鞋时皮肤检测不到，纯几何区域）

土地里没有手/脚时（半身照、手在背后）各函数安静返回原图。
"""
from __future__ import annotations

import os
import threading
from typing import List, Optional

import cv2
import numpy as np

from .paths import models_dir

MODEL_NAME = "hand_landmarker.task"
_LANDMARKS = 21

_lock = threading.RLock()   # 可重入：detect_hands 持锁调 _detector()，后者也拿同一把
_DETECTOR = None          # 惰性单例（模型 7.5MB，首次加载 ~1s）


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


def available() -> bool:
    return os.path.exists(model_path())


def _detector():
    """HandLandmarker 单例。mediapipe tasks 的实例非线程安全，加锁。"""
    global _DETECTOR
    with _lock:
        if _DETECTOR is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            opts = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=model_path()),
                num_hands=2,
                min_hand_detection_confidence=0.4)
            _DETECTOR = vision.HandLandmarker.create_from_options(opts)
        return _DETECTOR


def _to_np(result) -> List[np.ndarray]:
    """MediaPipe 结果 → 每只手 (21, 2) 像素坐标列表。"""
    out = []
    info = getattr(result, "hand_landmarks", None) or []
    for lm in info:
        pts = np.array([[p.x, p.y] for p in lm], np.float32)
        out.append(pts)
    return out


def detect_hands(img: np.ndarray) -> List[np.ndarray]:
    """检测手，返回像素坐标关节点列表（空图 = 没手）。

    历史教训：这里曾有 `except Exception: return []` 包住整段——
    `mp` 未导入的 NameError 被一起吞掉，400+ 张有手的照片全部静默
    返回空。异常要么不捕（让上层看到），要么记录后重抛。
    """
    if not available():
        return []
    import mediapipe as mp
    h, w = img.shape[:2]
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.ascontiguousarray(u8)              # 引擎全链是 RGB 语义
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _lock:
        result = _detector().detect(mp_image)
    hands = []
    for pts in _to_np(result):
        pix = pts.copy()
        pix[:, 0] *= w
        pix[:, 1] *= h
        hands.append(pix)
    return hands


def hand_mask(img: np.ndarray, hands: List[np.ndarray],
              pad: float = 0.06) -> np.ndarray:
    """关节点 → 手部软掩膜（凸包外扩 + 羽化）。

    pad：凸包外扩比例（手腕肉比骨架宽）。
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for pts in hands:
        hull = cv2.convexHull(pts.astype(np.float32))
        poly = hull.reshape(-1, 2).astype(np.float32)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        poly = np.stack([cx + (poly[:, 0] - cx) * (1 + pad),
                         cy + (poly[:, 1] - cy) * (1 + pad)], axis=-1)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0)
    if mask.any():
        mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, w / 300.0))
    return np.clip(mask, 0, 1)


def foot_regions(img: np.ndarray, pose_pts: np.ndarray,
                 pad: float = 0.35) -> List[tuple]:
    """姿态关节点 → 脚/鞋的包围盒列表（含脚踝向下延伸）。

    pose_pts: (33, 3) 像素坐标（可见性在第三列）。
    穿鞋时脚踝以下不再是皮肤而是鞋——同一几何区域两个用途：
    美化用上半（脚背露肤），祛瑕用全盒。
    """
    h, w = img.shape[:2]
    boxes = []
    for ankle_i, heel_i, foot_i in ((27, 29, 31), (28, 30, 32)):
        a = pose_pts[ankle_i]
        heel = pose_pts[heel_i]
        foot = pose_pts[foot_i]
        if a[2] < 0.4:            # 不可见
            continue
        pts = np.stack([a[:2], heel[:2], foot[:2]])
        x0 = max(0, int(pts[:, 0].min() - w * pad * 0.3))
        x1 = min(w, int(pts[:, 0].max() + w * pad * 0.3))
        y0 = max(0, int(a[1] - h * 0.02))
        y1 = min(h, int(pts[:, 1].max() + h * 0.06))
        if x1 - x0 > 6 and y1 - y0 > 6:
            boxes.append((x0, y0, x1, y1))
    return boxes


# --------------------------------------------------------------------------
# 美化管线：全部走 engine.skin 的既有函数，只是掩膜不同
# --------------------------------------------------------------------------

def beautify_skin_region(img: np.ndarray, mask: np.ndarray,
                         smooth: float, tone: float) -> np.ndarray:
    """掩膜内做皮肤平滑 + 均匀肤色（复用 skin.py 的既有管线）。"""
    from . import skin as SK

    out = img
    if smooth > 0:
        out = SK.smooth_skin(out, mask, smooth, texture=60.0)
    if tone > 0:
        out = SK.even_skin_tone(out, mask, tone)
    return out


def beautify_hands(img: np.ndarray, smooth: float = 0.0,
                   tone: float = 0.0, strength: float = 1.0
                   ) -> tuple:
    """手部美化。返回 (图, 手数)。"""
    hands = detect_hands(img)
    if not hands:
        return img, 0
    mask = hand_mask(img, hands)
    m = np.clip(mask * float(strength), 0, 1)
    out = beautify_skin_region(img, m, smooth, tone)
    return out, len(hands)


def beautify_feet(img: np.ndarray, pose_pts: np.ndarray,
                  smooth: float = 0.0, tone: float = 0.0,
                  strength: float = 1.0) -> tuple:
    """脚部美化（脚踝以下露肤区域）。返回 (图, 区域数)。"""
    h, w = img.shape[:2]
    regions = foot_regions(img, pose_pts)
    if not regions:
        return img, 0
    mask = np.zeros((h, w), np.float32)
    for x0, y0, x1, y1 in regions:
        mask[y0:y1, x0:x1] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), max(2.0, w / 400.0))
    m = np.clip(mask * float(strength), 0, 1)
    out = beautify_skin_region(img, m, smooth, tone)
    return out, len(regions)


def clean_shoes(img: np.ndarray, pose_pts: np.ndarray,
                strength: float = 0.6) -> tuple:
    """鞋子祛瑕疵：鞋面区域内找小暗斑/亮斑 → 中值差分检测 → 修复。

    strength 控制斑点检测的灵敏度（阈值越低越激进）。
    返回 (图, 修复斑点数)。
    """
    h, w = img.shape[:2]
    regions = foot_regions(img, pose_pts, pad=0.5)
    if not regions:
        return img, 0
    total = 0
    out = img.copy()
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    for x0, y0, x1, y1 in regions:
        roi = u8[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        med = cv2.medianBlur(gray, 7)
        diff = cv2.absdiff(gray, med)
        thr = max(12.0, float(diff.std()) * 2.2) * (1.6 - strength * 0.6)
        spots = (diff > thr).astype(np.uint8)
        spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN,
                                 np.ones((2, 2), np.uint8))
        n = int(spots.sum())
        if n < 4:
            continue
        total += n
        patch = cv2.inpaint(roi, spots * 255, 3, cv2.INPAINT_TELEA)
        # 只把斑点像素写回，避免整块区域被重绘
        m = cv2.GaussianBlur(spots.astype(np.float32), (0, 0), 0.8)
        roi_f = roi.astype(np.float32)
        patch_f = patch.astype(np.float32)
        blend = roi_f * (1 - m[..., None]) + patch_f * m[..., None]
        out[y0:y1, x0:x1] = blend / 255.0
    return out, total
