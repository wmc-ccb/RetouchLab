# -*- coding: utf-8 -*-
"""自动蒙版：一键把画面拆成可独立使用的区域蒙版.

参考软件（Photoshop 的「选择主体」+「选择并遮住」）能按人物部位出选区。
本项目已有两个分割模型，合起来正好覆盖"人脸/皮肤/衣服/背景/头发/配饰"
这几类：

  * `semantic`（MediaPipe selfie_multiclass，6 类）
      背景 / 头发 / 身体皮肤 / 脸部皮肤 / 衣物 / 配饰
  * `faceparse`（SegFormer-B2，19 类 -> 6 组）
      眉毛 / 眼睛 / 嘴唇 / 口腔 / 鼻子 / 面部皮肤

两者分工明确：semantic 负责**全身**（含背景与衣服），faceparse 负责
**五官细节**（它的输出只覆盖人脸框附近，做不了衣服/背景）。

输出约定：每个蒙版都是 float32、0..1、与原图同尺寸，可直接当 alpha
用于导出（PNG 透明）、局部调色、或作为其他功能的输入。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# 展示名与顺序（UI 按这个顺序出按钮）。
# 注意 semantic 的原始键是 `body-skin` / `face-skin`（连字符），对外统一
# 暴露成下划线，避免调用方记两套命名；映射在 _semantic_masks 里做。
SEMANTIC_PARTS = [
    ("background", "背景"),
    ("hair", "头发"),
    ("face_skin", "脸部皮肤"),
    ("body_skin", "身体皮肤"),
    ("clothes", "衣服"),
    ("accessories", "配饰"),
    ("subject", "人物整体"),
]
# 对外键 -> semantic 内部键
_SEM_KEY = {"face_skin": "face-skin", "body_skin": "body-skin"}
FACE_PARTS = [
    ("face", "面部皮肤"),
    ("brows", "眉毛"),
    ("eyes", "眼睛"),
    ("lips", "嘴唇"),
    ("mouth", "口腔"),
    ("nose", "鼻子"),
]


def _semantic_masks(img: np.ndarray) -> Dict[str, np.ndarray]:
    from . import semantic as SEM

    if not SEM.available():
        return {}
    reg = SEM.segment(img)
    if reg is None:
        return {}
    out = {}
    for key, _label in SEMANTIC_PARTS:
        try:
            if key == "subject":
                # subject 是方法（所有非背景类的并集），不是键
                m = reg.subject()
            else:
                m = reg.norm(_SEM_KEY.get(key, key))
        except Exception:
            m = None
        if m is not None and float(np.asarray(m).max()) > 0:
            out[key] = np.clip(np.asarray(m, np.float32), 0.0, 1.0)
    return out


def _face_masks(img: np.ndarray, face_box: Optional[tuple]
                ) -> Dict[str, np.ndarray]:
    """Face-detail masks from SegFormer.

    SegFormer is trained on 512 crops of *faces*, not whole frames: run on a
    full 900 px portrait the face occupies a few dozen pixels and every
    detail group (brows/mouth/nose) collapses to ~0 — measured 0.0001 for
    eyes and 0.0 for brows/mouth/nose. So the face is detected, cropped with
    context, parsed, and the result pasted back into a full-size mask.
    """
    from . import faceparse as FP

    if not FP.available():
        return {}

    h, w = img.shape[:2]
    box = face_box
    if box is None:
        try:
            from .face import FaceDetector

            faces = FaceDetector(dense=False).detect(img, max_faces=3)
            if faces:
                f = max(faces, key=lambda x: x.box[2] * x.box[3])
                box = tuple(float(v) for v in f.box)
        except Exception:
            box = None
    if box is None:
        return {}

    x, y, bw, bh = [float(v) for v in box]
    # 人脸框外扩 1.35 倍（带上下发际/下巴），确保五官都在裁剪内
    cx, cy = x + bw / 2, y + bh / 2
    half = max(bw, bh) * 1.35
    x0 = int(max(0, cx - half)); x1 = int(min(w, cx + half))
    y0 = int(max(0, cy - half)); y1 = int(min(h, cy + half))
    if x1 - x0 < 24 or y1 - y0 < 24:
        return {}
    crop = img[y0:y1, x0:x1]

    try:
        # 裁剪后传 face_box=None：模型自己按裁剪图整幅解析，脸部占比最大
        parts = FP.parse(crop, face_box=None)
    except Exception:
        return {}
    if not parts:
        return {}

    out: Dict[str, np.ndarray] = {}
    for key, _label in FACE_PARTS:
        try:
            m = parts[key]
        except Exception:
            m = None
        if m is None:
            continue
        m = np.asarray(m, np.float32)
        if m.max() <= 0:
            continue
        # 贴回全图尺寸
        import cv2

        full = np.zeros((h, w), np.float32)
        resized = cv2.resize(m, (x1 - x0, y1 - y0),
                             interpolation=cv2.INTER_LINEAR)
        full[y0:y1, x0:x1] = np.clip(resized, 0, 1)
        out[key] = full
    return out


def build(img: np.ndarray, face_box: Optional[tuple] = None,
          parts: Optional[List[str]] = None,
          feather: float = 1.0) -> Tuple[Dict[str, np.ndarray], dict]:
    """生成请求的区域蒙版.

    `parts` 为空表示全部；返回 dict 里只含实际可用的蒙版（模型缺失或
    没检出脸时会少几项），info 里如实记录哪些不可用及原因。
    """
    want = set(parts) if parts else None

    def keep(k: str) -> bool:
        return want is None or k in want

    masks: Dict[str, np.ndarray] = {}
    info: Dict[str, object] = {"requested": sorted(want) if want else "all",
                               "unavailable": {}}

    sem = _semantic_masks(img)
    for k, m in sem.items():
        if keep(k):
            masks[k] = m
    if not sem:
        info["unavailable"]["semantic"] = "模型不可用或无结果"

    if want is None or any(k in want for k, _ in FACE_PARTS):
        fp = _face_masks(img, face_box)
        for k, m in fp.items():
            if keep(k):
                masks[k] = m
        if not fp:
            info["unavailable"]["face"] = "未检出人脸或模型不可用"

    # 羽化：只对大区域做轻微高斯，避免硬边在合成时露锯齿。
    # 小区域（眼睛/嘴唇/鼻子，覆盖率 <1%）**不能羽化**——在全图尺度上
    # sigma=1 的高斯会把这些细小的蒙版稀释到接近 0（实测 lips 从 0.0035
    # 掉成 0.001），而它们恰恰需要最锐利的边缘。
    if feather and feather > 0:
        import cv2

        for k in list(masks):
            cov = float((masks[k] > 0.5).mean())
            if cov < 0.02:
                continue
            masks[k] = np.clip(
                cv2.GaussianBlur(masks[k], (0, 0), float(feather)), 0, 1)

    # 统计覆盖率，便于 UI 显示"这块区域多大"
    info["coverage"] = {k: round(float((m > 0.5).mean()), 4)
                        for k, m in masks.items()}
    info["available"] = sorted(masks.keys())
    return masks, info


def colourise(masks: Dict[str, np.ndarray], img: np.ndarray,
              alpha: float = 0.55) -> np.ndarray:
    """把所有蒙版叠成一张彩色预览图（每个区域一个颜色）."""
    palette = {
        "background": (0.25, 0.30, 0.40),
        "hair": (1.00, 0.75, 0.20),
        "face_skin": (1.00, 0.45, 0.45),
        "face": (1.00, 0.55, 0.55),
        "body_skin": (0.95, 0.70, 0.55),
        "clothes": (0.35, 0.65, 1.00),
        "accessories": (0.70, 0.40, 1.00),
        "brows": (0.55, 0.35, 0.20),
        "eyes": (0.25, 0.85, 0.85),
        "lips": (1.00, 0.25, 0.55),
        "mouth": (0.75, 0.15, 0.35),
        "nose": (0.55, 0.90, 0.45),
        "subject": (0.40, 1.00, 0.70),
    }
    out = np.asarray(img, np.float32).copy()
    for k, m in masks.items():
        if k == "subject":
            continue          # 人物整体会把所有细节盖住，预览时跳过
        c = np.array(palette.get(k, (1, 1, 1)), np.float32)
        a = (np.clip(m, 0, 1) * float(alpha))[..., None]
        out = out * (1 - a) + c.reshape(1, 1, 3) * a
    return np.clip(out, 0, 1)


def to_alpha(mask: np.ndarray, img: np.ndarray) -> np.ndarray:
    """把蒙版转成可直接写 PNG 的 RGBA（背景透明）."""
    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    rgba = np.zeros((img.shape[0], img.shape[1], 4), np.float32)
    rgba[..., :3] = np.asarray(img, np.float32)[..., :3]
    rgba[..., 3] = m
    return rgba


def info() -> dict:
    from . import faceparse as FP
    from . import semantic as SEM

    return {"semantic": SEM.available(), "faceparse": FP.available(),
            "parts": [{"key": k, "label": v} for k, v in
                      SEMANTIC_PARTS + FACE_PARTS],
            "pipeline": "selfie_multiclass (body) + SegFormer (face detail)"}
