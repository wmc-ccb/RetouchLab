# -*- coding: utf-8 -*-
"""生成式三件套：AI 布景 / 创成式填充 / 智能道具修复.

三者共用 genfill.fill() 的扩散 inpainting 管线，差别只在 mask 来源与
prompt：

  * **AI 布景**——mask = 主体之外的整片背景（semantic subject 的补集，
    同四季背景的保护方式），prompt 是用户对新场景的描述。与四季背景的
    程序化场景不同，这里是模型真正"画"出新背景。
  * **创成式填充**——用户手涂 mask + 描述想要什么（PS 生成式填充同款
    交互）。
  * **智能道具修复**——自动识别灯架/柔光箱/三脚架（复用 rigfix 的
    形状判据），prompt 让模型"继续周围环境"，即把道具抹掉。识别不到
    时回落到手涂 mask。

全部走同一保证：mask 外逐像素不变。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

NEG = "blurry, artifacts, watermark, text, deformed, lowres, person, face"


def _subject_mask(img: np.ndarray) -> Optional[np.ndarray]:
    try:
        from . import semantic as SEM

        reg = SEM.segment(img)
        return np.clip(reg.subject(), 0, 1) if reg is not None else None
    except Exception:
        return None


def _feather_subject(fg: np.ndarray) -> np.ndarray:
    """前景保护：膨胀 3px + 软边（防发丝处被背景色包边）。"""
    import cv2

    u8 = (fg > 0.5).astype(np.uint8)
    u8 = cv2.dilate(u8, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)))
    return np.clip(cv2.GaussianBlur(u8.astype(np.float32), (0, 0), 1.2),
                   0, 1)


def ai_backdrop(img: np.ndarray, prompt: str,
                subject_mask: Optional[np.ndarray] = None,
                strength: float = 1.0,
                steps: int = 20, seed: Optional[int] = None,
                model: str = 'sdxl') -> Tuple[np.ndarray, Dict]:
    """AI 布景：把主体背后的整片背景换成 prompt 描述的场景."""
    from . import genfill as GF

    h, w = img.shape[:2]
    subj = subject_mask
    if subj is None:
        subj = _subject_mask(img)
    if subj is None:
        raise RuntimeError("主体分割不可用，无法安全换布景")
    fg = _feather_subject(np.asarray(subj, np.float32))
    bg = 1.0 - fg
    if float((bg > 0.5).mean()) < 0.05:
        raise ValueError("主体几乎占满画面，没有可替换的背景")

    # prompt 附加场景关键词，避免模型画出人物
    full = f"%s, empty scene, no people, coherent lighting" % prompt
    out, info = GF.fill(img, bg, prompt=full, negative=NEG,
                        steps=steps, denoise=1.0, seed=seed, model=model)
    a = (bg * float(max(0.0, min(1.0, strength))))[..., None]
    out = img * (1 - a) + out * a
    info["feature"] = "ai_backdrop"
    info["bg_frac"] = round(float((bg > 0.5).mean()), 4)
    return np.clip(out, 0, 1).astype(np.float32), info


def generative_fill(img: np.ndarray, mask: np.ndarray, prompt: str,
                    steps: int = 20, seed: Optional[int] = None,
                    model: str = 'sdxl') -> Tuple[np.ndarray, Dict]:
    """创成式填充：用户手涂 mask + 描述。mask 直接给 genfill."""
    from . import genfill as GF

    m = np.clip(np.asarray(mask, np.float32), 0, 1)
    if m.ndim == 3:
        m = m[..., 0]
    if float((m > 0.5).sum()) < 64:
        raise ValueError("涂抹的区域太小（至少 8×8 像素）")
    out, info = GF.fill(img, m, prompt=prompt, negative=NEG,
                        steps=steps, seed=seed, model=model)
    info["feature"] = "generative_fill"
    return out, info


def remove_prop(img: np.ndarray, subject_mask: Optional[np.ndarray] = None,
                mask: Optional[np.ndarray] = None,
                strength: float = 1.0, steps: int = 20,
                seed: Optional[int] = None, model: str = "sdxl"
                ) -> Tuple[np.ndarray, Dict]:
    """智能道具修复：识别灯架/柔光箱/三脚架并让模型补全周围环境.

    自动识别复用 rigfix 的形状判据；`mask` 给定时直接用（手涂）。
    `model`：sdxl（质量）或 sd15（快）。识别不到道具时抛 ValueError。
    """
    from . import genfill as GF
    from . import rigfix as RF

    if mask is None:
        m, dinfo = RF.detect_rig(img, subject_mask)
        if not dinfo.get("poles") and not dinfo.get("heads"):
            raise ValueError(
                "没有识别到道具（灯架/柔光箱/三脚架）；可用手涂 mask 再试")
    else:
        m = np.clip(np.asarray(mask, np.float32), 0, 1)
        if m.ndim == 3:
            m = m[..., 0]
        dinfo = {"manual": True}
    # 道具抹除 prompt：让模型延续周围环境，不要发明新物体
    out, info = GF.fill(img, m,
                        prompt="clean empty background, seamless "
                               "continuation of surrounding area",
                        negative=NEG + ", object, pole, equipment",
                        steps=steps, seed=seed, model=model)
    a = (m * float(max(0.0, min(1.0, strength))))[..., None]
    out = img * (1 - a) + out * a
    info.update({k: v for k, v in dinfo.items()})
    info["feature"] = "remove_prop"
    return np.clip(out, 0, 1).astype(np.float32), info


def info() -> dict:
    from . import genfill as GF

    return {"features": ["ai_backdrop", "generative_fill", "remove_prop"],
            "comfyui": GF.available(),
            "server": GF.SERVER}
