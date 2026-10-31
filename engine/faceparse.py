# -*- coding: utf-8 -*-
"""Face sub-masks: brows, eyes, lips, teeth, nose — from face-parsing.

selfie_multiclass answers "where is the skin / hair / clothes" at six classes,
which is what whole-region retouching needs. It cannot separate an eye from the
skin around it, and 祛黑眼圈 / 美瞳 / 牙齿美白 all act on structures *inside*
the face, so they need a finer answer. This module provides it.

Model: `jonathandinu/face-parsing` (MIT), a SegFormer fine-tuned for 19 face
classes. Classes are combined into the masks an editor exposes:

    眉毛   = left brow + right brow
    眼睛   = left eye + right eye
    嘴唇   = upper lip + lower lip
    口腔   = mouth interior + teeth
    牙齿   = teeth
    鼻子   = nose
    皮肤   = skin + neck

`selfie_multiclass` stays the source for hair / clothes / accessories: the two
models are merged, not replaced — this one only sees the head and neck, so
asking it for "clothes" would be a guess.

The first inference loads the model into GPU memory; the module caches it and
the per-image result, same as the other mask sources.
"""
from __future__ import annotations

from .paths import models_dir
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

MODEL_NAME = "faceparsing.safetensors"
MODEL_DIR_NAME = "faceparsing"

# Class table — read from the published config.json's id2label rather than
# recalled. The first guess here was the CelebAMask-HQ layout, which has a
# "teeth" class at 18; this model's 18 is *cloth*, so "牙齿" lit up on the
# subject's jacket (26% of the frame) and every centroid check failed.
# There is no separate teeth class in this table: the mouth interior is one
# class, so 牙齿 is folded into 口腔.
CLS = {
    "background": 0, "skin": 1, "nose": 2, "eye_g": 3, "l_eye": 4,
    "r_eye": 5, "l_brow": 6, "r_brow": 7, "l_ear": 8, "r_ear": 9,
    "mouth": 10, "u_lip": 11, "l_lip": 12, "hair": 13, "hat": 14,
    "ear_r": 15, "neck_l": 16, "neck": 17, "cloth": 18,
}

# what the UI exposes, as groups over the class table
GROUPS: Dict[str, Tuple[int, ...]] = {
    "brows": (CLS["l_brow"], CLS["r_brow"]),
    "eyes": (CLS["l_eye"], CLS["r_eye"]),
    "lips": (CLS["u_lip"], CLS["l_lip"]),
    "mouth": (CLS["mouth"],),
    "nose": (CLS["nose"],),
    "skin": (CLS["skin"], CLS["neck"]),
}
GROUP_LABELS = {
    "brows": "眉毛", "eyes": "眼睛", "lips": "嘴唇", "mouth": "口腔",
    "nose": "鼻子", "skin": "面部皮肤",
}

_PIPE = None
_LOCK = threading.Lock()


def model_dir() -> str:
    return os.path.join(models_dir(), MODEL_DIR_NAME)


def model_path() -> str:
    # 加载器（transformers from_pretrained）读的是 **目录版** 权重
    # faceparsing/model.safetensors；根目录的同名独立副本是历史冗余，
    # 已从发行包移除（省 323MB）。
    return os.path.join(model_dir(), "model.safetensors")


def available() -> bool:
    # config.json is what the loader needs; weights alone are not enough
    return (os.path.exists(os.path.join(model_dir(), "config.json")) and
            os.path.exists(model_path()))


def _pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        from transformers import SegformerForSemanticSegmentation

        _PIPE = SegformerForSemanticSegmentation.from_pretrained(model_dir())
        dev = "cuda" if _cuda_ok() else "cpu"
        _PIPE = _PIPE.to(dev).eval()
    return _PIPE


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class FaceParts:
    """Per-group soft masks at the input resolution."""

    def __init__(self, raw: Dict[str, np.ndarray]):
        self.raw = raw
        self.masks = {}
        for gname, ids in GROUPS.items():
            acc = None
            for i in ids:
                a = raw.get(i)
                if a is None:
                    continue
                acc = a if acc is None else np.maximum(acc, a)
            if acc is not None:
                self.masks[gname] = np.clip(acc, 0.0, 1.0)

    def __getitem__(self, key: str) -> Optional[np.ndarray]:
        return self.masks.get(key)

    def names(self) -> list:
        return list(self.masks.keys())

    def coverage(self) -> Dict[str, float]:
        return {k: round(float((v > 0.5).mean()), 4)
                for k, v in self.masks.items()}

    def union(self, *keys: str) -> np.ndarray:
        acc = None
        for k in keys:
            m = self.masks.get(k)
            if m is None:
                continue
            acc = m if acc is None else np.maximum(acc, m)
        if acc is None:
            return np.zeros_like(next(iter(self.raw.values())))
        return np.clip(acc, 0.0, 1.0)


_CACHED: Optional[Tuple[tuple, FaceParts]] = None


def parse(img: np.ndarray, size: int = 512,
          face_box: Optional[Tuple[int, int, int, int]] = None
          ) -> Optional[FaceParts]:
    """Run face parsing; results upscaled to the input's resolution.

    **Crop to the face first.** The model's input is 512x512, so on a full frame
    a face occupying 8% of the width is resampled to ~40 px and the small
    structures are lost: measured on a night portrait, parsing the whole frame
    gave skin 2.2% and zero brows/mouth while parsing the face crop gave skin
    15.1%, nose 1.4% and visible eyes/lips. `face_box` (x, y, w, h) selects the
    crop; when it is absent one is estimated from the most confident detection.

    Cached per array identity, same policy as the other mask sources: the
    pipeline hands the same array to several operators inside one run, and
    re-running a 300 MB model per operator is the kind of slowdown that looks
    like a bug in something else.
    """
    global _CACHED
    if not available():
        return None
    # The cache key has to include the crop, not just the array identity: a
    # caller that first parses the whole frame and then asks for a face crop
    # would otherwise be served the whole-frame result, which measured skin
    # 2.5% instead of 15% and hid every small structure.
    fb = None if face_box is None else tuple(round(float(v), 1)
                                             for v in face_box[:4])
    key = (id(img), fb)
    if _CACHED is not None and _CACHED[0] == key:
        return _CACHED[1]

    import torch

    h, w = img.shape[:2]
    pipe = _pipeline()
    dev = next(pipe.parameters()).device

    # ---- pick the crop ---------------------------------------------------
    cx0, cy0, cw, ch = 0, 0, w, h
    if face_box is not None:
        bx, by, bw, bh = [float(v) for v in face_box[:4]]
        pad = max(bw, bh) * 0.55
        cx0 = max(0, int(bx - pad))
        cy0 = max(0, int(by - pad))
        cw = min(w, int(bx + bw + pad)) - cx0
        ch = min(h, int(by + bh + pad)) - cy0
    if cw < 8 or ch < 8:
        cx0, cy0, cw, ch = 0, 0, w, h
    crop = img[cy0:cy0 + ch, cx0:cx0 + cw]

    u8 = (np.clip(crop, 0, 1) * 255).astype(np.uint8)
    rgb = cv2.resize(u8, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(
        rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
    x = x.to(dev)

    # ImageNet normalisation, from the model's preprocessor_config.json. The
    # first run skipped it and the output collapsed: only skin (1.1%) and lips
    # (0.02%) fired while brows/eyes/mouth/nose were exactly zero — the network
    # was being fed a distribution it had never seen.
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    x = (x - mean) / std

    with torch.no_grad():
        logits = pipe(pixel_values=x).logits
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()   # (19, s, s)

    argmax = probs.argmax(axis=0)
    raw: Dict[int, np.ndarray] = {}
    for cid in set(GROUPS["brows"]) | set(GROUPS["eyes"]) | set(GROUPS["lips"]) \
            | set(GROUPS["mouth"]) | set(GROUPS["nose"]) | set(GROUPS["skin"]):
        hard = (argmax == cid).astype(np.float32)
        soft = np.clip((probs[cid] - 0.15) / 0.25, 0.0, 1.0)
        a = np.maximum(hard, soft)
        # 先回到裁剪区的分辨率，再贴回整图（其余区域为 0）
        if a.shape[:2] != (ch, cw):
            a = cv2.resize(a, (cw, ch), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((h, w), np.float32)
        full[cy0:cy0 + ch, cx0:cx0 + cw] = a
        raw[cid] = np.clip(full, 0.0, 1.0).astype(np.float32)

    parts = FaceParts(raw)
    _CACHED = (key, parts)
    return parts


def refine(mask: np.ndarray, img: np.ndarray, radius: int = 3) -> np.ndarray:
    """Snap a soft class mask onto the image's own edges."""
    from .imgutil import guided_filter, luminance

    g = luminance(img)
    out = guided_filter(g, mask[..., None], radius=max(2, radius), eps=1e-4)
    if out.ndim == 3:
        out = out[..., 0]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def info() -> dict:
    d = {"available": available(), "path": model_dir(),
         "groups": GROUP_LABELS,
         "license": "MIT (face-parsing / SegFormer)"}
    if available():
        d["providers"] = "cuda" if _cuda_ok() else "cpu"
    return d
