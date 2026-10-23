# -*- coding: utf-8 -*-
"""Local inpainting with LaMa (Apache-2.0).

Why this sits beside the diffusion filler rather than replacing it: the two
solve different problems and a retoucher reaches for different ones.

* **This (LaMa)** — object removal on a textured or structured background:
  a passer-by, a sign, a cable, a blemish on a wall. It is a feed-forward
  convolution, so it is deterministic, needs no prompt, and finishes in a
  second or two. Nothing is synthesised; it propagates the surrounding texture
  into the hole, which is exactly right when the background continues.
* **Diffusion (ComfyUI)** — replacing content with *something else*: a new
  backdrop, a removed garment, an added object. It needs a prompt and seconds
  to tens of seconds, and it invents structure.

Using diffusion for a cable removal produces invented detail on a wall that had
none, which reads as smudge; using LaMa to change a backdrop cannot work at
all. The API exposes both.

Implementation notes that matter:

* LaMa downsamples by 8, so inputs are **padded to a multiple of 8** rather
  than resized to a fixed square. Resizing to 512 loses detail on a 4000px
  frame and changes the structure the model is meant to continue.
* The mask is **dilated** before inference. A mask that ends exactly at the
  object's edge leaves a one-pixel rim of the original object, which after
  compositing reads as a thin outline.
* The composite uses a feathered mask so the seam does not step.
"""
from __future__ import annotations

from .paths import models_dir
import os
from typing import Optional, Tuple

import numpy as np
import cv2

_SESSION = None
_SESSION_PATH = None
_INPUT = None


def model_path() -> str:
    return os.path.join(models_dir(), "lama_fp32.onnx")


def available() -> bool:
    return os.path.exists(model_path())


def _session():
    global _SESSION, _SESSION_PATH, _INPUT
    path = model_path()
    if _SESSION is not None and _SESSION_PATH == path:
        return _SESSION, _INPUT
    if not os.path.exists(path):
        raise FileNotFoundError("LaMa model not found: " + path)

    import onnxruntime as ort

    from .matting import _prepare_gpu_dlls
    _prepare_gpu_dlls()
    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)

    # read the interface rather than assuming it: the exported signature varies
    # between conversions (4-channel RGB+mask vs separate inputs, fixed vs
    # dynamic size)
    ins = sess.get_inputs()
    spec = {"names": [i.name for i in ins],
            "shapes": [list(i.shape) for i in ins],
            "types": [i.type for i in ins],
            "outputs": [o.name for o in sess.get_outputs()]}
    _SESSION, _SESSION_PATH, _INPUT = sess, path, spec
    return _SESSION, _INPUT


def info() -> dict:
    d = {"available": available(), "path": model_path(),
         "license": "Apache-2.0 (LaMa / SAIC)"}
    if available():
        try:
            _, spec = _session()
            d.update(interface=spec)
            d["providers"] = _session()[0].get_providers()
        except Exception as exc:
            d["error"] = str(exc)[:200]
    return d


def _pad_to(arr: np.ndarray, mult: int = 8) -> Tuple[np.ndarray, int, int]:
    h, w = arr.shape[:2]
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        arr = cv2.copyMakeBorder(arr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
    return arr, h, w


def _run_window(sess, spec, img: np.ndarray, mask: np.ndarray,
                win: int) -> Tuple[np.ndarray, np.ndarray]:
    """Run the model on one `win`-sized window, padding at the frame edge.

    The exported graph is fixed at 512x512, so a window is taken at **native
    resolution** rather than resizing the frame to a square. Stretching a 700px
    frame into 512x512 distorts the texture scale the model is meant to
    continue, and the result would be a plausible fill at the wrong grain.
    """
    h, w = img.shape[:2]
    pad_y = max(0, win - h)
    pad_x = max(0, win - w)
    if pad_y or pad_x:
        img = cv2.copyMakeBorder(img, 0, pad_y, 0, pad_x,
                                 cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, 0, pad_y, 0, pad_x,
                                  cv2.BORDER_REFLECT_101)

    names = spec["names"]
    shapes = spec["shapes"]
    # LaMa's contract is that the hole is **already empty** in the image it
    # receives: the official pipeline multiplies the image by (1 - mask) before
    # inference. Feeding the original content inside the hole contradicts the
    # mask, and the model answers with a flat white region (measured: mask
    # interior max 1.0, mean 0.84) instead of continuing the surroundings.
    img = img * (1.0 - mask[..., None])
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[None]).astype(np.float32)
    mk = np.ascontiguousarray(mask[None, None]).astype(np.float32)

    feeds = {}
    for name, shape in zip(names, shapes):
        # dims excludes the batch dimension, so dims[0] is the channel count —
        # reading dims[1] here bound the mask to the image input and produced
        # "invalid dimensions for input: mask, Got 3 Expected 1"
        dims = [d for d in shape if isinstance(d, int)]
        ch = dims[0] if dims else 3
        if ch == 4:
            feeds[name] = np.concatenate([x, mk], axis=1)
        elif ch == 1:
            feeds[name] = mk
        else:
            feeds[name] = x
    if len(names) == 1 and list(feeds[names[0]].shape)[1] == 3:
        feeds[names[0]] = np.concatenate([x, mk], axis=1)

    out = sess.run(None, feeds)[0]
    out = np.asarray(out, np.float32)[0]
    if out.shape[0] == 3:
        out = out.transpose(1, 2, 0)

    # This export emits **0-255**, not 0-1. Clipping it to [0,1] saturates
    # everything above 1.0 and the "fill" comes back as a flat white region
    # (measured 96.7% of the hole pure white). The scale is detected from the
    # output itself rather than hardcoded, so an export that does emit 0-1, or
    # one that emits [-1,1], still lands correctly.
    lo, hi = float(out.min()), float(out.max())
    if hi > 1.5:
        out = out / 255.0
    elif lo < -0.05:
        out = (out + 1.0) / 2.0
    return np.clip(out, 0.0, 1.0), mask


def fill(img: np.ndarray, mask: np.ndarray, dilate: int = 4,
         feather: float = 2.0, win: int = 512,
         overlap: int = 64) -> Tuple[np.ndarray, dict]:
    """Remove the masked content, continuing the surrounding texture.

    `mask` is a 2-D array (or HxWx1) with >0.5 marking the region to remove.
    Returns (image, info).

    Because the graph is a fixed 512x512, only the windows that actually
    contain mask pixels are run. A blemish removal touches one window instead
    of the whole frame, which is the difference between a sub-second edit and a
    minute.
    """
    sess, spec = _session()
    h, w = img.shape[:2]

    m = np.asarray(mask, np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    binary = (m > 0.5).astype(np.uint8)
    if binary.sum() == 0:
        return img.copy(), {"removed_px": 0, "skipped": "empty mask"}

    # dilate so the object's own edge is inside the hole, otherwise a rim of
    # the original object survives and shows as an outline after compositing
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1,) * 2)
        binary = cv2.dilate(binary, k)
    binary = np.clip(binary, 0, 1).astype(np.float32)

    # a margin of real context around the hole: the model continues the
    # surroundings, so it has to see them
    ys, xs = np.nonzero(binary > 0)
    pad = max(32, win // 8)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w, int(xs.max()) + pad)

    step = max(64, win - overlap)
    oy = list(range(y0, max(y0 + 1, y1), step))
    ox = list(range(x0, max(x0 + 1, x1), step))
    total = len(oy) * len(ox)

    out = img.copy()
    # the region the call is allowed to touch: the dilated mask plus the
    # feather's spread. Returned so a caller can assert locality exactly
    # instead of guessing how far the feather reaches.
    touched = np.zeros((h, w), bool)
    done = 0
    for wy in oy:
        for wx in ox:
            # a window is skipped unless it actually covers mask
            cy0 = max(0, min(wy, max(0, h - win)))
            cx0 = max(0, min(wx, max(0, w - win)))
            cy1 = min(h, cy0 + win)
            cx1 = min(w, cx0 + win)
            sub_m = binary[cy0:cy1, cx0:cx1]
            if sub_m.max() <= 0:
                done += 1
                continue

            sub_img = img[cy0:cy1, cx0:cx1]
            filled, _ = _run_window(sess, spec, sub_img, sub_m, win)
            fh = min(cy1 - cy0, filled.shape[0])
            fw = min(cx1 - cx0, filled.shape[1])
            filled = filled[:fh, :fw]

            a = sub_m[:fh, :fw].astype(np.float32)
            if feather > 0:
                a = cv2.GaussianBlur(a, (0, 0), feather)
            a = np.clip(a, 0.0, 1.0)
            base = out[cy0:cy0 + fh, cx0:cx0 + fw]
            out[cy0:cy0 + fh, cx0:cx0 + fw] = \
                base * (1.0 - a[..., None]) + filled * a[..., None]
            touched[cy0:cy0 + fh, cx0:cx0 + fw] |= (a > 0.0)
            done += 1

    # clip only what was written: a decoded frame can hold values a hair above
    # 1.0, and a full-frame clip moves those by one ULP outside every window
    if touched.any():
        out[touched] = np.clip(out[touched], 0.0, 1.0)
    return out.astype(np.float32), {
        "removed_px": int((binary > 0).sum()),
        "size": [int(w), int(h)],
        "windows": total,
        "model": "lama",
        "fixed_input": [win, win],
        "touched": touched,
    }


def heal_regions(img: np.ndarray, rects) -> Tuple[np.ndarray, dict]:
    """Convenience: remove rectangular regions (x, y, w, h) in pixels."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.float32)
    n = 0
    for r in rects or []:
        x, y, rw, rh = [int(v) for v in r[:4]]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + max(1, rw)), min(h, y + max(1, rh))
        if x1 > x0 and y1 > y0:
            m[y0:y1, x0:x1] = 1.0
            n += 1
    out, info = fill(img, m)
    info["regions"] = n
    return out, info
