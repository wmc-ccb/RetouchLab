"""Portrait matting: separate the subject from the background.

Uses MODNet, a matting network designed for portraits. It emits a soft alpha
matte rather than a hard mask, which matters for hair: a binary cut-out leaves
a visible fringe, while a soft matte keeps the flyaway strands.

The matte has several uses here:

* **background separation** — replace, blur or darken the backdrop;
* **subject masking** — the tone stages use it to keep global grading off the
  background (see `RetouchPipeline.subject_mask`), which is far more accurate
  than the face-box approximation;
* **edge-aware retouching** — skin work can be limited to the subject without
  a chroma test.

The network is fully convolutional, so it runs at whatever resolution the
caller supplies. A bounded internal size keeps the cost predictable on large
photos while still producing a matte that upscales cleanly, because the matte
is smooth by nature.
"""
from __future__ import annotations

from .paths import models_dir
import os
import threading
from typing import Optional

import numpy as np
import cv2

MODEL_NAME = "modnet.onnx"

# Input normalisation used by the published model.
_MEAN = np.array([0.5, 0.5, 0.5], np.float32)
_STD = np.array([0.5, 0.5, 0.5], np.float32)

_SESSION = None
_SESSION_LOCK = threading.Lock()


def _prepare_gpu_dlls() -> None:
    """Make the CUDA runtime visible to onnxruntime on Windows.

    `onnxruntime-gpu` links against cuDNN and cuBLAS at load time but does not
    ship them. Torch bundles a matching set under `torch/lib` (CUDA 12.8 /
    cuDNN 9 here), so registering that directory lets the CUDA provider load
    instead of silently falling back to CPU. Without this the matting pass ran
    at ~3.9 s per image; with it, well under a second.

    A mismatch between the bundles is not fatal: the provider simply fails to
    initialise and the CPU provider is used, which the caller sees as a slower
    but correct result.
    """
    if os.name != "nt":
        return
    try:
        import torch  # noqa: F401  (only imported for its bundled runtime)

        lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isdir(lib):
            return
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(lib)
        if lib not in os.environ.get("PATH", ""):
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def model_path() -> str:
    return os.path.join(models_dir(), MODEL_NAME)


def available() -> bool:
    return os.path.exists(model_path())


def _session(providers=None):
    """One shared InferenceSession; building it is the expensive part."""
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    with _SESSION_LOCK:
        if _SESSION is not None:
            return _SESSION
        _prepare_gpu_dlls()
        import onnxruntime as ort

        path = model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"matting model not found at {path}. Download modnet.onnx "
                "into the models/ directory.")
        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in avail] or ["CPUExecutionProvider"]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            _SESSION = ort.InferenceSession(path, sess_options=so, providers=providers)
        except Exception:
            # an unusable GPU provider must not take the feature down with it
            _SESSION = ort.InferenceSession(path, sess_options=so,
                                            providers=["CPUExecutionProvider"])
    return _SESSION


def _to_input(img: np.ndarray, size: int) -> tuple:
    """Resize and normalise to NCHW float32.

    Returns (blob, padded_size) where `padded_size` is the (w, h) actually fed
    to the network. MODNet is an encoder-decoder: each downsample halves the
    spatial size and the skip concatenations require the halves to line up, so
    an odd dimension raises "Non concat axis dimensions must match". Sizes are
    rounded up to a multiple of 32, which covers every downsample the network
    performs.
    """
    MULTIPLE = 32
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh = max(MULTIPLE, int(round(h * scale)))
    nw = max(MULTIPLE, int(round(w * scale)))
    # round up, then clamp so a tiny input still has room
    ph = ((nh + MULTIPLE - 1) // MULTIPLE) * MULTIPLE
    pw = ((nw + MULTIPLE - 1) // MULTIPLE) * MULTIPLE

    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    if (pw, ph) != (nw, nh):
        # reflect-pad: repeating edge pixels would bleed background into the
        # matte at the border, while reflection continues the local structure
        small = cv2.copyMakeBorder(small, 0, ph - nh, 0, pw - nw,
                                   cv2.BORDER_REFLECT_101)
    x = (small.astype(np.float32) - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), (pw, ph, nw, nh)


def alpha(img: np.ndarray, size: int = 768,
          feather_px: float = 0.0) -> np.ndarray:
    """Return a soft subject matte in [0, 1] at the input's resolution.

    Parameters
    ----------
    img : float32 RGB in [0, 1]
    size : longest side used for inference; the matte is upscaled back.
    feather_px : optional extra blur, in pixels of the original image.
    """
    h, w = img.shape[:2]
    sess = _session()
    name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    blob, (pw, ph, nw, nh) = _to_input(img, size)
    pred = sess.run([out_name], {name: blob})[0]
    matte = np.asarray(pred, np.float32).reshape(pred.shape[-2], pred.shape[-1])

    # undo the padding, then resize to the original frame
    matte = matte[:nh, :nw]
    if (nw, nh) != (w, h):
        matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
    matte = np.clip(matte, 0.0, 1.0)

    if feather_px and feather_px > 0:
        k = int(max(3, round(feather_px * 6) | 1))
        matte = cv2.GaussianBlur(matte, (k, k), feather_px,
                                 borderType=cv2.BORDER_REPLICATE)
    return matte.astype(np.float32)


def refine_edges(matte: np.ndarray, img: np.ndarray,
                 radius: int = 4) -> np.ndarray:
    """Tighten the matte against the image's own luminance edges.

    A network matte is smooth and slightly soft at hair and shoulders. Guided
    filtering uses the image as a guide, which snaps the transition to the
    actual luminance edge without introducing the blockiness a hard threshold
    would.

    `guided_filter` takes a *single-channel* guide, so the image is reduced to
    luminance first. Passing the RGB image directly reaches a 4-D array and
    trips an OpenCV assertion.
    """
    from .imgutil import guided_filter, luminance

    g = luminance(img).astype(np.float32)
    m = matte.astype(np.float32)[..., None]
    out = guided_filter(g, m, radius=max(2, radius), eps=1e-4)
    if out.ndim == 3:
        out = out[..., 0]
    return np.clip(out, 0.0, 1.0)


def compose(img: np.ndarray, matte: np.ndarray,
            background: np.ndarray = None,
            bg_color: tuple = None,
            blur_bg: float = 0.0,
            darken_bg: float = 0.0,
            desat_bg: float = 0.0) -> np.ndarray:
    """Composite the subject over a backdrop.

    The backdrop is built first, then the subject is laid over it exactly once
    with the matte as the alpha. Passing `background` uses that image;
    otherwise the backdrop starts as the original pixels and the modifiers
    below are applied to it. `bg_color` replaces the backdrop with a flat
    colour, which is what the ID-photo and cut-out presets need.

    Building the backdrop first matters: blending the colour in with the matte
    and then compositing the subject again would apply the alpha twice and
    darken the edges of hair.
    """
    a = np.clip(matte, 0.0, 1.0)
    a3 = a[..., None]

    if background is not None:
        bg = background.astype(np.float32)
    elif bg_color is not None:
        c = np.asarray(bg_color, np.float32).reshape(-1)
        if c.size == 1:
            c = np.repeat(c, 3)
        if c.size != 3:
            raise ValueError(f"bg_color must be a scalar or 3 values, got {c.size}")
        bg = np.broadcast_to(c.reshape(1, 1, 3),
                             (img.shape[0], img.shape[1], 3)).astype(np.float32).copy()
    else:
        bg = img.astype(np.float32).copy()

    # the modifiers describe the *backdrop*, so they apply to it directly
    if blur_bg > 0:
        k = int(max(3, round(blur_bg * 4) | 1))
        bg = cv2.GaussianBlur(bg, (k, k), blur_bg,
                              borderType=cv2.BORDER_REPLICATE)
    if darken_bg > 0:
        bg = bg * (1.0 - float(np.clip(darken_bg, 0.0, 1.0)))
    if desat_bg > 0:
        g = (bg * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2,
                                                                     keepdims=True)
        k = float(np.clip(desat_bg, 0.0, 1.0))
        bg = bg * (1.0 - k) + g * k

    out = img.astype(np.float32) * a3 + bg * (1.0 - a3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def alpha_hybrid(img: np.ndarray, size: int = 768,
                 feather_px: float = 1.0) -> Tuple[np.ndarray, str]:
    """Subject alpha that keeps accessories, with fine hair edges.

    Two models answer different halves of the question and neither is enough
    alone:

    * **MODNet** is trained for hair — its alpha follows individual strands,
      which a 256x256 semantic map cannot resolve. But it is a *portrait*
      matte: glasses, earrings, a necklace or a held prop are frequently
      classified as background and cut away.
    * **selfie_multiclass** labels accessories as their own class, so nothing
      is silently dropped, but its masks are low-resolution and blocky at hair.

    Taking the per-pixel maximum keeps MODNet's hair detail and restores
    everything the semantic model recognised as subject. The result is then
    snapped to the image's own luminance edges.

    Returns (alpha, source) where `source` names what was actually used, so a
    caller can report the fallback.
    """
    from . import semantic as SEM

    base = None
    source = "none"
    if available():
        try:
            base = alpha(img, size=size, feather_px=0.0)
            source = "modnet"
        except Exception:
            base = None

    sem = SEM.segment(img) if SEM.available() else None
    if sem is not None:
        # adaptive: the model's absolute confidence is image-dependent, so the
        # mask is normalised against its own distribution rather than cut at a
        # fixed value (a fixed 0.35 turned a 0.04-confidence face into a
        # zero-area mask)
        subj = sem.subject()
        if float(subj.max()) > 1e-3:
            base = subj if base is None else np.maximum(base, subj)
            source = "semantic" if source == "none" else "modnet+semantic"

    if base is None:
        raise RuntimeError(
            "no matting model available: install modnet.onnx or "
            "selfie_multiclass_256x256.tflite into models/")

    if feather_px and feather_px > 0:
        k = int(max(3, round(feather_px * 6) | 1))
        base = cv2.GaussianBlur(base, (k, k), feather_px,
                                borderType=cv2.BORDER_REPLICATE)
    try:
        base = refine_edges(base, img)
    except Exception:
        pass
    return np.clip(base, 0.0, 1.0).astype(np.float32), source


def subject_alpha(img: np.ndarray, mode: str = "hybrid", size: int = 768,
                  feather_px: float = 1.0) -> np.ndarray:
    """One entry point for the three mask sources.

    mode: "portrait" (MODNet only), "semantic" (classes only), or "hybrid"
    (both, which is what the UI should use by default).
    """
    if mode == "portrait":
        m = alpha(img, size=size, feather_px=feather_px)
        try:
            return refine_edges(m, img)
        except Exception:
            return m
    if mode == "semantic":
        from . import semantic as SEM

        reg = SEM.segment(img)
        if reg is None:
            raise RuntimeError("semantic model not installed")
        s = reg.subject()
        try:
            return refine_edges(s, img)
        except Exception:
            return s
    out, _ = alpha_hybrid(img, size=size, feather_px=feather_px)
    return out


def replace_background(img: np.ndarray, background: np.ndarray,
                       size: int = 768, feather_px: float = 1.0,
                       mode: str = "hybrid") -> np.ndarray:
    """Swap the backdrop, matching the new background to the frame."""
    h, w = img.shape[:2]
    bg = background
    if bg.shape[:2] != (h, w):
        # cover-fit rather than stretch, so faces and horizons keep proportion
        sh, sw = bg.shape[:2]
        scale = max(w / sw, h / sh)
        nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
        bg = cv2.resize(bg, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = max(0, (nw - w) // 2)
        y0 = max(0, (nh - h) // 2)
        bg = bg[y0:y0 + h, x0:x0 + w]
    m = subject_alpha(img, mode=mode, size=size, feather_px=feather_px)
    return compose(img, m, background=bg)
