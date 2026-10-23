# -*- coding: utf-8 -*-
"""Generative fill through a local ComfyUI instance.

The reference application's generative families (GenerativeFill, FillLawn,
ShoeClean) are one operation underneath: mask a region, let a diffusion model
synthesise content there, composite the result back so untouched pixels stay
bit-exact. What varies is the prompt and how the mask was made.

ComfyUI already runs locally with an inpaint checkpoint
(`sd15_inpaint.ckpt`) plus the inpaint ControlNet, so this module drives it
over its HTTP API instead of loading a second SD1.5 into this process: one
GPU-resident weights file serves both programs, and its queue serialises
access.

Graph
-----
`LoadImage` -> `InpaintModelConditioning` (pixels + mask -> positive /
negative / noised latent) with the inpaint ControlNet applied to both
conditionings, then KSampler -> VAEDecode. Three schema facts cost two failed
attempts to learn:

* `LoadImage` validates against the server's input directory, so inputs must
  be POSTed to `/upload/image` first — inline base64 is rejected.
* `ControlNetApplyAdvanced.image` needs an IMAGE, not a MASK; the frame
  itself is the control signal.
* This ComfyUI build requires `noise_mask=True` on
  `InpaintModelConditioning`; without it the node reports a missing input
  even though `mask` is supplied.

Compositing is done here rather than by the model: only the masked region is
pasted back, feathered by the mask's own soft edge, which keeps the edit
local and the seam invisible.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional, Tuple

import numpy as np

SERVER = os.environ.get(
    "RETOUCHLAB_COMFYUI", "http://127.0.0.1:8388")
# 端口 8188 落在本机 Hyper-V 保留段 8115-8214 内（netsh excludedportrange
# 实测），bind 直接 PermissionError；8388 在 8315-8514 的可用窗口里。
CHECKPOINT = "sd15_inpaint.ckpt"
CONTROLNET = "control_v11p_sd15_inpaint_fp16.safetensors"


_AVAIL_CACHE = {"at": 0.0, "ok": False}
_AVAIL_TTL = 30.0


def available(timeout: float = 1.5, cache: bool = True) -> bool:
    """True when ComfyUI answers and the checkpoint is registered.

    结果缓存 30 秒：这个探测要连两次 ComfyUI，未安装时每次约 2 秒
    （连接被拒 + 超时）。健康检查这类高频端点不能每次都付这个代价，
    而 ComfyUI 装好/关掉不会在几秒内反复变化。
    超时压到 1.5s：本地服务要么立刻应答，要么就是没装。
    """
    now = time.time()
    if cache and now - _AVAIL_CACHE["at"] < _AVAIL_TTL:
        return bool(_AVAIL_CACHE["ok"])
    ok = _available_probe(timeout)
    _AVAIL_CACHE["at"] = now
    _AVAIL_CACHE["ok"] = ok
    return ok


def _available_probe(timeout: float = 5.0) -> bool:
    """ComfyUI 在跑、且注册了 inpaint checkpoint，才算可用。

    本模块本身只用 numpy / cv2 / requests 做 mask 合成与 HTTP 调用，
    **不依赖 torch**（生成是在 ComfyUI 那一侧跑的）。所以判定只看
    ComfyUI 是否应答且模型就位。
    """
    try:
        import requests
        if requests.get(SERVER + "/system_stats", timeout=timeout).status_code != 200:
            return False
        r = requests.get(SERVER + "/object_info/CheckpointLoaderSimple",
                         timeout=timeout)
        names = json.dumps(r.json())
        return CHECKPOINT in names
    except Exception:
        return False


def _png_rgb(img: np.ndarray) -> bytes:
    import cv2
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def _upload(img: np.ndarray, name: str) -> str:
    """POST /upload/image; returns the reference LoadImage accepts."""
    import requests
    r = requests.post(
        SERVER + "/upload/image",
        files={"image": (name, _png_rgb(img), "image/png")},
        data={"overwrite": "true", "type": "input"},
        timeout=120)
    if r.status_code != 200:
        raise RuntimeError("upload failed: %s %s" % (r.status_code, r.text[:200]))
    j = r.json()
    sub, fn = j.get("subfolder", ""), j.get("name", name)
    return (sub + "/" + fn) if sub else fn


def _build_graph(img_ref: str, mask_ref: str, prompt: str, negative: str,
                 seed: int, steps: int, cfg: float, denoise: float,
                 controlnet_strength: float, model: str = "sd15") -> dict:
    """Build the inpaint workflow.

    * `sd15`（默认，快）——CheckpointLoaderSimple + inpaint ControlNet，
      768px 工作尺度。
    * `sdxl`（质量）——DiffusersLoader 加载 models/diffusers/sdxl-inpainting/
      （fp16 全套），1024px 工作尺度，无 ControlNet（SDXL inpainting
      checkpoint 本身以 masked latent 训练，加 ControlNet 反而牵制细节）。
    """
    if model == "sdxl":
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": img_ref}},
            "2": {"class_type": "LoadImageMask",
                  "inputs": {"image": mask_ref, "channel": "red"}},
            "3": {"class_type": "DiffusersLoader",
                  "inputs": {"model_path": "sdxl-inpainting"}},
            "4": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": prompt, "clip": ["3", 1]}},
            "5": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": negative, "clip": ["3", 1]}},
            "6": {"class_type": "InpaintModelConditioning",
                  "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                             "vae": ["3", 2], "pixels": ["1", 0],
                             "mask": ["2", 0], "noise_mask": True}},
            "9": {"class_type": "KSampler",
                  "inputs": {"seed": int(seed), "steps": int(steps),
                             "cfg": float(cfg), "sampler_name": "euler",
                             "scheduler": "normal", "denoise": float(denoise),
                             "model": ["3", 0], "positive": ["6", 0],
                             "negative": ["6", 1], "latent_image": ["6", 2]}},
            "10": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["9", 0], "vae": ["3", 2]}},
            "11": {"class_type": "SaveImage",
                   "inputs": {"images": ["10", 0],
                              "filename_prefix": "retouchlab_fill"}},
        }
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": img_ref}},
        "2": {"class_type": "LoadImageMask",
              "inputs": {"image": mask_ref, "channel": "red"}},
        "3": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": CHECKPOINT}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["3", 1]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["3", 1]}},
        "6": {"class_type": "InpaintModelConditioning",
              "inputs": {"positive": ["4", 0], "negative": ["5", 0],
                         "vae": ["3", 2], "pixels": ["1", 0],
                         "mask": ["2", 0], "noise_mask": True}},
        "7": {"class_type": "ControlNetLoader",
              "inputs": {"control_net_name": CONTROLNET}},
        "8": {"class_type": "ControlNetApplyAdvanced",
              "inputs": {"positive": ["6", 0], "negative": ["6", 1],
                         "control_net": ["7", 0], "image": ["1", 0],
                         "strength": float(controlnet_strength),
                         "start_percent": 0.0, "end_percent": 1.0}},
        "9": {"class_type": "KSampler",
              "inputs": {"seed": int(seed), "steps": int(steps),
                         "cfg": float(cfg), "sampler_name": "euler",
                         "scheduler": "normal", "denoise": float(denoise),
                         "model": ["3", 0], "positive": ["8", 0],
                         "negative": ["8", 1], "latent_image": ["6", 2]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["3", 2]}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0],
                          "filename_prefix": "retouchlab_fill"}},
    }


def _fetch(pid: str, timeout: float) -> np.ndarray:
    """Wait for the prompt to finish and return its first output image."""
    import cv2
    import requests

    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        hist = requests.get(SERVER + "/history/" + pid, timeout=30).json()
        if pid in hist:
            break
        time.sleep(1.0)
    else:
        raise TimeoutError("ComfyUI did not finish in %.0fs" % timeout)

    for out in hist[pid].get("outputs", {}).values():
        for im in out.get("images", []):
            vr = requests.get(SERVER + "/view",
                              params={"filename": im["filename"],
                                      "subfolder": im.get("subfolder", ""),
                                      "type": im.get("type", "output")},
                              timeout=120)
            arr = np.frombuffer(vr.content, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) \
                    .astype(np.float32) / 255.0
    raise RuntimeError("no output image in ComfyUI history")


def fill(img: np.ndarray, mask: np.ndarray,
         prompt: str = "clean background, seamless, high quality",
         negative: str = "blurry, artifacts, watermark, text, deformed",
         steps: int = 20, cfg: float = 7.0, denoise: float = 1.0,
         controlnet_strength: float = 1.0, seed: Optional[int] = None,
         timeout: float = 900.0, model: str = "sd15") -> Tuple[np.ndarray, dict]:
    """Synthesise the masked region and composite it back.

    `mask` is float32 [0,1]; values above 0.5 are replaced and its soft edge
    feathers the paste. Frames are downscaled to the model's training scale
    for generation (sd15: 768 long edge; sdxl: 1024) and the result is
    upscaled back — a pass through the RetouchLab superres model can lift
    the generated region further when needed.
    `model`: "sd15"（快，~10s）或 "sdxl"（质量，~30-60s，纹理细节明显更好）。
    Returns (composited image, info).
    """
    import cv2
    import random
    import requests

    if img.shape[:2] != mask.shape[:2]:
        raise ValueError("image and mask must match")
    if model not in ("sd15", "sdxl"):
        raise ValueError(f"unknown model: {model}")

    h, w = img.shape[:2]
    work = 1024.0 if model == "sdxl" else 768.0
    scale = min(1.0, work / max(h, w))
    if scale < 1.0:
        nw = max(8, int(round(w * scale)) // 8 * 8)
        nh = max(8, int(round(h * scale)) // 8 * 8)
        img_s = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        img_s, mask_s = img, mask

    img_ref = _upload(img_s, "retouchlab_frame.png")
    mask_ref = _upload(mask_s, "retouchlab_mask.png")
    t0 = time.perf_counter()

    seed = int(seed) if seed is not None else random.randrange(1 << 31)
    graph = _build_graph(img_ref, mask_ref, prompt, negative, seed,
                         steps, cfg, denoise, controlnet_strength,
                         model=model)
    cid = str(uuid.uuid4())
    r = requests.post(SERVER + "/prompt",
                      json={"prompt": graph, "client_id": cid}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError("ComfyUI rejected the graph: %s"
                           % json.dumps(r.json())[:400])
    pid = r.json()["prompt_id"]

    gen = _fetch(pid, timeout)
    if gen.shape[:2] != (h, w):
        gen = cv2.resize(gen, (w, h), interpolation=cv2.INTER_LANCZOS4)

    a = np.clip(mask, 0.0, 1.0)[..., None]
    out = img.astype(np.float32) * (1.0 - a) + gen.astype(np.float32) * a
    info = {"seed": seed, "steps": steps, "cfg": cfg, "denoise": denoise,
            "model": model, "work_scale": round(scale, 3),
            "controlnet": controlnet_strength,
            "masked_fraction": round(float((mask > 0.5).mean()), 4),
            "seconds": round(time.perf_counter() - t0, 1)}
    return np.clip(out, 0.0, 1.0).astype(np.float32), info
