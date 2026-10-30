# -*- coding: utf-8 -*-
"""轻量 CLIP 图像编码器（ONNX 版），用于「挑图」而不引入 PyTorch。

为什么要这个模块
----------------
「挑图」（`/api/cull`）靠 CLIP 图像嵌入给照片打"互相有多像"的分，
用来从连拍里挑出最好的一张。原来的实现走 transformers+torch，
这两个包加起来 4.2GB——为了一个图像编码器把主程序撑到 5GB 不值得。

装 3GB 依赖只为跑一个 CLIP 图像塔，在打包场景下代价太大。这里用
onnxruntime（主程序本来就有，因为人脸/抠图/超分/Lama 都靠它）跑同一个
CLIP 视觉塔，输出维度与归一化方式完全一致，文本塔不需要（挑图只比图像
之间的相似度）。

模型
----
    models/clip_image_encoder.onnx   (~350MB, ViT-B/32 视觉塔)

文件不存在时 `available()` 返回 False，调用方应退回"按清晰度/曝光排序"
的降级路径——这时挑图仍然可用，只是不懂"内容相似"。
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import numpy as np

_MODEL_NAME = "clip_image_encoder.onnx"
SESSION = None
_SESSION_LOCK = threading.Lock()   # 防并发双初始化；ORT run 本身线程安全


def model_path() -> str:
    from .paths import models_dir
    return os.path.join(models_dir(), _MODEL_NAME)


def available() -> bool:
    return os.path.exists(model_path())


def _session():
    global SESSION
    with _SESSION_LOCK:
        if SESSION is not None:
            return SESSION
        if not available():
            return None
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    # 优先 CUDA；没有 GPU 就退 CPU，两者都试不到就报错
    providers = []
    for p in ("CUDAExecutionProvider", "CPUExecutionProvider"):
        if p in ort.get_available_providers():
            providers.append(p)
    SESSION = ort.InferenceSession(model_path(), sess_options=opts,
                                   providers=providers or None)
    return SESSION


def _preprocess(images) -> np.ndarray:
    """[0,1] RGB float -> CLIP 输入张量 (N,3,224,224)，ImageNet 归一化。

    CLIP 的标准预处理：
      1. 短边缩放到 224（这里直接中心裁 224，保持构图中心）
      2. RGB 通道按 (0.48145466, 0.4578275, 0.40821073) 均值、
         (0.26862954, 0.26130258, 0.27577711) 标准差归一化
    """
    import cv2

    mean = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    std = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
    out = []
    for im in images:
        a = np.clip(im, 0, 1).astype(np.float32)
        h, w = a.shape[:2]
        s = 224.0 / min(h, w)
        if s < 1:
            a = cv2.resize(a, (max(224, int(round(w * s))),
                               max(224, int(round(h * s)))),
                           interpolation=cv2.INTER_AREA)
            h, w = a.shape[:2]
        y0 = max(0, (h - 224) // 2)
        x0 = max(0, (w - 224) // 2)
        a = a[y0:y0 + 224, x0:x0 + 224]
        if a.shape[:2] != (224, 224):
            a = cv2.resize(a, (224, 224), interpolation=cv2.INTER_AREA)
        a = (a - mean) / std
        out.append(a.transpose(2, 0, 1))
    return np.stack(out).astype(np.float32)


def embed_images(images, batch: int = 8) -> np.ndarray:
    """返回 L2 归一化的图像嵌入，shape (N, D)。

    与 transformers 版 `match.embed_images` 的契约一致：行已归一化，
    可以直接点积当余弦相似度用。
    """
    sess = _session()
    if sess is None:
        raise RuntimeError("CLIP ONNX 模型缺失: %s" % model_path())

    inp = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]

    chunks = []
    for i in range(0, len(images), batch):
        x = _preprocess(images[i:i + batch])
        res = sess.run(out_names, {inp: x})
        feats = res[0]
        feats = np.asarray(feats, np.float32)
        if feats.ndim == 3:            # (N, tokens, D) -> 取 CLS
            feats = feats[:, 0]
        # 归一化到单位球面
        norm = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.clip(norm, 1e-8, None)
        chunks.append(feats)
    return np.concatenate(chunks, axis=0)


def info() -> dict:
    return {"backend": "onnx", "path": model_path(),
            "available": available(),
            "size_mb": (round(os.path.getsize(model_path()) / 1024 ** 2, 1)
                        if available() else None)}
