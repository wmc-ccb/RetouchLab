# -*- coding: utf-8 -*-
"""生成质量严格审视：软度失配 / 纹理合理性 / 大区域表现."""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def sharpness(img):
    """Laplacian 方差：锐度的常用代理。"""
    import cv2
    from engine.imgutil import luminance
    lum = (luminance(img) * 255).astype(np.uint8)
    return float(cv2.Laplacian(lum, cv2.CV_64F).var())


def main():
    from engine.imgutil import load_image, fit_long_edge, save_image
    from engine import gentools as GT
    import cv2

    img = fit_long_edge(load_image(SRC), 900)
    h, w = img.shape[:2]

    def lab(x, t):
        u8 = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 测试 1：大区域背景替换（最常见用例，也是最考验质量的）
    out1, i1 = GT.ai_backdrop(img, "plain white studio backdrop, "
                              "professional photo studio", steps=22, seed=42)
    # 生成区 vs 原图的锐度比（软度失配的直接度量）
    # 生成区 = 背景掩膜；用 subject 补集近似（再跑一次太贵，用中心带）
    band = img[int(h * 0.1):int(h * 0.4)]
    band2 = out1[int(h * 0.1):int(h * 0.4)]
    s0, s1 = sharpness(band), sharpness(band2)
    print("测试1 AI布景 %.0fs" % i1["seconds"])
    print("  上带锐度: 原 %.0f -> 生成 %.0f  (比率 %.2f, <0.5 即明显偏软)"
          % (s0, s1, s1 / max(s0, 1e-6)))

    # 测试 2：具体纹理（木地板——结构性强，最能暴露 SD1.5 的纹理弱点）
    m2 = np.zeros((h, w), np.float32)
    m2[int(h * 0.72):, :] = 1.0
    out2, i2 = GT.generative_fill(img, m2, "wooden floor planks, "
                                  "natural wood texture, realistic",
                                  steps=22, seed=7)
    dm = float(np.abs(out2 - img)[m2 > 0.5].mean())
    print("测试2 木地板 %.0fs  mask内变化 %.3f" % (i2["seconds"], dm))

    # 拼图审视
    save_image(os.path.join(OUT, "_genq1.jpg"),
               np.hstack([lab(img, "original"),
                          lab(out1, "AI backdrop: white studio")]), 90)
    save_image(os.path.join(OUT, "_genq2.jpg"),
               np.hstack([lab(img, "original"),
                          lab(out2, "fill: wooden floor")]), 90)

    # 保存裁切特写（生成区放大看细节）
    crop1 = out1[int(h * 0.05):int(h * 0.45), int(w * 0.05):int(w * 0.55)]
    save_image(os.path.join(OUT, "_genq1_crop.jpg"), crop1, 92)
    crop2 = out2[int(h * 0.70):, :]
    save_image(os.path.join(OUT, "_genq2_crop.jpg"), crop2, 92)
    print("saved _genq1.jpg/_genq2.jpg + crops")
    return 0


if __name__ == "__main__":
    sys.exit(main())
