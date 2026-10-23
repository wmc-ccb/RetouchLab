# -*- coding: utf-8 -*-
"""路线 A + B 对照图：几何遮罩 / 妆容 / 表情，以及 5 点 vs 106 点。

同时验证：106 点修正归一化后是否能给出可用于遮罩的几何（路线 B 的可行性）。
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image
from engine.face import FaceDetector
from engine import geoface as GEO
from engine import portrait as PT
from engine import landmarks as LMK

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
OUT = os.path.join(ROOT, "out")


def lab(im, text):
    u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
    bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 34), (18, 18, 20), -1)
    cv2.putText(bgr, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (240, 240, 240), 1, cv2.LINE_AA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def face_crop(img, lm5, pad=2.0):
    iod = GEO._iod(lm5)
    mid = (lm5[0] + lm5[1]) / 2
    cx, cy = mid
    half = iod * pad
    x0 = int(max(0, cx - half)); x1 = int(min(img.shape[1], cx + half))
    y0 = int(max(0, cy - half * 0.8))
    y1 = int(min(img.shape[0], cy + half * 1.9))
    return x0, y0, x1, y1


def main():
    img = fit_long_edge(load_image(SRC), 900)
    h, w = img.shape[:2]
    f = FaceDetector(dense=False).detect(img)[0]
    lm5 = np.asarray(f.landmarks, np.float32)
    x, y, bw, bh = [float(v) for v in f.box]
    pts106 = LMK.Landmark106().predict(
        img, np.asarray([x, y, x + bw, y + bh], np.float32)).points

    x0, y0, x1, y1 = face_crop(img, lm5)
    def cut(im):
        return im[y0:y1, x0:x1]

    masks = GEO.region_masks(img, lm5)
    vis = img.copy()
    pal = {"eye_r": (1, .2, .2), "eye_l": (1, .2, .2), "brows": (.3, .75, 1),
           "lips": (1, .25, .8), "blush": (1, .75, .2), "nose": (.35, 1, .35)}
    for k, c in pal.items():
        a = masks[k][..., None] * 0.62
        vis = vis * (1 - a) + np.array(c, np.float32) * a
    vis = np.clip(vis, 0, 1)

    mk = PT.makeup(img, lm5, None, lip=85, lip_tone="rose",
                   blush=60, blush_tone="coral",
                   shadow=45, shadow_tone="nude", brow=45, brow_tone="brick")
    ex = PT.expression(img, lm5, smile=65)

    lmvis = img.copy()
    for p in pts106:
        cv2.circle(lmvis, (int(p[0]), int(p[1])), 2, (1.0, .25, .2), -1)
    for p in lm5:
        cv2.circle(lmvis, (int(p[0]), int(p[1])), 4, (0.2, 1.0, 0.35), -1)
    lmvis = np.clip(lmvis, 0, 1)

    panels = [lab(cut(img), "1 original"),
              lab(cut(vis), "2 geometric masks (A)"),
              lab(cut(lmvis), "3 5pt(green) + 106pt(red)"),
              lab(cut(mk), "4 makeup"),
              lab(cut(ex), "5 expression smile")]
    ph = min(p.shape[0] for p in panels)
    pw = min(p.shape[1] for p in panels)
    panels = [p[:ph, :pw] for p in panels]
    grid = np.hstack(panels)
    save_image(os.path.join(OUT, "geoface_ab.jpg"), grid, 92)
    print("saved out/geoface_ab.jpg", grid.shape)

    # 定量：几何遮罩 vs 两眼中心
    iod = GEO._iod(lm5)
    for k in ("eye_r", "eye_l", "lips", "brows", "blush", "nose"):
        m = masks[k] > 0.5
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            print("  %-7s 空" % k)
            continue
        print("  %-7s 覆盖%.3f  中心(%4.0f,%4.0f)" % (
            k, float(m.mean()), xs.mean(), ys.mean()))
    print("\n5点: 右眼%s 左眼%s 鼻%s 嘴角%s/%s" % tuple(
        np.round(p).astype(int) for p in lm5))


if __name__ == "__main__":
    main()
