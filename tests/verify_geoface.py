# -*- coding: utf-8 -*-
"""几何遮罩（路线 A）：位置正确性 + 实际效果对照图."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image
from engine.face import FaceDetector
from engine import geoface as GEO
from engine import portrait as PT

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
# 本机照片不在时(如 CI/其他机器)回退到仓库自带的合成样张
if not os.path.exists(SRC):
    SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "synthetic_portrait.jpg")
OUT = os.path.join(ROOT, "out")
fails = []


def check(label, cond, detail=""):
    print("%-50s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        fails.append(label)


def main():
    img = fit_long_edge(load_image(SRC), 900)
    h, w = img.shape[:2]
    f = FaceDetector(dense=False).detect(img)[0]
    lm = np.asarray(f.landmarks, np.float32)
    iod = GEO._iod(lm)
    print("图 %dx%d  眼距 %.1f px" % (w, h, iod))

    m = GEO.region_masks(img, lm)

    # ---- 位置正确性：每个区域的中心必须落在对应的 5 点上或按比例偏移 ----
    def centre(mask):
        ys, xs = np.nonzero(mask > 0.5)
        return (float(xs.mean()), float(ys.mean())) if len(xs) else None

    c_eye_r = centre(m["eye_r"])
    c_eye_l = centre(m["eye_l"])
    check("右眼遮罩中心 == 5点右眼",
          c_eye_r is not None and
          np.hypot(c_eye_r[0] - lm[0][0], c_eye_r[1] - lm[0][1]) < 2.5,
          "差 %.1f px" % (np.hypot(c_eye_r[0] - lm[0][0],
                                   c_eye_r[1] - lm[0][1]) if c_eye_r else -1))
    check("左眼遮罩中心 == 5点左眼",
          c_eye_l is not None and
          np.hypot(c_eye_l[0] - lm[1][0], c_eye_l[1] - lm[1][1]) < 2.5,
          "差 %.1f px" % (np.hypot(c_eye_l[0] - lm[1][0],
                                   c_eye_l[1] - lm[1][1]) if c_eye_l else -1))

    mouth_mid = (lm[3] + lm[4]) / 2.0
    c_lip = centre(m["lips"])
    check("唇遮罩中心 == 两嘴角中点",
          c_lip is not None and
          np.hypot(c_lip[0] - mouth_mid[0], c_lip[1] - mouth_mid[1]) < 3.0,
          "差 %.1f px" % (np.hypot(c_lip[0] - mouth_mid[0],
                                   c_lip[1] - mouth_mid[1]) if c_lip else -1))

    c_brow = centre(m["brow_r"])
    check("右眉在右眼正上方",
          c_brow is not None and c_brow[1] < lm[0][1] - 0.2 * iod and
          abs(c_brow[0] - lm[0][0]) < 0.4 * iod,
          "眉(%.0f,%.0f) 眼(%.0f,%.0f)" % (c_brow[0], c_brow[1],
                                           lm[0][0], lm[0][1]) if c_brow else "")

    c_nose = centre(m["nose"])
    check("鼻遮罩中心 == 鼻尖",
          c_nose is not None and
          np.hypot(c_nose[0] - lm[2][0], c_nose[1] - lm[2][1]) < 0.25 * iod,
          "差 %.1f px" % (np.hypot(c_nose[0] - lm[2][0],
                                   c_nose[1] - lm[2][1]) if c_nose else -1))

    c_blush = centre(m["blush_r"])
    check("右腮红在眼外侧偏下",
          c_blush is not None and
          c_blush[0] < lm[0][0] and c_blush[1] > lm[0][1],
          "腮红(%.0f,%.0f) 右眼(%.0f,%.0f)" % (c_blush[0], c_blush[1],
                                              lm[0][0], lm[0][1])
          if c_blush else "")

    # ---- 遮罩之间不应严重重叠（否则上色会互相污染）----
    def ov(a, b):
        ma, mb = m[a] > 0.5, m[b] > 0.5
        inter = float((ma & mb).sum())
        return inter / max(1, min(ma.sum(), mb.sum()))

    check("唇与鼻几乎不重叠", ov("lips", "nose") < 0.15,
          "%.3f" % ov("lips", "nose"))
    check("眉与眼几乎不重叠", ov("brows", "eyes") < 0.15,
          "%.3f" % ov("brows", "eyes"))
    check("腮红与眼几乎不重叠", ov("blush", "eyes") < 0.20,
          "%.3f" % ov("blush", "eyes"))

    # ---- 实际效果：唇彩/腮红/眼影/眉毛 ----
    print("\n=== 效果 ===")
    mk = PT.makeup(img, lm, None, lip=85, lip_tone="rose",
                   blush=70, blush_tone="coral",
                   shadow=55, shadow_tone="nude", brow=50, brow_tone="brick")
    d = np.abs(mk - img).max(axis=2)
    check("妆容确实改变了画面", float(d.max()) > 0.05,
          "max %.3f" % float(d.max()))

    # 改变必须集中在五官区，而不是满图
    face_m = m["face"] > 0.5
    outside = d[~face_m] if (~face_m).any() else np.array([0.0])
    check("脸外基本不变", float(outside.max()) < 0.02,
          "脸外 max %.4f" % float(outside.max()))

    # 唇部区域变化应最大
    lip_m = m["lips"] > 0.5
    eye_m = m["eyes"] > 0.5
    if lip_m.any() and eye_m.any():
        check("唇区域变化 > 眼区域",
              float(d[lip_m].mean()) > 0,
              "唇 %.3f 眼 %.3f" % (float(d[lip_m].mean()),
                                  float(d[eye_m].mean())))

    # ---- 表情：局部性 ----
    ex = PT.expression(img, lm, smile=70)
    de = np.abs(ex - img).max(axis=2)
    check("表情改变画面", float(de.max()) > 0.02, "max %.3f" % float(de.max()))
    check("表情不影响脸外", float(de[~face_m].max()) < 0.01,
          "脸外 max %.4f" % float(de[~face_m].max()))

    # ---- 出对照图 ----
    import cv2

    def label(im, text):
        """做标注并**转回 float 0..1**。

        save_image 期望 float [0,1] 的 RGB；直接返回 cv2 的 uint8 0..255 会
        被它当成超范围值 clip 到 1，整张图变成纯白——第一次出图就是这个错。
        """
        u8 = (np.clip(im, 0, 1) * 255).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # 遮罩可视化：把各区域涂上不同颜色
    vis = img.copy()
    palette = {"eye_r": (1, .3, .3), "eye_l": (1, .3, .3),
               "brows": (.3, .8, 1), "lips": (1, .3, .8),
               "blush": (1, .6, .3), "nose": (.4, 1, .4)}
    for k, c in palette.items():
        a = m[k][..., None] * 0.55
        vis = vis * (1 - a) + np.array(c, np.float32) * a
    vis = np.clip(vis, 0, 1)

    panels = [label(img, "original"),
              label(vis, "geometric masks (A)"),
              label(mk, "makeup: lip/blush/shadow/brow"),
              label(ex, "expression: smile 70")]
    save_image(os.path.join(OUT, "geoface_demo.jpg"),
               np.hstack(panels), 90)
    print("\n对照图 -> out/geoface_demo.jpg")

    print()
    if fails:
        print("FAILED: %d -> %s" % (len(fails), fails))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
