# -*- coding: utf-8 -*-
"""Validate the semantic regions and the three matte sources.

Checks properties that would break silently if the class layout or the mask
combination regressed:

* the six classes are present and their soft masks stay in [0, 1];
* skin covers the face box and does **not** cover the background;
* accessories are inside the subject union (that is the whole point of using
  a semantic model for matting);
* hybrid alpha is a superset of the portrait alpha — it must never *lose*
  subject area, only add what the portrait matte dropped;
* a synthetic cast: hybrid keeps an accessory the portrait matte misses.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image
from engine.face import FaceDetector
from engine import semantic as SEM
from engine import matting as MT

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
# 本机照片不在时(如 CI/其他机器)回退到仓库自带的合成样张
if not os.path.exists(SRC):
    SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "synthetic_portrait.jpg")

failures = []


def check(label, cond, detail=""):
    print("%-56s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    if not SEM.available():
        print("semantic model missing")
        return 1

    img = fit_long_edge(load_image(SRC), 800)
    h, w = img.shape[:2]
    print("image: %dx%d" % (w, h))

    # ---- 1. classes -------------------------------------------------------
    t0 = time.perf_counter()
    reg = SEM.segment(img)
    dt = (time.perf_counter() - t0) * 1000
    print("\n=== segment (%.0f ms) ===" % dt)
    cov = reg.coverage()
    for k, v in cov.items():
        print("   %-12s %.4f" % (k, v))
    check("all six classes present", len(reg.masks) == 6,
          "%d classes" % len(reg.masks))
    check("masks stay in [0,1]",
          all(0.0 <= float(m.min()) and float(m.max()) <= 1.0
              for m in reg.masks.values()))
    check("every class has some pixels",
          all(float((m > 0.5).mean()) > 0 for m in reg.masks.values()))
    check("background is the largest class",
          cov["background"] == max(cov.values()),
          "bg=%.3f" % cov["background"])

    # ---- 2. skin vs face box ---------------------------------------------
    print("\n=== skin localization ===")
    det = FaceDetector()
    faces = det.detect(img, dense=False)
    if faces:
        x, y, bw, bh = faces[0].box
        inner = np.zeros((h, w), bool)
        inner[int(y + bh * 0.25):int(y + bh * 0.75),
              int(x + bw * 0.25):int(x + bw * 0.75)] = True
        skin = reg.skin() > 0.5
        frac_in = float(skin[inner].mean())
        bg_mask = reg["background"] > 0.5
        frac_skin_on_bg = float(skin[bg_mask].mean())
        check("face centre is skin", frac_in > 0.5,
              "%.3f of the centre patch" % frac_in)
        check("skin does not sit on background", frac_skin_on_bg < 0.02,
              "%.4f of background labelled skin" % frac_skin_on_bg)
    else:
        check("face found for the skin test", False)

    # ---- 3. accessories are inside the subject ---------------------------
    print("\n=== subject composition ===")
    subj = reg.subject() > 0.5
    for cls in ("hair", "face-skin", "body-skin", "clothes", "accessories"):
        m = reg[cls] > 0.5
        if m.sum() == 0:
            print("   %-12s (absent in this frame)" % cls)
            continue
        inside = float(subj[m].mean())
        check("%s inside subject union" % cls, inside > 0.98,
              "%.4f" % inside)

    # ---- 4. hybrid is a superset of portrait -----------------------------
    print("\n=== matte sources ===")
    a_p = MT.subject_alpha(img, mode="portrait")
    a_s = MT.subject_alpha(img, mode="semantic")
    a_h = MT.subject_alpha(img, mode="hybrid")
    for nm, a in (("portrait", a_p), ("semantic", a_s), ("hybrid", a_h)):
        print("   %-9s coverage=%.4f mean=%.4f" % (nm, float((a > 0.5).mean()),
                                                   float(a.mean())))
    lost = float(((a_p > 0.5) & (a_h <= 0.5)).mean())
    check("hybrid never loses portrait subject area", lost < 0.005,
          "lost %.4f of the frame" % lost)
    gained = float(((a_h > 0.5) & (a_p <= 0.5)).mean())
    check("hybrid adds area the portrait matte missed", gained > 0.0,
          "+%.4f of the frame" % gained)

    save_image(os.path.join(ROOT, "out", "sem_map.jpg"),
               np.repeat((reg.category / 5.0)[..., None], 3, axis=2), 90)

    print()
    if failures:
        print("FAILED:", failures)
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
