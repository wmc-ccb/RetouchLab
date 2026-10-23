# -*- coding: utf-8 -*-
"""闭眼检测：合成场景有真值 + 真实照片链路."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import eyes as E
from engine import library as L

FAIL = []


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def synth_face(open_eyes: bool, seed: int = 0):
    """合成人脸：肤色底 + 一对眼睛（睁/闭）."""
    import cv2

    rng = np.random.default_rng(seed)
    h, w = 300, 240
    img = np.ones((h, w, 3), np.float32) * np.array(
        [0.85, 0.72, 0.64], np.float32)
    # 轻微噪声，避免完美纯色让梯度能量判定失真
    img += rng.normal(0, 0.012, img.shape).astype(np.float32)
    er, el = (95.0, 120.0), (145.0, 120.0)
    for (cx, cy) in (er, el):
        cx, cy = int(cx), int(cy)
        if open_eyes:
            cv2.ellipse(img, (cx, cy), (14, 9), 0, 0, 360,
                        (0.95, 0.95, 0.95), -1)
            cv2.circle(img, (cx, cy), 6, (0.08, 0.08, 0.10), -1)
        else:
            cv2.ellipse(img, (cx, cy), (14, 9), 0, 0, 360,
                        (0.80, 0.66, 0.58), -1)
            cv2.line(img, (cx - 13, cy), (cx + 13, cy),
                     (0.42, 0.33, 0.28), 2)
    lm = np.array([er, el, (120.0, 150.0), (105.0, 190.0), (135.0, 190.0)],
                  np.float32)
    return np.clip(img, 0, 1), lm


def main():
    # 1) 单帧：睁眼的两项指标都应高于闭眼
    o, lm = synth_face(True)
    c, _ = synth_face(False)
    so, sc = E.eye_scores(o, lm), E.eye_scores(c, lm)
    check("睁眼 open 高于闭眼", so["open"] > sc["open"],
          "%.3f vs %.3f" % (so["open"], sc["open"]))
    check("睁眼 edge 高于闭眼", so["edge"] > sc["edge"],
          "%.4f vs %.4f" % (so["edge"], sc["edge"]))

    # 2) 批次判定：混入闭眼帧应被挑出来
    imgs, lms, truth = [], [], []
    for i in range(8):
        closed = i in (2, 5)
        im, l = synth_face(not closed, seed=i)
        imgs.append(im); lms.append(l); truth.append(closed)
    res = E.detect_closed(imgs, lms)
    got = sorted(res["closed"])
    check("批次里挑出闭眼帧", len(got) == 2, "挑出 %s（真值 [2,5]）" % got)
    check("挑出的正是闭眼帧", got == [2, 5], str(got))

    # 3) 无 landmark 时不崩
    res2 = E.detect_closed(imgs[:2], [None, None])
    check("无关键点不崩", res2.get("closed") == [], str(res2.get("closed")))

    # 4) 真实照片链路：scan 应产出 closed 分组与 eye_open 指标
    p = os.path.join("M:" + os.sep, "场照精修")
    if os.path.isdir(p):
        r = L.scan(p, limit=8)
        check("scan 输出含 closed 分组", "closed" in r.get("groups", {}),
              str(r.get("groups")))
        withface = [it for it in r["items"] if (it.get("eye_open") or -1) >= 0]
        check("真人照片能算眼睛指标", len(withface) >= 1,
              "%d/%d 张" % (len(withface), r["count"]))
        if withface:
            vals = [it["eye_open"] for it in withface]
            check("eye_open 在合理区间 0~1",
                  all(0 <= v <= 1 for v in vals),
                  "%.2f~%.2f" % (min(vals), max(vals)))
    else:
        print("  跳过真实照片（M: 不可用）")

    # 5) info 描述
    inf = E.info()
    check("info 声明判据", "dark ratio" in str(inf.get("proxies")))

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
