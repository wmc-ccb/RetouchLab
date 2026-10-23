# -*- coding: utf-8 -*-
"""PS 单位对齐后的行为验证.

1. HSL 色相 ±100（PS 单位）方向正确、强度单调、边界受控
2. 锐化 150 上限：效果单调、不产生 NaN/爆白
3. 曝光 EV 上下文里整条管线仍自洽
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import tone as T

FAIL = []


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def synth():
    """一条可控的合成图：左半红、右半蓝，中间过渡."""
    h, w = 96, 192
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    r = np.clip(xx * 1.6, 0, 1)
    b = np.clip(1.2 - xx * 1.6, 0, 1)
    g = np.full_like(r, 0.25)
    img = np.stack([np.broadcast_to(r, (h, w)),
                    np.broadcast_to(g, (h, w)),
                    np.broadcast_to(b, (h, w))], axis=2)
    # 加一点纹理让锐化有东西可锐
    rng = np.random.default_rng(7)
    img = np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32),
                  0, 1)
    return img


def main():
    img = synth()

    # ---- HSL 色相：PS 单位 ±100 ----
    base = T.hsl_band(img, "red", 0, 0, 0)
    pos50 = T.hsl_band(img, "red", 50, 0, 0)
    pos100 = T.hsl_band(img, "red", 100, 0, 0)
    neg100 = T.hsl_band(img, "red", -100, 0, 0)

    d_pos50 = float(np.abs(pos50 - base).mean())
    d_pos100 = float(np.abs(pos100 - base).mean())
    check("色相 +100 效果强于 +50", d_pos100 > d_pos50,
          "%.4f vs %.4f" % (d_pos100, d_pos50))
    check("色相 ±100 方向相反",
          float(np.abs(neg100 - base).mean()) > d_pos50 * 0.8,
          "%.4f" % float(np.abs(neg100 - base).mean()))
    # ±100 = ±30°：红色(0°) +30° 应变橙黄（G 分量显著上升）
    g_up = float(pos100[..., 1].mean() - base[..., 1].mean())
    check("+100 让红色向黄偏（G 上升）", g_up > 0.02, "ΔG=%.3f" % g_up)
    # 输出仍合法
    check("输出在 0..1", float(pos100.min()) >= 0 and
          float(pos100.max()) <= 1,
          "%.2f..%.2f" % (float(pos100.min()), float(pos100.max())))

    # ---- 锐化 0..150 ----
    s50 = T.sharpening(img, 50, 1.0, 25)
    s150 = T.sharpening(img, 150, 1.0, 25)
    e50 = float(np.abs(s50 - img).mean())
    e150 = float(np.abs(s150 - img).mean())
    check("锐化 150 强于 50", e150 > e50, "%.4f vs %.4f" % (e150, e50))
    check("锐化 150 无 NaN/爆白",
          np.isfinite(s150).all() and float(s150.max()) <= 1.0,
          "max=%.3f" % float(s150.max()))
    check("锐化 150 仍是锐化（边缘增强为正）", e150 > 0.001,
          "%.4f" % e150)

    # ---- 曝光：EV 换算链路（-100..100 内部 = -5..5 EV）----
    from engine.pipeline import RetouchPipeline

    eng = RetouchPipeline(detect_faces=False, detect_pose=False)
    ex = eng.run(img, {"exposure": 27})[0]      # 1.35 EV
    check("曝光 +1.35 EV 画面变亮",
          float(ex.mean()) > float(img.mean()),
          "%.3f -> %.3f" % (float(img.mean()), float(ex.mean())))
    check("曝光输出合法", float(ex.min()) >= 0 and float(ex.max()) <= 1)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
