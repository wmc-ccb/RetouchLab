# -*- coding: utf-8 -*-
"""Super-resolution and LaMa removal.

Both models are large downloads, so each half skips cleanly when its weights are
absent — a missing 200 MB file should not read as a code failure.

What is actually checked:

* **super-resolution** really increases resolution by the advertised factor,
  changes the image (it is not a no-op) and **does not change colour** — a
  model that shifts hue or clips highlights is worse than useless on a
  deliverable;
* tiles are stitched without a seam: the result must be **stable under a
  change of tile size**, because a seam shows up as a difference between the
  two runs;
* **LaMa** fills exactly the masked area and leaves the rest bit-identical;
* what it fills with **continues the surroundings** rather than averaging to
  flat grey — measured as the filled region's texture energy against the
  surrounding background;
* an empty mask is a no-op.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge, save_image, luminance
from engine import superres as SR
from engine import lama as LA

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
# 本机照片不在时(如 CI/其他机器)回退到仓库自带的合成样张
if not os.path.exists(SRC):
    SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples", "synthetic_portrait.jpg")
failures = []


def check(label, cond, detail=""):
    print("%-54s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def texture_energy(a: np.ndarray) -> float:
    """Mean |laplacian| — how much structure a patch carries."""
    import cv2
    g = (np.clip(luminance(a), 0, 1) * 255).astype(np.uint8)
    return float(np.abs(cv2.Laplacian(g, cv2.CV_32F)).mean())


def main():
    print("=== 模型可用性 ===")
    print("  超分:", SR.info())
    li = LA.info()
    print("  消除: available=%s path=%s" % (li["available"], li["path"]))
    if "interface" in li:
        print("        输入:", li["interface"]["names"], li["interface"]["shapes"])

    # ---- super-resolution ------------------------------------------------
    if SR.available():
        print("\n=== 超分 ===")
        img = fit_long_edge(load_image(SRC), 220)
        h, w = img.shape[:2]
        print("  输入 %dx%d" % (w, h))
        import time
        t0 = time.perf_counter()
        up = SR.upscale(img, tile=192, overlap=32)
        dt = time.perf_counter() - t0
        oh, ow = up.shape[:2]
        check("输出放大 4 倍",
              abs(oh - h * 4) <= 4 and abs(ow - w * 4) <= 4,
              "%dx%d -> %dx%d" % (w, h, ow, oh))
        check("耗时合理", dt < 180, "%.1f 秒" % dt)
        check("结果被裁剪到 [0,1]",
              float(up.min()) >= 0.0 and float(up.max()) <= 1.0)

        up2 = fit_long_edge(up, w)
        d = np.abs(up2 - img)
        check("确实改变了画面（不是原样放大）", float(d.mean()) > 0.001,
              "mean %.5f" % float(d.mean()))

        # colour must be preserved: compare channel means and the cast between
        # the input and its downscaled output
        m_in = img.reshape(-1, 3).mean(axis=0)
        m_out = up2.reshape(-1, 3).mean(axis=0)
        rel = float(np.abs(m_out / np.maximum(m_in, 1e-6) - 1.0).max())
        check("通道均值基本保持（不偏色）", rel < 0.12,
              "最大偏差 %.1f%%" % (100 * rel))
        check("没有大面积裁切", float((up >= 0.999).mean()) < 0.05,
              "%.2f%% 像素到顶" % (100 * float((up >= 0.999).mean())))

        # tiling stability: a different tile size must give a nearly identical
        # result, otherwise the seams are in the output
        up_b = SR.upscale(img, tile=128, overlap=32)
        same = float(np.abs(up - up_b).mean())
        check("换 tile 尺寸结果稳定（无接缝）", same < 0.01,
              "平均差 %.5f" % same)
        save_image(os.path.join(ROOT, "out", "sr_demo.jpg"),
                   np.hstack([cv2_resize(img, oh), up]), 92)
    else:
        print("\n超分模型未下载，跳过")

    # ---- LaMa ------------------------------------------------------------
    if LA.available():
        print("\n=== LaMa 消除 ===")
        img = fit_long_edge(load_image(SRC), 700)
        h, w = img.shape[:2]
        m = np.zeros((h, w), np.float32)
        # a rectangle over the backdrop, well away from the subject
        x0, y0 = int(w * 0.04), int(h * 0.30)
        m[y0:y0 + int(h * 0.18), x0:x0 + int(w * 0.10)] = 1.0
        out, info = LA.fill(img, m)
        check("返回图像", out.shape == img.shape, str(out.shape))
        check("报告移除像素数", info["removed_px"] > 0, str(info["removed_px"]))

        d = np.abs(out - img).max(axis=2)
        inside = m > 0.5
        # "outside" must be the region the call is *allowed* to touch: the
        # dilated mask plus the feather's spread. Using the raw mask counted
        # the feather ring as a leak (0.9% of the frame, bounded exactly by the
        # mask bbox plus the dilation radius), which is the algorithm working
        # as specified rather than a bug.
        touched = info.get("touched")
        if touched is None:
            touched = inside
        outside = ~touched
        check("生效区域之外逐位不变", float(d[outside].max()) == 0.0,
              "max %.2e（%.1f%% 像素）"
              % (float(d[outside].max()), 100 * outside.mean()))
        check("生效区域确实扩大（膨胀+羽化）",
              int(touched.sum()) > int(inside.sum()),
              "%d -> %d 像素" % (int(inside.sum()), int(touched.sum())))
        check("掩膜内确实重绘", float(d[inside].min()) > 0.0
              and float(d[inside].max()) > 0.02,
              "max %.4f" % float(d[inside].max()))
        # the fill must not be a flat wash: LaMa that receives a contradicting
        # mask answers with pure white, which is what this catches
        filled_vals = out[inside]
        check("填补不是纯白/纯色（掩膜约定正确）",
              float(filled_vals.max()) < 0.999 and float(filled_vals.std()) > 0.02,
              "max %.3f std %.3f" % (float(filled_vals.max()),
                                     float(filled_vals.std())))

        # the fill must continue the surroundings, not average to flat
        filled = out[y0:y0 + int(h * 0.18), x0:x0 + int(w * 0.10)]
        ring = out[max(0, y0 - 40):y0, x0:x0 + int(w * 0.10)]
        if ring.size:
            ef, er = texture_energy(filled), texture_energy(ring)
            check("填补区保留了背景纹理（非平涂）", ef > er * 0.15,
                  "填补 %.2f vs 周围 %.2f" % (ef, er))
        # and it should not be a copy of something else in the frame
        check("填补不是简单复制（与原内容不同）",
              float(np.abs(out[inside] - img[inside]).mean()) > 0.01,
              "平均差 %.4f" % float(np.abs(out[inside] - img[inside]).mean()))

        empty, einfo = LA.fill(img, np.zeros((h, w), np.float32))
        check("空掩膜 = 零操作", float(np.abs(empty - img).max()) == 0.0,
              str(einfo.get("skipped")))

        save_image(os.path.join(ROOT, "out", "lama_demo.jpg"),
                   np.hstack([img, out]), 92)
    else:
        print("\nLaMa 模型未下载，跳过")

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


def cv2_resize(a, target_h):
    import cv2
    s = target_h / float(a.shape[0])
    return cv2.resize(a, (int(a.shape[1] * s), target_h),
                      interpolation=cv2.INTER_CUBIC)


if __name__ == "__main__":
    sys.exit(main())
