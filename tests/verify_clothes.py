# -*- coding: utf-8 -*-
"""衣物四项（去褶皱/去污/版型/染色）端到端."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAIL = []


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def synth():
    """灰布（带褶皱条纹 + 污点）+ 非衣物区（强纹理，不应被改动）."""
    import cv2

    rng = np.random.default_rng(11)
    h, w = 300, 400
    img = np.full((h, w, 3), 0.55, np.float32)
    # 褶皱：几条宽缓亮暗带（低频）
    yy = np.linspace(0, 1, h)[:, None]
    folds = 0.10 * np.sin(yy * 18) * np.exp(-((xx := np.linspace(0, 1, w)[None, :]) - 0.5) ** 2 / 0.3)
    img += folds[..., None]
    # 污点：一小块深色
    img[120:145, 150:190] *= 0.45
    # 非衣物区：右侧强纹理块（不应被褶皱平滑吃掉）
    tx = img[:, 300:400].copy()
    img[:, 300:400] = rng.random((h, 100, 3)).astype(np.float32) * 0.6 + 0.2
    return np.clip(img, 0, 1)


def main():
    from engine import portrait as PT
    from engine.imgutil import rgb_to_lab, lab_to_rgb

    img = synth()
    h, w = img.shape[:2]
    # 掩膜：左 3/4 是衣物
    m = np.zeros((h, w), np.float32)
    m[:, :300] = 1.0

    # 1) 空参数 = 恒等
    same = PT.clothes(img, m, 0, 0, 0, 0)
    check("空参数恒等", float(np.abs(same - img).max()) < 1e-6)

    # 2) 去褶皱：低频形变被压平，缝（陡边）保留
    wr = PT.clothes(img, m, 80, 0, 0, 0)
    # 亮暗带振幅（用左半区亮度 std 衡量）
    std0 = float(img[:, :300].std())
    std1 = float(wr[:, :300].std())
    check("去褶皱压平明暗带", std1 < std0 * 0.9, "%.3f -> %.3f" % (std0, std1))
    # 掩膜外不动
    d_out = float(np.abs(wr[:, 350:] - img[:, 350:]).mean())
    check("掩膜外不动（褶皱）", d_out < 0.001, "%.4f" % d_out)

    # 3) 去污点：污区变亮贴近周围
    st = PT.clothes(img, m, 0, 70, 0, 0)
    stain_before = float(img[132, 170].mean())
    stain_after = float(st[132, 170].mean())
    check("污点被提亮", stain_after > stain_before + 0.05,
          "%.3f -> %.3f" % (stain_before, stain_after))

    # 4) 版型强化：掩膜**边界带**的能量提升（unsharp 只增强边缘，
    #    测整片掩膜区会被大片平坦布面稀释到看不见）
    st2 = PT.clothes(img, m, 0, 0, 60, 0)
    band = np.zeros((h, w), bool)
    band[0:6, :300] = True; band[294:300, :300] = True
    band[:, 294:300] = True; band[:, 0:6] = True
    e1 = float(np.abs(np.diff(st2[band].mean(axis=1))).mean())
    e0 = float(np.abs(np.diff(img[band].mean(axis=1))).mean())
    check("版型强化提升边界带能量", e1 > e0 * 1.05,
          "%.4f -> %.4f" % (e0, e1))

    # 5) 染色：a 通道向目标色移动、明度基本不动（本库 Lab 量纲 a/b≈±1，
    #    灰布 a≈0、红 a≈+0.6，80% 混合后 a 应显著为正）
    tinted = PT.clothes(img, m, 0, 0, 0, 80, (0.80, 0.20, 0.25))
    lab0 = rgb_to_lab(img[:, :300])
    lab1 = rgb_to_lab(tinted[:, :300])
    a0 = float(lab0[..., 1].mean()); a1 = float(lab1[..., 1].mean())
    check("染色 a 通道向红移动", a1 > a0 + 0.15, "%.2f -> %.2f" % (a0, a1))
    l0 = float(lab0[..., 0].mean()); l1 = float(lab1[..., 0].mean())
    check("染色保明度", abs(l1 - l0) < 0.08, "%.2f vs %.2f" % (l0, l1))

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
