# -*- coding: utf-8 -*-
"""手足美化 / 沙滩美化 / 祛双眼皮贴 的回归。

覆盖 2026-10-30 功能冲刺的三个新模块：
- engine/handfoot.py  HandLandmarker 手部检测（官方测试图是基准）
- engine/beach.py     沙面识别 + 祛杂物 + 平整
- engine/skin.eyelid_tape_soften  眼睑条带高频衰减
- pipeline 集成：新参数在 DEFAULTS、管线阶段可跑、掩膜外不变量

无手模型文件时手部段 SKIPPED（其余照跑）。
"""
import sys

import numpy as np

sys.path.insert(0, ".")

results = []


def check(label, cond, detail=""):
    print("%-44s %s %s" % (label, "OK  " if cond else "FAIL", detail))
    results.append(bool(cond))


def main():
    from engine.imgutil import load_image
    from engine import handfoot as HF, beach as BH
    from engine.skin import eyelid_tape_soften
    from engine.face import Face
    from engine.pipeline import DEFAULTS, RetouchPipeline

    # ---- DEFAULTS / spec ----
    for k in ("hand_smooth", "hand_tone", "foot_smooth", "foot_tone",
              "shoe_clean", "eyelid_tape",
              "beach_debris", "beach_flatten", "beach_warm"):
        check("DEFAULTS 含 %s" % k, k in DEFAULTS)

    # ---- 手部: 官方测试图是 ground truth（存在才测）----
    import os
    ref = "out/_hand_thumb_up.jpg"
    if not os.path.exists(ref):
        print("SKIP: 无官方手部测试图（out/_hand_thumb_up.jpg）")
    elif not HF.available():
        print("SKIP: 无 hand_landmarker.task")
    else:
        img = load_image(ref)
        hands = HF.detect_hands(img)
        check("官方图检出 ≥1 手", len(hands) >= 1, str(len(hands)))
        if hands:
            m = HF.hand_mask(img, hands)
            check("手掩膜覆盖 3-60%", 0.03 < m.mean() < 0.6,
                  "%.3f" % m.mean())
            out, n = HF.beautify_hands(img, smooth=0.7, tone=0.5)
            delta_out = float(abs((out - img) * (m[..., None] > 0.15)).mean())
            delta_outside = float(abs(
                (out - img) * (m[..., None] <= 0.02)).max())
            check("掩膜内有变化", delta_out > 1e-4, "%.5f" % delta_out)
            check("掩膜外逐像素不变", delta_outside < 2e-3,
                  "%.6f" % delta_outside)

    # ---- 沙滩: 合成场景（低噪声沙面 + 烟头）----
    h, w = 400, 600
    rng = np.random.default_rng(7)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    base = np.clip(0.62 + 0.18 * t, 0, 1)
    noise = rng.normal(0, 0.02, (h, w)).astype(np.float32)
    sand = np.dstack([np.clip(base + noise, 0, 1),
                      np.clip(base * 0.86 + noise, 0, 1),
                      np.clip(base * 0.66 + noise, 0, 1)])[150:]
    sky = np.zeros((150, w, 3), np.float32)
    sky[..., 0] = 0.55; sky[..., 1] = 0.72; sky[..., 2] = 0.95
    scene = np.vstack([sky, sand])
    scene[260:268, 300:308] = 0.12              # 烟头 8x12
    out, info = BH.beautify(scene, debris=0.7, flatten=0.5)
    check("沙面识别 >40%", info.get("sand_coverage", 0) > 0.4,
          str(info))
    d = float(abs(out[260:268, 300:308]
                  - scene[260:268, 300:308]).mean())
    check("烟头被清除", d > 0.03, "%.3f" % d)
    # 天空不应被沙滩管线污染
    dsky = float(abs(out[10:60] - scene[10:60]).mean())
    check("天空不受影响", dsky < 1e-3, "%.5f" % dsky)

    # ---- 祛双眼皮贴: 合成脸 + 贴纸线 ----
    face_img = np.full((300, 400, 3), 0.72, np.float32)
    face_img[130:150, 150:250] = 0.55           # 眼
    face_img[120:123, 155:245] = 0.25           # 贴纸高对比线
    face = Face(box=(100, 80, 200, 180), score=1.0,
                landmarks=np.array([[160, 140], [240, 140], [200, 180],
                                    [180, 210], [220, 210]], np.float32))
    face.eyes = [(150.0, 140.0, 100.0, 20.0)]
    out = eyelid_tape_soften(face_img, face, 80)
    lb = float(abs(face_img[120:123, 155:245] - 0.72).mean())
    la = float(abs(out[120:123, 155:245] - 0.72).mean())
    check("贴纸线对比衰减", la < lb * 0.9, "%.3f -> %.3f" % (lb, la))
    check("眼部低频保留", float(abs(out[135:145, 160:240]
                                   - face_img[135:145, 160:240]).mean())
          < 0.05)

    # ---- pipeline 集成: 新参数可跑、无手画面安静跳过 ----
    pipe = RetouchPipeline()
    img2 = np.full((300, 400, 3), 0.6, np.float32) + \
        np.random.default_rng(1).normal(0, 0.04, (300, 400, 3)).astype(np.float32)
    out2, info = pipe.run(img2, {"hand_smooth": 60, "foot_tone": 40,
                                 "shoe_clean": 40, "beach_debris": 50,
                                 "beach_flatten": 40}, detect=False)
    check("管线跑通新阶段", "hands_feet" in info["applied"]
          and "beach" in info["applied"], str(info["applied"]))

    n_pass = sum(results)
    print("\n%d/%d 通过" % (n_pass, len(results)))
    if n_pass == len(results):
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
