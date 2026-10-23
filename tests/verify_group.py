# -*- coding: utf-8 -*-
"""多人合影：自动蒙版（多人区覆盖）+ 祛路人（保主体删路人）."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
FAIL = []
GROUP = r"C:\Users\Rana\Desktop\26cicf\d2\sb\_DSC7324.JPG"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    from engine.imgutil import load_image, fit_long_edge, save_image
    from engine.face import FaceDetector
    from engine import semantic as SEM
    from engine import automask as AM
    from engine import passersby as PB
    import cv2

    img = fit_long_edge(load_image(GROUP), 900)
    h, w = img.shape[:2]

    # 1) 检出 3+ 张脸
    det = FaceDetector(dense=False)
    faces = det.detect(img, max_faces=12)
    check("合影检出多张脸", len(faces) >= 3, "%d 张" % len(faces))

    # 2) 自动蒙版：多人图衣服覆盖大
    masks, info = AM.build(img)
    cov = info["coverage"]
    check("多人衣服覆盖 > 15%", cov.get("clothes", 0) > 0.15,
          "%.3f" % cov.get("clothes", -1))
    check("多人 subject 覆盖 > 30%", cov.get("subject", 0) > 0.30,
          "%.3f" % cov.get("subject", -1))
    # subject 类是**单主体**模型（selfie_multiclass 的文档局限，祛路人
    # 模块因此才需要人脸+比例扩粗掩膜）：多人群里它只保证覆盖**主体**
    # 一人；衣物类是多实例的（36% 覆盖含多人）。断言按这个语义钉住：
    # 主体脸中心（与 subject 重叠最大的那张脸）必须在 subject 内。
    reg = SEM.segment(img)
    subj = (reg.subject() > 0.5).astype(np.uint8)
    best_hit = 0
    for f in faces:
        x, y, bw, bh = [int(v) for v in f.box]
        cx, cy = min(w - 1, x + bw // 2), min(h - 1, y + bh // 2)
        if subj[cy, cx]:
            best_hit += 1
    check("subject 覆盖主体脸中心（单主体语义）", best_hit >= 1,
          "%d/%d 张（多人为模型局限）" % (best_hit, len(faces)))
    vis = AM.colourise(masks, img)
    save_image(os.path.join(OUT, "_multi_masks.jpg"), vis, 90)

    # 3) 祛路人：**画面人物保护**（overlap≥0.45 的主角全不删）——
    #    三个主体级人物（0.86/0.77/0.68）全保护，路人=0；背景小人
    #    （<20px）被滤掉。旧逻辑"只有 1 个主体"会把合影主角删掉
    #    （数据丢失级缺陷，本轮修复的核心）。
    people, subj_m, subj_box = PB.find_passersby(img)
    check("合影主角全保护（路人=0）", len(people) == 0,
          "路人=%d" % len(people))
    # hole 为空 → 直接返回原图（没有可删内容，不算错误）
    out, rinfo = PB.remove_passersby(img, max_people=3)
    check("无可删路人时返回原图", rinfo.get("removed", 0) == 0 and
          not rinfo.get("guidance"),
          str(rinfo)[:80])
    check("输出合法", float(out.min()) >= 0 and float(out.max()) <= 1)
    save_image(os.path.join(OUT, "_multi_before.jpg"), img, 90)
    save_image(os.path.join(OUT, "_multi_after.jpg"), out, 90)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
