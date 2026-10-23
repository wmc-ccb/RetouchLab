# -*- coding: utf-8 -*-
"""用 5 点作锚，自动把 106 点归到各面部区域。

看图确认这 106 点不是 insightface 的顺序轮廓，而是**按区域聚集的稠密点云**：
0-9 下颌、10-19 左轮廓、20-29 右轮廓、40-49 右眼、60-69 嘴唇、90-99 左眼、
100-105 右眉……组边界不规则，不能靠固定索引表，但可以靠**空间位置**归属。

做法：YuNet 的 5 点给出左右眼中心、鼻尖、两嘴角，这些是确定的。然后用
距离 + 相对位置把每个 106 点分到最近的语义区。这样不管模型内部怎么编号，
结果都按真实几何归类。

输出一张映射表，供 portrait.py 使用。
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.imgutil import load_image, fit_long_edge
from engine.face import FaceDetector
from engine import landmarks as LMK

SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
OUT = os.path.join(ROOT, "out")


def classify(pts, lm5):
    """把每个点分到区域。返回 dict: region -> [索引].

    用**椭圆归一化距离**而不是欧氏距离：眼睛是横长的，欧氏距离会让纵向
    的点（眉毛、面颊）也被算成"眼周"，实测那样 eye_l 会吸走 36 个点、眉毛
    变成 0 个。每个区域给它自己的横纵尺度，再比较归一化后的值。

    顺序也重要：从最具体（眼睛、嘴唇）到最笼统（轮廓），一旦归属就跳过，
    否则轮廓点会被宽区域截走。
    """
    er, el, nose = lm5[0], lm5[1], lm5[2]
    mr, ml = lm5[3], lm5[4]
    iod = float(np.linalg.norm(el - er))
    mid_eye = (er + el) / 2.0
    mouth_c = (mr + ml) / 2.0

    ang = float(np.arctan2(*(el - er)[::-1]))
    down = np.array([-np.sin(ang), np.cos(ang)], np.float32)
    right = np.array([np.cos(ang), np.sin(ang)], np.float32)

    def ell(p, centre, rx, ry):
        d = p - centre
        u = float(np.dot(d, right)) / (rx * iod)
        v = float(np.dot(d, down)) / (ry * iod)
        return u * u + v * v

    regions = {k: [] for k in ("eye_r", "eye_l", "brow_r", "brow_l",
                               "lips", "nose", "contour_r", "contour_l",
                               "chin")}

    # 眉毛锚点：眼中心沿法线向上 lift。
    # 实测右眼(396,161) 的眉点在 (411,148)~(418,143)，即向上 13~18px，
    # 眼距 56.5 -> 0.23~0.32 眼距。所以取 0.30，而不是初版的 0.44。
    lift = 0.30 * iod

    for i, p in enumerate(pts):
        # 1) 眼睛：横 0.45、纵 0.20 眼距的扁椭圆。
        #    实测眼周点集中在 d=0.09~0.18，0.20 的纵半径刚好不含眉毛。
        if ell(p, er, 0.45, 0.20) < 1.0:
            regions["eye_r"].append(i); continue
        if ell(p, el, 0.45, 0.20) < 1.0:
            regions["eye_l"].append(i); continue
        # 2) 眉毛：更高、更扁的椭圆
        if ell(p, er + down * (-lift), 0.50, 0.14) < 1.0:
            regions["brow_r"].append(i); continue
        if ell(p, el + down * (-lift), 0.50, 0.14) < 1.0:
            regions["brow_l"].append(i); continue
        # 3) 嘴唇：横 0.78、纵 0.28 眼距
        if ell(p, mouth_c, 0.78, 0.28) < 1.0:
            regions["lips"].append(i); continue
        # 4) 鼻子：竖长椭圆，横向窄
        if ell(p, nose, 0.30, 0.34) < 1.0:
            regions["nose"].append(i); continue
        # 5) 其余：按左右与上下分轮廓/下巴
        s = float(np.dot(p - mid_eye, right)) / iod
        t = float(np.dot(p - mouth_c, down)) / iod
        if t > 0.42:
            regions["chin"].append(i)
        elif s < 0:
            regions["contour_r"].append(i)
        else:
            regions["contour_l"].append(i)

    return regions


def main():
    img = fit_long_edge(load_image(SRC), 900)
    f = FaceDetector(dense=False).detect(img)[0]
    x, y, bw, bh = [float(v) for v in f.box]
    lm5 = np.asarray(f.landmarks, np.float32)
    pts = LMK.Landmark106().predict(
        img, np.asarray([x, y, x + bw, y + bh], np.float32)).points

    reg = classify(pts, lm5)
    print("区域归属（共 %d 点）:" % len(pts))
    total = 0
    for k, v in reg.items():
        total += len(v)
        print("  %-11s %2d 点  %s" % (k, len(v), v))
    print("  合计 %d（应等于 %d）" % (total, len(pts)))

    # 交叉检验：各区域中心是否落在合理位置
    er, el, nose = lm5[0], lm5[1], lm5[2]
    iod = float(np.linalg.norm(el - er))
    mr, ml = lm5[3], lm5[4]
    mouth_c = (mr + ml) / 2.0
    print("\n区域中心与锚点距离（眼距为单位）:")
    checks = {"eye_r": er, "eye_l": el, "nose": nose, "lips": mouth_c}
    for k, anchor in checks.items():
        if not reg[k]:
            print("  %-6s 空!" % k)
            continue
        c = pts[reg[k]].mean(axis=0)
        print("  %-6s 中心距锚点 %.2f 眼距" % (k, np.linalg.norm(c - anchor) / iod))

    # 保存映射供引擎读取
    path = os.path.join(ROOT, "engine", "lm106_regions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({k: v for k, v in reg.items()}, fh, ensure_ascii=False,
                  indent=1)
    print("\n映射已保存 -> engine/lm106_regions.json")


if __name__ == "__main__":
    main()
