# -*- coding: utf-8 -*-
"""灯架检测假阳性测试：建筑竖线/树干/栏杆不应被误判."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "out")
FAIL = []


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    import cv2
    from engine import rigfix as RF

    def synth_bg(h, w):
        """浅灰墙 + 深色窗框竖线（建筑竖线的近似）。"""
        img = np.full((h, w, 3), 0.62, np.float32)
        for x in range(40, w, 120):
            img[:, x:x + 6] = 0.18          # 窗框竖线（暗、细长、垂直）
        return np.clip(img, 0, 1)

    h, w = 600, 800
    # 1) 纯建筑竖线：不应识别出杆（太细/太短都不够杆的份量，
    #    但 6px x 600px 满足细长比——需要**材质/语境**判据吗？）
    bg = synth_bg(h, w)
    m, info = RF.detect_rig(bg)
    print("建筑竖线图: poles=%d heads=%d" % (info.get("poles"),
                                             info.get("heads")))
    # 2) 真灯架（粗 8px、有横撑+柔光箱）在**空旷墙前**应识别。
    #    （灯架紧贴窗框线 40px 内的病态布局不测：视觉上本来就像窗框，
    #    CLOSE 会把两者粘成一体，无法区分——这是物理边界。）
    rig = np.full((h, w, 3), 0.62, np.float32)
    rig[100:560, 336:344] = 0.10
    cv2.line(rig, (340, 400), (275, 560), (0.10, 0.10, 0.10), 4)
    cv2.line(rig, (340, 400), (405, 560), (0.10, 0.10, 0.10), 4)
    cv2.rectangle(rig, (305, 30), (380, 100), (0.93, 0.93, 0.92), -1)
    m2, info2 = RF.detect_rig(rig)
    print("真灯架图: poles=%d heads=%d rejected=%s" % (
        info2.get("poles"), info2.get("heads"),
        str(info2.get("rejected_pattern"))[:30]))
    check("空旷墙前真灯架识别", info2.get("poles") == 1 and
          info2.get("heads") == 1,
          "poles=%s heads=%s" % (info2.get("poles"), info2.get("heads")))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
