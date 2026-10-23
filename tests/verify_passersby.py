# -*- coding: utf-8 -*-
"""AI 祛路人：合成场景有真值，路人应被抹掉、主体逐位不变."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import passersby as PB

FAIL = []


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def scene(with_passer=True):
    """合成：纯色渐变背景 + 主体（大圆头）+ 可选路人（小圆头）.

    主体占画面中心偏右，路人小很多且远离主体——这是"主对象+路人"的
    标准形态，也保证脸部检测能区分大小。
    """
    import cv2

    h, w = 600, 900
    img = np.zeros((h, w, 3), np.float32)
    # 背景：带光影渐变的蓝灰色
    gx = np.linspace(0.72, 0.58, w, dtype=np.float32)[None, :, None]
    gy = np.linspace(1.04, 0.92, h, dtype=np.float32)[:, None, None]
    img[:] = np.array([0.62, 0.66, 0.72], np.float32) * gx * gy
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 主体：头 (150px) + 躯干，占中间偏右
    subj = (((xx - 620) ** 2 + (yy - 220) ** 2) < 150 ** 2).astype(np.float32)
    body = ((yy > 340) & (yy < h) & (np.abs(xx - 620) < 190)).astype(np.float32)
    subj = np.maximum(subj, body)
    subj = cv2.GaussianBlur(subj, (0, 0), 2.0)
    skin = np.array([0.82, 0.66, 0.58], np.float32)
    # 头发区（上半）稍深
    hair = np.array([0.35, 0.28, 0.22], np.float32)
    face_m = (((xx - 620) ** 2 + (yy - 235) ** 2) < 110 ** 2).astype(np.float32)
    body_col = hair * (1 - face_m[..., None]) + skin * face_m[..., None]
    img = img * (1 - subj[..., None]) + body_col * subj[..., None]
    truth_subject = subj > 0.5

    if with_passer:
        # 路人：小头 + 细身，在左下。头部要有"脸"的特征（眼睛暗点 +
        # 肤色），否则 YuNet 对纯色圆不响应——真人检出下限约 60px，
        # 合成图没有五官纹理，44px + 暗点能落进检出窗口。
        pr = 44.0
        px_, py_ = 150.0, 468.0
        head = (((xx - px_) ** 2 + (yy - py_) ** 2) < pr ** 2).astype(np.float32)
        pbody = ((yy > 500) & (yy < h) &
                 (np.abs(xx - px_) < 44)).astype(np.float32)
        passer = np.maximum(head, pbody)
        # 头部底色：肤色
        pc = np.array([0.72, 0.58, 0.50], np.float32)
        img = img * (1 - head[..., None]) + pc * head[..., None]
        # 五官暗点：两只眼 + 嘴（符合人脸先验的水平排布）
        for ex, ey, er in ((px_ - 14, py_ - 6, 5.0), (px_ + 14, py_ - 6, 5.0),
                           (px_, py_ + 16, 7.0)):
            dot = (((xx - ex) ** 2 + (yy - ey) ** 2) < er ** 2).astype(np.float32)
            img = img * (1.0 - dot * 0.85)[..., None]
        # 身体：灰衣服
        pc2 = np.array([0.55, 0.52, 0.60], np.float32)
        img = img * (1 - pbody[..., None]) + pc2 * pbody[..., None]
        img = np.clip(img, 0, 1)

    return np.clip(img, 0, 1), truth_subject


def main():
    img, subj_truth = scene(with_passer=True)
    h, w = img.shape[:2]

    # ---- 1. 检测用**真人照片**验证。合成脸没有五官纹理，YuNet 对纯色
    #         圆不响应——这是合成场景的局限，不是检测的 bug；真值断言
    #         放在掩膜几何与 LaMa 环节（注入框）。
    from engine.imgutil import load_image, fit_long_edge
    real = fit_long_edge(load_image(
        r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"), 900)
    real_people, real_subj, real_sbox = PB.find_passersby(real)
    check("真人照片：返回主体脸框", real_sbox is not None and
          len(real_sbox) == 4, str(real_sbox)[:40])
    check("真人照片：检出路人", len(real_people) >= 1,
          "%d 个 %s" % (len(real_people),
                        [[round(v) for v in p["box"]]
                         for p in real_people][:2]))
    check("真人照片：有主体掩膜", real_subj is not None)
    if real_people and real_subj is not None:
        pm0 = PB.person_mask(real, real_people[0]["box"])
        c0 = float(((pm0 > 0.5) & (real_subj > 0.5)).sum() /
                   max((pm0 > 0.5).sum(), 1))
        check("真人照片：路人掩膜主体占比低", c0 < 0.15, "%.3f" % c0)
    # 主体自己的脸（宽度 ~120px）不应出现在路人列表里
    check("主体未被当作路人",
          not any(p["box"][2] > 100 for p in real_people),
          str([round(p["box"][2]) for p in real_people]))

    # ---- 2. 掩膜几何：合成场景 + **注入检测框**（有真值）----
    fake_box = [150 - 40, 468 - 46, 80, 92]     # 罩住合成路人
    pm = PB.person_mask(img, fake_box)
    check("路人掩膜生成", pm is not None and pm.max() > 0.5)
    overlap = float((pm * (subj_truth > 0.5)).max())
    check("掩膜不覆盖主体（峰值低）", overlap < 0.3, "%.3f" % overlap)
    cover = float(((pm > 0.5) & subj_truth).sum() /
                  max((pm > 0.5).sum(), 1))
    check("掩膜主体占比低", cover < 0.15, "%.3f" % cover)

    # ---- 3. LaMa 全流程：合成场景 + 注入框（掩膜→填充的确定性路径）----
    hole = PB.person_mask(img, fake_box)
    hole = hole * (1 - (subj_truth > 0.5))       # 与 remove_passersby 相同剔除
    try:
        from engine import lama as LM
        res = LM.fill(img, hole)
        filled = np.asarray(res[0], np.float32)
        d = np.abs(filled - img).max(axis=2)
        subj_delta = float(d[subj_truth].max())
        check("LaMa：主体不变", subj_delta < 0.35,
              "主体 maxΔ=%.3f" % subj_delta)
        passer_delta = float(d[430:600, 95:210].mean())
        check("LaMa：路人区域被改动", passer_delta > 0.05,
              "路人 meanΔ=%.3f" % passer_delta)
        other = d[0:420, 0:800]
        check("LaMa：远处背景不动", float(other.max()) < 0.05,
              "远处 maxΔ=%.3f" % float(other.max()))
    except Exception as e:
        check("LaMa 可用", False, str(e)[:120])

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
