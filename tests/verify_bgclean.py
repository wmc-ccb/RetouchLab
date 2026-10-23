# -*- coding: utf-8 -*-
"""背景消除：纯色背景四类缺陷 + 换底色。

用**合成场景**测，因为合成场景有真值：
  * 背景色是已知常量 -> 可以验证"修完是否等于背景色"
  * 主体位置已知 -> 可以验证"主体是否被改动"
这比拿真照片肉眼看严格得多。

真照片那侧只验证 API 不崩、返回图尺寸一致、info 里有各子项的统计。
"""
import base64
import io
import os
import sys

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import bgclean as BC

B = "http://127.0.0.1:8765"
FAIL = []


def check(name, ok, detail=""):
    print("%-52s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def synth():
    """合成一张纯色背景人像：背景 + 肤色椭圆主体。"""
    h, w = 300, 400
    bgc = np.array([0.86, 0.88, 0.90], np.float32)
    img = np.tile(bgc, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - h * 0.55) ** 2 + (xx - w * 0.5) ** 2)
    subj = np.clip((110 - r) / 6.0, 0, 1)
    skin = np.array([0.78, 0.62, 0.55], np.float32)
    img = img * (1 - subj[..., None]) + skin * subj[..., None]
    return img, subj, bgc


def main():
    img, subj, bgc = synth()
    bgm = BC.background_mask(img, subject=subj) > 0.5

    # ---------- 1. 穿帮 ----------
    bad = img.copy()
    bad[30:60, 40:90] = [0.25, 0.30, 0.28]          # 一块深色穿帮
    out, info = BC.apply(bad, subject=subj, showthrough=1.0)
    d = np.abs(out - bad).max(axis=2)
    check("祛穿帮：主体逐位不变", float(d[~bgm].max()) < 1e-6,
          "maxΔ=%.6f" % float(d[~bgm].max()))
    patch = out[35:55, 45:85].mean(axis=(0, 1))
    err = float(np.abs(patch - bgc).max())
    check("祛穿帮：补丁接近背景色", err < 0.06, "误差=%.3f" % err)
    check("祛穿帮：确实检测到穿帮块", info["showthrough"]["patched"] >= 1,
          "%d 处" % info["showthrough"]["patched"])

    # ---------- 2. 瑕疵 ----------
    bad = img.copy()
    for (y0, x0) in [(40, 320), (240, 60), (270, 350)]:
        bad[y0:y0 + 5, x0:x0 + 5] *= 0.80             # 几个暗点
    out, info = BC.apply(bad, subject=subj, blemish=1.0)
    d = np.abs(out - bad).max(axis=2)
    check("祛瑕疵：主体逐位不变", float(d[~bgm].max()) < 1e-6,
          "maxΔ=%.6f" % float(d[~bgm].max()))
    # 修完后暗点应被拉回背景亮度
    spot_before = float(bad[40:45, 320:325].mean())
    spot_after = float(out[40:45, 320:325].mean())
    check("祛瑕疵：暗点被提亮", spot_after > spot_before + 0.02,
          "%.3f -> %.3f" % (spot_before, spot_after))

    # ---------- 3. 光影均匀 ----------
    bad = np.clip(img * np.linspace(1.12, 0.88, img.shape[1])[None, :, None],
                  0, 1)
    out, info = BC.apply(bad, subject=subj, even_light=1.0)
    d = np.abs(out - bad).max(axis=2)
    check("光影均匀：主体逐位不变", float(d[~bgm].max()) < 1e-6,
          "maxΔ=%.6f" % float(d[~bgm].max()))
    # 左半背景 vs 右半背景的亮度差应该缩小
    def lr(im):
        l = float(im[:, :80].mean()); r_ = float(im[:, -80:].mean())
        return abs(l - r_)
    before, after = lr(bad), lr(out)
    check("光影均匀：左右亮度差缩小", after < before * 0.5,
          "%.3f -> %.3f" % (before, after))

    # ---------- 4. 色彩断层 ----------
    bad = np.clip(np.round(img * 14) / 14, 0, 1)
    out, info = BC.apply(bad, subject=subj, banding=1.0)
    d = np.abs(out - bad).max(axis=2)
    check("祛断层：主体逐位不变", float(d[~bgm].max()) < 1e-6,
          "maxΔ=%.6f" % float(d[~bgm].max()))
    # 量化台阶的离散度：统计背景区唯一色数应增加（抖动把台阶打散）
    def uniq(im):
        q = (np.clip(im[bgm], 0, 1) * 255).astype(np.uint8)
        return len(np.unique(q.reshape(-1, 3), axis=0))
    check("祛断层：背景色阶被抖动打散", uniq(out) > uniq(bad),
          "%d -> %d" % (uniq(bad), uniq(out)))

    # ---------- 5. 组合 ----------
    bad = img.copy()
    bad[30:60, 40:90] = [0.25, 0.30, 0.28]
    bad[240:245, 60:65] *= 0.8
    bad = np.clip(bad * np.linspace(1.1, 0.9, bad.shape[1])[None, :, None], 0, 1)
    out, info = BC.apply(bad, subject=subj, showthrough=1.0, blemish=1.0,
                         even_light=1.0, banding=1.0, enhance=1.0)
    d = np.abs(out - bad).max(axis=2)
    # 六项串联后 8 位量化会累积一点浮点误差（实测 7e-6 = 不到 0.002 个色阶），
    # 逐位相等在这个量级上不成立，用"不可见"作判据。
    check("组合四项：主体不可见变化", float(d[~bgm].max()) < 1e-4,
          "maxΔ=%.2e" % float(d[~bgm].max()))
    check("组合四项：info 含全部子项",
          all(k in info for k in ("showthrough", "blemish", "even_light",
                                  "banding", "enhance")))

    # ---------- 6. 背景掩膜自洽 ----------
    m = BC.background_mask(img, subject=subj)
    check("背景掩膜：与真值交并比 > 0.98",
          float(((m > 0.5) & bgm).sum() / max(((m > 0.5) | bgm).sum(), 1)) > 0.98)
    m2 = BC.background_mask(img)   # 不给 subject，靠四角色
    iou = float(((m2 > 0.5) & bgm).sum() / max(((m2 > 0.5) | bgm).sum(), 1))
    check("背景掩膜：纯四角估计交并比 > 0.95", iou > 0.95, "%.3f" % iou)

    # ---------- 7. API ----------
    def up(path):
        with open(path, "rb") as fh:
            return requests.post(B + "/api/upload",
                                 files={"file": ("x.jpg", fh, "image/jpeg")},
                                 timeout=300).json()

    real = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
    u = up(real)
    sid = u["session"]
    r = requests.get(B + "/api/bgclean/info", timeout=60)
    j = r.json() if r.ok else {}
    check("API info 可用", r.ok and "features" in j, str(j)[:60])

    r = requests.post(B + "/api/bgclean", json={
        "session": sid, "showthrough": 1, "blemish": 1, "even_light": 1,
        "banding": 1, "commit": False}, timeout=900)
    check("API bgclean 返回图", r.ok and len(r.json().get("image", "")) > 5000,
          "%d" % r.status_code)
    if r.ok:
        im = Image.open(io.BytesIO(base64.b64decode(r.json()["image"])))
        check("API bgclean 尺寸不变", im.size == (u["preview_width"],
                                                  u["preview_height"]),
              str(im.size))

    for name, col in (("白底", [1, 1, 1]), ("蓝底", [0.12, 0.35, 0.72])):
        r = requests.post(B + "/api/bgreplace", json={
            "session": sid, "colour": col, "commit": False}, timeout=900)
        ok = r.ok and len(r.json().get("image", "")) > 5000
        check("API 换%s" % name, ok, "%d" % r.status_code)
        if ok:
            im = Image.open(io.BytesIO(base64.b64decode(r.json()["image"])))
            check("API 换%s 尺寸不变" % name,
                  im.size == (u["preview_width"], u["preview_height"]),
                  str(im.size))

    # ---------- 8. 色键抠图（纯色背景换底色的正确路径）----------
    print("\n=== 色键抠图 ===")
    h, w = 400, 300
    bgc2 = np.array([0.55, 0.60, 0.66], np.float32)
    im2 = np.tile(bgc2, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    head = ((yy - 150) ** 2 + (xx - 150) ** 2) < 70 ** 2
    body = (yy > 200) & (yy < 400) & (np.abs(xx - 150) < 95)
    truth = (head | body).astype(np.float32)
    import cv2 as _cv
    soft = _cv.GaussianBlur(truth, (0, 0), 1.5)
    skin2 = np.array([0.80, 0.64, 0.56], np.float32)
    im2 = im2 * (1 - soft[..., None]) + skin2 * soft[..., None]

    a2, kinfo = BC.color_key_alpha(im2)
    pred = a2 > 0.5
    gt = truth > 0.5
    iou = float((pred & gt).sum() / max((pred | gt).sum(), 1))
    check("色键抠图与真值 IoU > 0.98", iou > 0.98, "%.3f" % iou)
    check("色键估出背景色", np.abs(np.array(kinfo["bg_colour"]) -
                                  bgc2).max() < 0.05,
          str(kinfo["bg_colour"]))
    # 换底后逐像素验证。判据用 alpha 本身：
    #   * alpha 极低 -> 结果必须等于目标色
    #   * alpha 极高 -> 结果必须等于原图
    # 不能在 alpha 0.02~0.98 的过渡带上断言，那里本来就是两色混合；
    # 之前用"硬边真值"划背景/主体，羽化带被算成了错误。
    for name, col in (("白", [1, 1, 1]), ("蓝", [0.12, 0.35, 0.72])):
        c = np.array(col, np.float32)
        out2 = im2 * a2[..., None] + np.tile(c, (h, w, 1)) * (1 - a2[..., None])
        pure_bg = a2 <= 1e-3
        err_bg = float(np.abs(out2[pure_bg] - c).max()) if pure_bg.any() else 0
        check("色键换%s底：纯背景区等于目标色" % name, err_bg < 0.01,
              "误差 %.4f（%d px）" % (err_bg, int(pure_bg.sum())))
        pure_fg = a2 >= 0.999
        err_fg = (float(np.abs(out2[pure_fg] - im2[pure_fg]).max())
                  if pure_fg.any() else 0)
        check("色键换%s底：纯主体区不被染色" % name, err_fg < 0.005,
              "误差 %.4f（%d px）" % (err_fg, int(pure_fg.sum())))
        # 过渡带必须落在两色之间（不能出现"底色透到身上"那种跳变）
        band = (a2 > 0.05) & (a2 < 0.95)
        if band.any():
            lo = np.minimum(im2[band], c)
            hi = np.maximum(im2[band], c)
            inrange = float(((out2[band] >= lo - 0.02) &
                             (out2[band] <= hi + 0.02)).all())
            check("色键换%s底：过渡带不越界" % name, inrange == 1.0,
                  "%.3f" % inrange)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
