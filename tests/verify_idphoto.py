# -*- coding: utf-8 -*-
"""证件照：规格尺寸 / 头部占比 / 底色纯净 / 无脸回退."""
import base64
import io
import os
import sys

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
OUT = os.path.join(ROOT, "out")
FAIL = []
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"

SPECS = {"one_inch": (295, 413), "two_inch": (413, 579),
         "small_two": (413, 531), "visa": (350, 450),
         "passport": (390, 567)}


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]

    info = requests.get(B + "/api/idphoto/info", timeout=60).json()
    check("规格表 5 种", len(info.get("specs", {})) == 5,
          str(list(info.get("specs", {}).keys())))

    # 每种规格尺寸必须精确
    for spec, (w, h) in SPECS.items():
        r = requests.post(B + "/api/idphoto", json={
            "session": sid, "spec": spec, "colour": [255, 255, 255]},
            timeout=600)
        if not r.ok:
            check("%s 生成" % spec, False, "%d %s" % (r.status_code,
                                                      r.text[:80]))
            continue
        j = r.json()
        check("%s 尺寸 %dx%d" % (spec, w, h),
              j.get("width") == w and j.get("height") == h,
              "%sx%s" % (j.get("width"), j.get("height")))
        if spec == "two_inch":
            im = Image.open(io.BytesIO(base64.b64decode(j["image"])))
            im.save(os.path.join(OUT, "idphoto_two.jpg"), quality=94)
            arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
            # 四角应为白（底色纯净）——留 6% 边避开头发
            hh, ww = arr.shape[:2]
            m = int(min(hh, ww) * 0.06)
            corners = np.concatenate([
                arr[:m, :m].reshape(-1, 3), arr[:m, -m:].reshape(-1, 3)])
            dev = float(np.abs(corners - 1.0).max())
            check("两寸四角接近纯白", dev < 0.12, "最大偏差 %.3f" % dev)
            # 头部占比：人脸高度 / 画面高度 应在 0.45~0.75
            from engine.face import FaceDetector
            from engine.imgutil import load_image
            f_img = np.asarray(im.convert("RGB"), np.float32) / 255.0
            faces = FaceDetector(dense=False).detect(f_img, max_faces=3)
            check("两寸能检出人脸", len(faces) >= 1, "%d 张" % len(faces))
            if faces:
                fh = max(f.box[3] for f in faces)
                ratio = fh / arr.shape[0]
                # 断言的是"YuNet 脸框"占比，不是规范里的"头部（含发）"占比。
                # 脸框只覆盖眉骨到下唇，约头高的 0.58；规范头部占比 0.60~
                # 0.70 对应脸框占比 0.35~0.41。早先按 0.45~0.75 断言，把
                # 正确的构图判成失败。
                check("脸框占比在规范换算区间 0.33~0.45",
                      0.33 <= ratio <= 0.45, "%.2f" % ratio)

    # 蓝底：四角应接近指定蓝
    r = requests.post(B + "/api/idphoto", json={
        "session": sid, "spec": "one_inch",
        "colour": [28, 90, 184]}, timeout=600)
    if r.ok:
        j = r.json()
        im = Image.open(io.BytesIO(base64.b64decode(j["image"])))
        arr = np.asarray(im.convert("RGB"), np.float32) / 255.0
        hh, ww = arr.shape[:2]
        m = int(min(hh, ww) * 0.06)
        corners = np.concatenate([
            arr[:m, :m].reshape(-1, 3), arr[:m, -m:].reshape(-1, 3)])
        target = np.array([28, 90, 184], np.float32) / 255.0
        dev = float(np.abs(corners - target).max())
        check("一寸蓝底四角接近目标色", dev < 0.15, "最大偏差 %.3f" % dev)
        im.save(os.path.join(OUT, "idphoto_blue.jpg"), quality=94)

    # 无脸图：回退中心裁切，不崩
    blank = Image.new("RGB", (800, 600), (120, 140, 160))
    buf = io.BytesIO()
    blank.save(buf, "JPEG")
    up2 = requests.post(B + "/api/upload",
                        files={"file": ("b.jpg", buf.getvalue(),
                                        "image/jpeg")}, timeout=300).json()
    r = requests.post(B + "/api/idphoto", json={
        "session": up2["session"], "spec": "two_inch"}, timeout=600)
    check("无脸图不崩且回退", r.ok and
          (r.json().get("info") or {}).get("fallback_center") is True,
          "%d %s" % (r.status_code,
                     str((r.json().get("info") or {}).get("fallback_center"))
                     if r.ok else r.text[:80]))

    # 未知规格 -> 400
    r = requests.post(B + "/api/idphoto", json={
        "session": sid, "spec": "nope"}, timeout=60)
    check("未知规格 400", r.status_code == 400, str(r.status_code))

    # 导出到文件
    out_path = os.path.join(OUT, "_idphoto_export.jpg")
    if os.path.exists(out_path):
        os.remove(out_path)
    r = requests.post(B + "/api/idphoto", json={
        "session": sid, "spec": "two_inch",
        "out_path": out_path}, timeout=600)
    ok = r.ok and os.path.exists(out_path)
    check("导出到文件", ok, "%s" % (r.json().get("path") if r.ok else ""))
    if ok:
        with Image.open(out_path) as im:
            size = im.size
        check("导出尺寸正确", size == (413, 579), str(size))
        try:
            os.remove(out_path)
        except PermissionError:
            pass   # Windows 上 PIL 句柄可能延迟释放，不影响断言

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
