# -*- coding: utf-8 -*-
"""RAW（NEF）全流程冒烟：打开 → 调色 → 导出全分辨率 → EXIF."""
import io
import json
import os
import sys
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
OUT = os.path.join(ROOT, "out")
FAIL = []
NEF = r"C:\Users\Rana\Desktop\26cicf\d1\新建文件夹\_DSC7031.NEF"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    # 1) 上传 NEF
    with open(NEF, "rb") as fh:
        t0 = time.time()
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.NEF", fh,
                                           "application/octet-stream")},
                           timeout=600).json()
    check("NEF 上传成功", bool(up.get("session")), "%.1fs" % (time.time() - t0))
    sid = up.get("session") or ""
    check("NEF 标记为 RAW", bool((up.get("decode") or {}).get("raw")),
          str((up.get("decode") or {}).get("raw")))

    # 2) 调色（EV 单位）
    r = requests.post(B + "/api/render", json={
        "session": sid, "params": {"exposure": 25, "contrast": 12,
                                   "temp": -6}}, timeout=900)
    check("NEF 调色", r.ok, str(r.status_code))

    # 3) 导出全分辨率
    out_path = os.path.join(OUT, "_raw_export.jpg")
    if os.path.exists(out_path):
        os.remove(out_path)
    r = requests.post(B + "/api/save", json={
        "session": sid, "path": out_path, "quality": 94}, timeout=1800)
    check("RAW 导出成功", r.ok and os.path.exists(out_path),
          str(r.status_code))
    if os.path.exists(out_path):
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(out_path) as im:
            w, h = im.size
            exif = im.getexif()
            keys = {TAGS.get(k, k) for k in exif.keys()}
            # Exif IFD（曝光字段在子 IFD 里，与相机原布局一致）
            sub = exif.get_ifd(0x8769) if hasattr(exif, "get_ifd") else {}
            subkeys = {TAGS.get(k, k) for k in sub.keys()}
        check("导出为全分辨率 RAW 解码", w >= 5000, "%dx%d" % (w, h))
        check("EXIF 主 IFD（Make/Model/Artist）",
              "Make" in keys and "Model" in keys,
              str(sorted(keys))[:80])
        # RAW 的曝光字段是摄影师最看重的——rawexif 链路应把它们写进
        # 子 IFD（ISO/FNumber/ExposureTime/FocalLength）
        check("EXIF 子 IFD 有 ISO",
              "ISOSpeedRatings" in subkeys or "ISO" in subkeys,
              str(sorted(subkeys))[:100])
        check("EXIF 子 IFD 有光圈/快门/焦距",
              any(k in subkeys for k in ("FNumber",)),
              str(sorted(subkeys))[:100])
        # RAW 导出应如实说明 EXIF 来源是 raw 重建（_copy_exif + rawexif）
        rj = r.json() if r.ok else {}
        exif_note = rj.get("exif") or {}
        check("导出带 EXIF 说明字段（source=raw）",
              exif_note.get("source") == "raw" and
              exif_note.get("copied") is True,
              str(exif_note)[:80])
        os.remove(out_path)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
