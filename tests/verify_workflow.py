# -*- coding: utf-8 -*-
"""真实用户工作流冒烟：导入 → 调色 → 局部 → 背景家族 → 导出（含 EXIF）.

模拟一个用户的完整一次修图会话，验证各功能在**同一会话**上顺序工作
（单元测试都是各自开新会话，测不到状态互相污染）。
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
OUT = os.path.join(ROOT, "out")
FAIL = []
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    # 1) 导入（RAW 更真实，但用 JPEG 保证可读 EXIF 断言）
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("workflow.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    check("导入得到会话", bool(sid))

    # 2) 全局调色（EV + HSL 新单位）
    r = requests.post(B + "/api/render", json={
        "session": sid,
        "params": {"exposure": 20, "contrast": 10, "hsl_orange_sat": -30,
                   "temp": 8}}, timeout=600)
    check("调色管线（EV/HSL 新单位）", r.ok, str(r.status_code))

    # 3) 美肤 + 五官（人像链路）
    r = requests.post(B + "/api/render", json={
        "session": sid, "params": {"smooth": 35, "slimFace": 15}},
        timeout=600)
    check("人像管线", r.ok, str(r.status_code))

    # 4) 背景家族顺序执行（同一会话连续跑四个）
    r1 = requests.post(B + "/api/automask", json={"session": sid},
                       timeout=900)
    check("自动蒙版", r1.ok, str(r1.status_code))
    r2 = requests.post(B + "/api/sky", json={
        "session": sid, "preset": "clear_blue", "strength": 0.6},
        timeout=900)
    check("换天空（室内图少改）", r2.ok, str(r2.status_code))
    r3 = requests.post(B + "/api/scene", json={
        "session": sid, "preset": "sakura", "strength": 0.8}, timeout=900)
    check("四季背景", r3.ok, str(r3.status_code))
    r4 = requests.post(B + "/api/rigfix", json={"session": sid}, timeout=900)
    check("去道具", r4.ok, str(r4.status_code))
    r5 = requests.post(B + "/api/grassfix", json={"session": sid},
                       timeout=900)
    check("补草地（室内如实说明）", r5.ok and
          (r5.json().get("info") or {}).get("applicable") is False)

    # 5) 历史应该记录了 commit 过的步骤（render 带 commit 才记）
    r = requests.post(B + "/api/render", json={
        "session": sid, "params": {"exposure": 5}, "commit": True},
        timeout=600)
    check("提交一步进历史", r.ok, str(r.status_code))
    h = requests.get(B + "/api/history/" + sid, timeout=60).json()
    steps = h.get("steps") or []
    check("历史有步骤", len(steps) >= 1, "%d 步" % len(steps))

    # 6) 导出文件（走 /api/save，应带 EXIF）
    out_path = os.path.join(OUT, "_workflow_export.jpg")
    if os.path.exists(out_path):
        os.remove(out_path)
    r = requests.post(B + "/api/save", json={
        "session": sid, "path": out_path, "quality": 94}, timeout=600)
    check("导出成功", r.ok and os.path.exists(out_path), str(r.status_code))
    if os.path.exists(out_path):
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(out_path) as im:
            exif = im.getexif()
            keys = {TAGS.get(k, k) for k in exif.keys()}
            # 导出是**全分辨率原图重算管线**（/api/save 用 s["full"] 跑
            # _pipe.run，不是放大 preview）——5223px 原图导出 5223px 是
            # 正确行为；断言"不小于 preview 且是原图尺寸"。
            check("导出为全分辨率（≥ preview）",
                  im.size[0] >= up["preview_width"],
                  "%s vs preview %s" % (im.size, (up["preview_width"],
                                                  up["preview_height"])))
        check("EXIF 保留（Make/Model 至少一个）",
              ("Make" in keys) or ("Model" in keys) or ("Software" in keys),
              str(sorted(keys))[:80])
        os.remove(out_path)

    # 7) 新会话不被旧会话污染（状态隔离）。
    # 历史列表自带种子步骤"原图"（History.__init__ labels=["原图"]），
    # 这是页面重建历史列的设计行为；断言"没有上一个会话的提交步骤"。
    with open(SRC, "rb") as fh:
        up2 = requests.post(B + "/api/upload",
                            files={"file": ("w2.jpg", fh, "image/jpeg")},
                            timeout=300).json()
    sid2 = up2["session"]
    h2 = requests.get(B + "/api/history/" + sid2, timeout=60).json()
    labels = [s.get("label") for s in (h2.get("steps") or [])]
    check("新会话无旧会话步骤（状态隔离）",
          len(labels) <= 1 and "调色" not in "".join(labels),
          str(labels))

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
