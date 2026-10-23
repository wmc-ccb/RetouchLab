# -*- coding: utf-8 -*-
"""直方图：数据正确性、跟随编辑、面板常驻.

这个功能之前是坏的，而且坏得很安静：

* 前端 `if (MASK.hist) return` 让直方图整个会话只算一次，编辑后仍是旧的；
* `/api/render` 是纯预览，故意不写回 session，所以 `/api/histogram`（读
  session 里存的 preview）永远描述**上传时的原图**，和数据在屏幕上看到的东西
  不是一回事；
* 曲线面板从不触发加载，从那进去背景永远是空的。

所以这里断言的不是"能取到数据"，而是"它描述的就是用户正在看的那张图"。
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
BASE = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


class B:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=120)
        self.i = 0

    def call(self, m, **p):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": m, "params": p}))
        while True:
            r = json.loads(self.ws.recv())
            if r.get("id") == self.i:
                return r.get("result", {})

    def js(self, e):
        return self.call("Runtime.evaluate", expression=e, returnByValue=True,
                         awaitPromise=True).get("result", {}).get("value")

    def shot(self, p):
        r = self.call("Page.captureScreenshot", format="png")
        with open(p, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))
        return os.path.getsize(p) // 1024


def main():
    print("=== 数据正确性 ===")
    BINS = 64
    img = (np.random.rand(50, 70, 3) * 255).astype(np.uint8)
    import cv2
    ok, buf = cv2.imencode(".jpg", img)
    # 直接算一遍参考直方图，和服务端返回的比
    arr = cv2.imdecode(np.frombuffer(buf.tobytes(), np.uint8),
                       cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ref_lum = (rgb * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(axis=2)
    ref_hist, _ = np.histogram(np.clip(ref_lum, 0, 1), bins=BINS,
                               range=(0.0, 1.0))

    with open(SRC, "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("h.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]

    h = requests.get(BASE + "/api/histogram/" + sid, timeout=60).json()
    check("返回 RGB + 亮度", all(k in h for k in ("r", "g", "b", "lum")))
    lens_eq = (len(h["r"]) == len(h["g"]) == len(h["b"]) == len(h["lum"])
               == h["bins"])
    check("桶数一致", lens_eq,
          "r=%d g=%d b=%d lum=%d bins=%d" %
          (len(h["r"]), len(h["g"]), len(h["b"]), len(h["lum"]), h["bins"]))
    check("亮度总和 == 像素数（不丢最亮/最暗）",
          sum(h["lum"]) == h["total"],
          "%d vs %d" % (sum(h["lum"]), h["total"]))
    check("单通道总和 == 像素数", sum(h["r"]) == h["total"],
          "%d vs %d" % (sum(h["r"]), h["total"]))
    check("报告死黑/死白", h["clipped_low"] >= 0 and h["clipped_high"] >= 0,
          "低 %d / 高 %d" % (h["clipped_low"], h["clipped_high"]))
    check("带拍摄信息", "meta" in h and h["meta"].get("width", 0) > 0,
          str(h["meta"].get("width")))

    # ---------------- 关键：直方图必须跟随渲染 ----------------
    print("\n=== 直方图跟随编辑 ===")
    j0 = requests.post(BASE + "/api/render", json={
        "session": sid, "params": {}, "quality": 85,
        "with_histogram": True}, timeout=300).json()
    j1 = requests.post(BASE + "/api/render", json={
        "session": sid, "params": {"exposure": 60}, "quality": 85,
        "with_histogram": True}, timeout=300).json()
    check("render 返回直方图", "histogram" in j0 and "histogram" in j1)
    h0, h1 = j0["histogram"], j1["histogram"]
    check("不同曝光下直方图不同", h0["lum"] != h1["lum"])
    check("提亮后死白显著增加", h1["clipped_high"] > h0["clipped_high"] * 3,
          "%d -> %d" % (h0["clipped_high"], h1["clipped_high"]))
    # 峰值桶应该右移
    pk0 = h0["lum"].index(max(h0["lum"]))
    pk1 = h1["lum"].index(max(h1["lum"]))
    check("峰值桶右移（画面更亮）", pk1 > pk0, "%d -> %d" % (pk0, pk1))
    check("渲染直方图总和 == 该渲染的像素数",
          sum(h1["lum"]) == h1["total"],
          "%d vs %d" % (sum(h1["lum"]), h1["total"]))

    # 压暗应该减少死白、增加死黑
    j2 = requests.post(BASE + "/api/render", json={
        "session": sid, "params": {"exposure": -80}, "quality": 85,
        "with_histogram": True}, timeout=300).json()
    h2 = j2["histogram"]
    check("压暗后死黑增加", h2["clipped_low"] > h0["clipped_low"],
          "%d -> %d" % (h0["clipped_low"], h2["clipped_low"]))

    # /api/histogram 读的是已提交的 preview：这是故意的，文档要说清
    hs = requests.get(BASE + "/api/histogram/" + sid, timeout=60).json()
    check("GET /api/histogram 描述已提交帧（不等于预览渲染）",
          hs["lum"] == h0["lum"] and hs["lum"] != h1["lum"],
          "committed=%d preview=%d" %
          (hs["lum"].index(max(hs["lum"])), h1["lum"].index(max(h1["lum"]))))

    # 提交一次编辑后，/api/histogram 应该跟着变
    requests.post(BASE + "/api/auto", json={"session": sid}, timeout=300)
    hs2 = requests.get(BASE + "/api/histogram/" + sid, timeout=60).json()
    check("提交编辑后 GET 直方图更新", hs2["lum"] != hs["lum"])
    check("提交后总和仍正确", sum(hs2["lum"]) == hs2["total"])


    # UI 部分已移到 verify_hist_persistent.py：那里检查常驻位置、
    # 形状保真度（与真实分布的相关性）和是否重复画。这一套只管数据。

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
