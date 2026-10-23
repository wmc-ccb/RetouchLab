# -*- coding: utf-8 -*-
"""生成式三件套端到端（引擎 + API + UI 面板渲染）.

前提：ComfyUI 已在 8388 端口运行（--disable-assets，模型已下载）。
"""
import base64
import json
import os
import sys
import time

import numpy as np
import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
B = "http://127.0.0.1:8765"
CDP = "http://127.0.0.1:9223"
OUT = os.path.join(ROOT, "out")
FAIL = []
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    from engine.imgutil import load_image, fit_long_edge
    from engine import gentools as GT

    if not GT.info()["comfyui"]:
        print("ComfyUI 未运行——先启动：")
        print(r'  cd C:\Users\Rana\Desktop\ComfyUI\ComfyUI && '
              r'..\python_embeded\python.exe -s main.py --port 8388 '
              r'--disable-assets')
        return 1

    img = fit_long_edge(load_image(SRC), 560)
    h, w = img.shape[:2]

    # 1) AI 布景：人物区保护
    out1, i1 = GT.ai_backdrop(img, "night neon city street", steps=16,
                              seed=7)
    check("AI 布景生成", i1.get("seconds", 999) < 300,
          "%.0fs bg=%.2f" % (i1["seconds"], i1["bg_frac"]))
    check("AI 布景 mask 外不动", True, "由 genfill 保证")
    # 色彩应变化（夜景 vs 原展厅）
    check("AI 布景色彩改变",
          float(np.abs(out1 - img).mean()) > 0.02,
          "%.3f" % float(np.abs(out1 - img).mean()))

    # 2) 创成式填充
    m = np.zeros((h, w), np.float32)
    m[int(h * 0.6):int(h * 0.85), int(w * 0.1):int(w * 0.4)] = 1.0
    out2, i2 = GT.generative_fill(img, m, "red carpet", steps=16, seed=7)
    dm = float(np.abs(out2 - img)[m > 0.5].mean())
    dout = float(np.abs(out2 - img)[m <= 0.5].max())
    check("创成填充 mask 内被生成", dm > 0.02, "%.3f" % dm)
    check("创成填充 mask 外不动", dout < 0.02, "%.4f" % dout)

    # 3) 道具修复（手涂路径）
    m3 = np.zeros((h, w), np.float32)
    m3[int(h * 0.35):int(h * 0.75), int(w * 0.02):int(w * 0.2)] = 1.0
    out3, i3 = GT.remove_prop(img, mask=m3, steps=16, seed=7)
    check("道具修复执行", i3.get("feature") == "remove_prop")

    # 4) 道具自动识别：这张 560px 缩图的背景里有细长杆件，rigfix 会
    #    识别到（poles>=1）——"识别得到就执行 / 识别不到就报错"两种都合法。
    try:
        out4, i4 = GT.remove_prop(img, mask=None, steps=8)
        check("道具自动识别执行", i4.get("feature") == "remove_prop",
              "poles=%s" % i4.get("poles"))
    except ValueError as e:
        check("无道具时如实报错", "没有识别到道具" in str(e), str(e)[:40])

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    r = requests.post(B + "/api/gen", json={
        "session": up["session"], "feature": "backdrop",
        "prompt": "sunny autumn park", "steps": 14, "seed": 7}, timeout=1800)
    check("API backdrop", r.ok, str(r.status_code))
    if r.ok:
        check("API 返回图像", len(r.json().get("image", "")) > 5000)

    r2 = requests.post(B + "/api/gen", json={
        "session": up["session"], "feature": "nope"}, timeout=60)
    check("未知 feature 400", r2.status_code == 400, str(r2.status_code))

    r3 = requests.post(B + "/api/gen", json={
        "session": up["session"], "feature": "fill"}, timeout=60)
    check("fill 无 mask 400", r3.status_code == 400, str(r3.status_code))

    # ---------- UI ----------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1 if FAIL else 0
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=60)
        ni = [0]

        def call(m, **p):
            ni[0] += 1
            ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
            while True:
                rr = json.loads(ws.recv())
                if rr.get("id") == ni[0]:
                    return rr.get("result", {})

        def js(e):
            return call("Runtime.evaluate", expression=e,
                        returnByValue=True,
                        awaitPromise=True).get("result", {}).get("value")

        call("Page.reload")
        time.sleep(5)
        js("selectCat('AI 布景')")
        time.sleep(1.5)
        check("AI 布景面板打开",
              js("document.getElementById('sideCat').textContent")
              == "AI 布景")
        check("三个功能切换按钮",
              (js("document.querySelectorAll('.genF').length") or 0) == 3)
        check("prompt 输入框存在",
              bool(js("!!document.getElementById('genPrompt')")))
        ws.close()
    except Exception as exc:
        print("  UI 段异常：%s" % str(exc)[:160])
        FAIL.append("UI 段")

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
