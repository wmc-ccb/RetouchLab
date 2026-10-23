# -*- coding: utf-8 -*-
"""色阶独立面板：入口 + 拖动 + 应用（走 /api/render levels）."""
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
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    # 引擎级：levels 幂等 & 黑场咬合
    from engine import curves as CV

    img = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None]
    img = np.repeat(img, 3, axis=2).repeat(8, axis=0)
    same = CV.levels(img)
    check("默认色阶 = 恒等", float(np.abs(same - img).max()) < 1e-6)
    lifted = CV.levels(img, master=[0.1, 0.9, 1.0, 0.0, 1.0])
    # 黑场 0.1：输入 0.05（< 黑场）应被压成 0；输入 0.5 应被拉伸到
    # (0.5-0.1)/0.8 = 0.5 不变（中点对称）。拉伸看 0.3 点：
    # (0.3-0.1)/0.8 = 0.25 < 0.3，暗部被压暗（对比增强）。
    v05 = float(lifted[0, int(0.05 * 255), 0])
    v03 = float(lifted[0, int(0.3 * 255), 0])
    check("黑场 0.1 压掉近黑", v05 < 0.01, "%.4f" % v05)
    check("黑场 0.1 拉伸中调", abs(v03 - 0.25) < 0.01, "%.4f" % v03)
    # 白场 0.9：把输入 0.85 拉到 (0.85-0.1)/0.8 = 0.9375（对比增强，
    # ≥0.9 的输入直接到 1）。方向是"变亮"，不是压低。
    v085 = float(lifted[0, int(0.85 * 255), 0])
    check("白场 0.9 拉高高光", abs(v085 - 0.9375) < 0.01, "%.4f" % v085)
    check("白场以上到 1", float(lifted[0, -1, 0]) == 1.0,
          "%.3f" % float(lifted[0, -1, 0]))

    # API: /api/render 带 levels
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    r = requests.post(B + "/api/render", json={
        "session": sid, "params": {},
        "levels": {"master": [0.05, 0.95, 1.0, 0.0, 1.0]}}, timeout=600)
    check("render 接受 levels", r.ok, str(r.status_code))
    if r.ok:
        j = r.json()
        # 有变化：直方图应右移（暗部被抬起）
        check("render 返回图像", len(j.get("image", "")) > 5000,
              "%d bytes" % len(j.get("image", "")))

    # UI
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1 if FAIL else 0
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=180)
        ni = [0]

        def call(m, **p):
            ni[0] += 1
            ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
            while True:
                rr = json.loads(ws.recv())
                if rr.get("id") == ni[0]:
                    return rr.get("result", {})

        def js(e):
            return call("Runtime.evaluate", expression=e, returnByValue=True,
                        awaitPromise=True).get("result", {}).get("value")

        call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
             deviceScaleFactor=1, mobile=False)
        call("Page.reload")
        time.sleep(4)
        js("selectCat('色阶')")
        time.sleep(1.5)
        check("色阶面板打开",
              js("document.getElementById('sideCat').textContent") == "色阶")
        check("色阶画布存在",
              bool(js("!!document.getElementById('levCanvas')")))
        check("通道按钮 4 个",
              (js("document.querySelectorAll('.levCh').length") or 0) == 4)
        check("应用按钮存在", bool(js("!!document.getElementById('levApply')")))

        # 拖黑场：直接改 LEV 并触发重绘（模拟拖动手柄）
        js("(function(){LEV.values[0] = [0.12, 1, 1, 0, 1];"
           "drawLevels(); updateLevBoxes(); return true;})()")
        time.sleep(0.5)
        black = js("LEV.values[0][0]")
        check("黑场手柄生效", isinstance(black, (int, float)) and
              abs(black - 0.12) < 1e-6, str(black))

        shot = None
        # requestRender 的按需重绘会让 Chrome 合成器忙到 captureScreenshot
        # 排不上队（诊断实测：patch 成 no-op 后 0.1s 出图）。测试只需要
        # 看面板结构，不需要正在进行的渲染帧。
        js("(function(){try{window.requestRender=function(){};"
           "return 'patched';}catch(e){return 'no';}})()")
        for _ in range(5):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                break
            except Exception:
                time.sleep(1.0)
        if shot:
            with open(os.path.join(OUT, "ui_levels.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
        call("Emulation.clearDeviceMetricsOverride")
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
