# -*- coding: utf-8 -*-
"""四季背景：引擎 + API + UI."""
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
    from engine.imgutil import load_image, fit_long_edge
    from engine import scene as SC

    img = fit_long_edge(load_image(SRC), 700)
    outs = {}
    for key, label in SC.PRESETS:
        out, info = SC.apply_scene(img, key)
        outs[key] = out
        check("%s 覆盖背景 %.0f%%" % (label, info["bg_frac"] * 100),
              0.3 < info["bg_frac"] < 0.95, str(info["bg_frac"]))

    # 人物区（掩膜内）逐像素不变——固定框会含背景角，掩膜才测得到保护
    from engine import semantic as SEM
    reg = SEM.segment(img)
    subj = np.clip(reg.subject(), 0, 1)
    subj = (subj > 0.5)
    for key in outs:
        d = float(np.abs(outs[key][subj] - img[subj]).mean())
        check("%s 人物掩膜内不动" % key, d < 0.02, "%.4f" % d)

    # 色彩主题：上部背景的主色调随季节变化
    h, w = img.shape[:2]
    top = slice(0, int(h * 0.15))
    means = {k: float(v[top].mean()) for k, v in outs.items()}
    check("各季节上部色调互异",
          max(means.values()) - min(means.values()) > 0.15,
          " ".join("%s=%.2f" % (k, v) for k, v in means.items()))
    check("输出全部合法",
          all(float(v.min()) >= 0 and float(v.max()) <= 1
              for v in outs.values()))

    # 未知 preset
    try:
        SC.apply_scene(img, "nope")
        check("未知 preset 抛错", False)
    except ValueError:
        check("未知 preset 抛错", True)

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    r = requests.post(B + "/api/scene", json={
        "session": up["session"], "preset": "winter"}, timeout=900)
    check("API winter", r.ok, str(r.status_code))
    if r.ok:
        j = r.json()
        check("API 返回图像", len(j.get("image", "")) > 5000,
              "%d bytes" % len(j.get("image", "")))
    r2 = requests.post(B + "/api/scene", json={
        "session": up["session"], "preset": "bad"}, timeout=60)
    check("未知 preset 400", r2.status_code == 400, str(r2.status_code))

    # ---------- UI ----------
    print("\n=== UI ===")
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
        js("selectCat('四季背景')")
        time.sleep(1.5)
        check("四季背景面板打开",
              js("document.getElementById('sideCat').textContent")
              == "四季背景")
        check("4 个季节按钮",
              (js("document.querySelectorAll('.scPre').length") or 0) == 4)
        check("换背景按钮存在", bool(js("!!document.getElementById('scRun')")))
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        time.sleep(0.3)
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_scene.png"), "wb") as fh:
                    fh.write(base64.b64decode(shot["data"]))
                break
            except Exception:
                time.sleep(1.0)
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
