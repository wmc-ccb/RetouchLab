# -*- coding: utf-8 -*-
"""换天空：识别不误判 + 正向替换 + API + UI."""
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
OUTDOOR = os.path.join(OUT, "_outdoor.json")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"   # 室内


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    from engine import sky as SK

    # 1) 程序化天空纹理合法性
    tex = SK._sky_texture(200, 300, (0.3, 0.55, 0.95), (0.72, 0.85, 1.0),
                          (0.5, 0.18), 0.2, seed=42)
    check("天空纹理 0..1 且上深下浅",
          float(tex.min()) >= 0 and float(tex.max()) <= 1 and
          float(tex[10].mean()) < float(tex[-10].mean()),
          "%.2f -> %.2f" % (float(tex[10].mean()), float(tex[-10].mean())))

    # 2) 室内图不误判（这张图没有天空）
    from engine.imgutil import load_image, fit_long_edge
    from engine import semantic as SEM
    indoor = fit_long_edge(load_image(SRC), 700)
    bg = SEM.segment(indoor).norm("background")
    m, dinfo = SK.detect_sky(indoor, bg_mask=bg)
    check("室内图天空占比接近 0", dinfo["coverage"] < 0.02,
          "%.4f" % dinfo["coverage"])

    # 3) 户外图正向替换
    if os.path.exists(OUTDOOR):
        p = json.load(open(OUTDOOR))[0]
        img = fit_long_edge(load_image(p), 700)
        bg = SEM.segment(img).norm("background")
        out, info = SK.replace_sky(img, "clear_blue", bg_mask=bg)
        check("户外图识别出天空", 0.03 < info["coverage"] < 0.9,
              "%.3f" % info["coverage"])
        # 上部应显著偏蓝（新天空）
        top = out[: img.shape[0] // 6]
        blueness = float((top[..., 2] - (top[..., 0] + top[..., 1]) / 2)
                         .mean())
        check("换天后上部偏蓝", blueness > 0.05, "%+.3f" % blueness)
        # 人物区域几乎不动（取画面中下部做差）
        h = img.shape[0]
        body = slice(int(h * 0.6), h)
        delta = float(np.abs(out[body] - img[body]).mean())
        check("中下部（人物区）不被改写", delta < 0.01, "%.4f" % delta)
    else:
        print("跳过户外图（无候选）")

    # 4) 未知 preset 400
    try:
        SK.replace_sky(indoor, "nope")
        check("未知 preset 抛错", False)
    except ValueError:
        check("未知 preset 抛错", True)

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    r = requests.post(B + "/api/sky", json={
        "session": sid, "preset": "sunset", "strength": 1.0}, timeout=900)
    check("API 换天空（室内，几乎无变化）", r.ok, str(r.status_code))
    if r.ok:
        j = r.json()
        check("API 返回图像", len(j.get("image", "")) > 5000,
              "%d bytes" % len(j.get("image", "")))
        check("API info 含 coverage",
              isinstance((j.get("info") or {}).get("coverage"), float))
    r2 = requests.post(B + "/api/sky", json={
        "session": sid, "preset": "bad"}, timeout=60)
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
        js("selectCat('换天空')")
        time.sleep(1.5)
        check("换天空面板打开",
              js("document.getElementById('sideCat').textContent") == "换天空")
        check("6 个预设按钮",
              (js("document.querySelectorAll('.skyPre').length") or 0) == 6)
        check("强度滑杆存在", bool(js("!!document.getElementById("
                                     "'r_skyStrength')")))
        check("亮度匹配开关默认勾选",
              js("document.getElementById('skyMatch').checked") is True)
        # patch requestRender 让截图能出
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        time.sleep(0.3)
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_sky.png"), "wb") as fh:
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
