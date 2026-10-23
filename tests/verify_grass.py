# -*- coding: utf-8 -*-
"""AI 补草地：识别正确性 + LaMa 补全 + API + UI."""
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


def synth():
    """草地 + 背包 + 白纸（有真值）。"""
    import cv2

    rng = np.random.default_rng(3)
    h, w = 300, 400
    base = np.zeros((h, w, 3), np.float32)
    base[..., 0] = 0.30 + rng.normal(0, 0.02, (h, w))
    base[..., 1] = 0.48 + rng.normal(0, 0.02, (h, w))
    base[..., 2] = 0.22 + rng.normal(0, 0.02, (h, w))
    img = np.clip(base, 0, 1)
    cv2.ellipse(img, (150, 180), (26, 18), 20, 0, 360,
                (0.15, 0.13, 0.12), -1)
    cv2.rectangle(img, (280, 120), (315, 145), (0.92, 0.92, 0.90), -1)
    return img


def main():
    from engine import grassfix as GF

    img = synth()

    # 1) 识别
    m, info = GF.detect_debris(img)
    check("合成图识别出 2 处穿帮", info.get("debris_count") == 2,
          str(info))
    check("背包中心被标出", float(m[180, 150]) > 0.5)
    check("白纸中心被标出", float(m[132, 297]) > 0.5)
    check("纯草区不受影响", float(m[60, 60]) < 0.05)

    # 2) 无草地场景如实说明
    no_grass = np.full((300, 400, 3), 0.5, np.float32)
    m2, info2 = GF.detect_debris(no_grass)
    check("无草地时 applicable=false",
          info2.get("applicable") is False, str(info2.get("reason"))[:30])

    # 3) LaMa 补全：穿帮消失
    out, finfo = GF.fix_grass(img)
    check("补全后背包区变草地",
          float(np.abs(out[168:194, 138:164] -
                       img[168:194, 138:164]).mean()) > 0.05,
          "%.4f" % float(np.abs(out[168:194, 138:164] -
                                 img[168:194, 138:164]).mean()))
    check("补全区颜色贴近草地",
          abs(float(out[180, 150, 1]) - 0.48) < 0.12,
          "G=%.3f（草地 G≈0.48）" % float(out[180, 150, 1]))
    check("补全后无 NaN", np.isfinite(out).all())

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    r = requests.post(B + "/api/grassfix", json={"session": sid},
                      timeout=900)
    check("API 室内图 applicable=false", r.ok and
          (r.json().get("info") or {}).get("applicable") is False,
          str((r.json().get("info") or {}).get("reason"))[:30]
          if r.ok else r.text[:80])

    # 合成图走 API（PNG 上传）
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    import cv2
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    up2 = requests.post(B + "/api/upload",
                        files={"file": ("g.png", buf.tobytes(),
                                        "image/png")}, timeout=300).json()
    r2 = requests.post(B + "/api/grassfix", json={"session": up2["session"]},
                       timeout=900)
    check("API 草地图补全成功", r2.ok and
          (r2.json().get("info") or {}).get("applicable") is True,
          str((r2.json().get("info") or {}).get("debris_count"))
          if r2.ok else r2.text[:80])

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
        js("selectCat('补草地')")
        time.sleep(1.5)
        check("补草地面板打开",
              js("document.getElementById('sideCat').textContent") == "补草地")
        check("识别按钮存在", bool(js("!!document.getElementById('gfRun')")))
        check("强度滑杆存在", bool(js("!!document.getElementById("
                                     "'r_gfStrength')")))
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        time.sleep(0.3)
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_grass.png"), "wb") as fh:
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
