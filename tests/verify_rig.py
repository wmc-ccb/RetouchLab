# -*- coding: utf-8 -*-
"""去道具：合成场景（有真值）+ API + UI."""
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


def synth():
    """灰蓝背景 + 灯架(杆+三脚架腿) + 白柔光箱 + 人物块."""
    import cv2

    h, w = 480, 360
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 0.55; img[..., 1] = 0.62; img[..., 2] = 0.70
    img[140:470, 248:253] = np.array([0.10, 0.10, 0.11], np.float32)
    cv2.line(img, (250, 400), (200, 470), (0.10, 0.10, 0.11), 4)
    cv2.line(img, (250, 400), (300, 470), (0.10, 0.10, 0.11), 4)
    img[60:140, 215:290] = np.array([0.93, 0.93, 0.92], np.float32)
    img[300:470, 60:130] = np.array([0.25, 0.18, 0.15], np.float32)
    return img


def main():
    from engine import rigfix as RF

    img = synth()

    # 识别
    m, info = RF.detect_rig(img)
    check("识别杆", info.get("poles") == 1, str(info.get("poles")))
    check("识别灯头", info.get("heads") == 1, str(info.get("heads")))
    check("杆中心入 mask", float(m[300, 250]) > 0.5)
    check("柔光箱入 mask", float(m[100, 250]) > 0.5)
    check("斜腿入 mask", float(m[440, 225]) > 0.3)
    check("人物区不入 mask", float(m[380, 90]) < 0.3)

    # 移除后：杆/箱位置变成背景色，人物原样
    out, finfo = RF.remove_rig(img)
    bgc = np.array([0.55, 0.62, 0.70], np.float32)
    dev_pole = float(np.abs(out[250, 250] - bgc).max())
    check("杆位补成背景色", dev_pole < 0.15, "%.3f" % dev_pole)
    dev_head = float(np.abs(out[100, 250] - bgc).max())
    check("柔光箱补成背景色", dev_head < 0.15, "%.3f" % dev_head)
    same_person = float(np.abs(out[380, 90] - img[380, 90]).max())
    check("人物区逐像素不动", same_person < 0.01, "%.4f" % same_person)
    check("输出无 NaN", np.isfinite(out).all())

    # 独立灯箱（无杆，天花挂的 LED 板/柔光箱）
    box_img = np.full((300, 400, 3), 0.62, np.float32)
    box_img[60:140, 150:260] = np.array([0.95, 0.95, 0.94], np.float32)
    mb, ib = RF.detect_rig(box_img, None)
    check("独立灯箱检出（无杆）", ib.get("lone_heads", 0) == 1 and
          float(mb[100, 200]) > 0.5,
          "lone=%s" % ib.get("lone_heads"))

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    r = requests.post(B + "/api/rigfix", json={"session": up["session"]},
                      timeout=900)
    check("API 室内图可跑（识别不到就少改）", r.ok, str(r.status_code))
    if r.ok:
        j = r.json()
        check("API 返回图像", len(j.get("image", "")) > 5000,
              "%d bytes" % len(j.get("image", "")))

    import cv2
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    up2 = requests.post(B + "/api/upload",
                        files={"file": ("r.png", buf.tobytes(),
                                        "image/png")}, timeout=300).json()
    r2 = requests.post(B + "/api/rigfix", json={"session": up2["session"]},
                       timeout=900)
    check("API 合成图识别杆+灯头", r2.ok and
          (r2.json().get("info") or {}).get("poles") == 1 and
          (r2.json().get("info") or {}).get("heads") == 1,
          str(r2.json().get("info", {}).get("poles")) + "/" +
          str(r2.json().get("info", {}).get("heads")) if r2.ok else "")

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
        js("selectCat('去道具')")
        time.sleep(1.5)
        check("去道具面板打开",
              js("document.getElementById('sideCat').textContent") == "去道具")
        check("移除按钮存在", bool(js("!!document.getElementById('rfRun')")))
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        time.sleep(0.3)
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_rig.png"), "wb") as fh:
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
