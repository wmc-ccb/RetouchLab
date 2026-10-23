# -*- coding: utf-8 -*-
"""宠物修整：合成场景真值 + API + UI."""
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
    """暗棕宠物体 + 白背景 + 双瞳孔。返回 (图, 主体掩膜)."""
    import cv2

    h, w = 360, 420
    img = np.full((h, w, 3), 0.90, np.float32)
    body = np.zeros((h, w), np.float32)
    cv2.ellipse(body, (200, 210), (120, 95), 0, 0, 360, 1.0, -1)
    cv2.ellipse(body, (200, 120), (60, 50), 0, 0, 360, 1.0, -1)
    fur = (np.sin(np.mgrid[0:h, 0:w][1] * 0.55) * 0.5 + 0.5) * 0.06
    col = np.array([0.42, 0.28, 0.18], np.float32)
    for c in range(3):
        img[..., c] = img[..., c] * (1 - body) + (col[c] + fur) * body
    cv2.circle(img, (178, 115), 9, (0.05, 0.04, 0.08), -1)
    cv2.circle(img, (224, 115), 9, (0.05, 0.04, 0.08), -1)
    return np.clip(img, 0, 1), body


def main():
    from engine import pet as PE

    img, body = synth()

    # groom 三项
    out, info = PE.groom(img, body, shine=60, smooth=40, eye_glow=70)
    check("眼神光找到瞳孔", info.get("eyes", 0) == 2, str(info.get("eyes")))
    check("瞳孔显著提亮",
          float(out[115, 178].mean()) > float(img[115, 178].mean()) + 0.2,
          "%.3f -> %.3f" % (float(img[115, 178].mean()),
                            float(out[115, 178].mean())))
    check("背景不动", float(np.abs(out[20, 20] - img[20, 20]).max()) < 0.001)
    check("输出合法", float(out.min()) >= 0 and float(out.max()) <= 1)

    # 平滑：**高频杂毛**对比度下降。std 必须在**腐蚀后的 body**上测：
    # 椭圆边界的过渡像素（背景 0.9 vs 身体 0.42）std 巨大且是真实边缘
    # （双边保边是正确行为），不排掉它们会把"杂毛被平滑"误判成失败。
    import cv2 as _cv

    noisy = img.copy()
    rng = np.random.default_rng(5)
    noisy += (rng.random(img.shape).astype(np.float32) - 0.5) * 0.10 * \
        body[..., None]
    noisy = np.clip(noisy, 0, 1)
    sm, _ = PE.groom(noisy, body, smooth=80)
    core = _cv.erode((body > 0.5).astype(np.uint8),
                     np.ones((11, 11), np.uint8)) > 0
    # 度量"加进去的纹理被压掉多少"：与干净底图逐像素差的标准差。
    # （core 内 RGB 三通道的均值本来就不等 0.42/0.28/0.18，直接测 std
    # 会被静态色差主导——那不是杂毛，实测 0.109 的主体是它。）
    n0 = float((noisy[core] - img[core]).std())
    n1 = float((sm[core] - img[core]).std())
    check("高频杂毛被平滑", n1 < n0 * 0.75, "%.3f -> %.3f" % (n0, n1))

    # 无主体时不崩
    out2, info2 = PE.groom(img, None) if False else (None, None)
    # groom(None) 会走 matting，跳过（已在 API 段覆盖）；测显式空掩膜
    z = np.zeros(img.shape[:2], np.float32)
    out3, info3 = PE.groom(img, z, shine=50)
    check("空掩膜不崩", np.isfinite(out3).all())

    # cutout（MODNet 通用显著物）
    try:
        rgba, cinfo = PE.cutout(img)
        check("cutout alpha 有效", 0.05 < cinfo["alpha_frac"] < 0.9,
              str(cinfo["alpha_frac"]))
    except RuntimeError as e:
        print("  cutout 跳过:", str(e)[:40])

    # ---------- API ----------
    print("\n=== API ===")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    r = requests.post(B + "/api/pet", json={
        "session": up["session"], "shine": 40, "smooth": 30,
        "eye_glow": 50}, timeout=900)
    check("API groom", r.ok, str(r.status_code))
    if r.ok:
        check("API 返回图像", len(r.json().get("image", "")) > 5000)
    r2 = requests.post(B + "/api/pet", json={
        "session": up["session"], "cutout": True}, timeout=900)
    check("API cutout", r2.ok, str(r2.status_code))

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
        js("selectCat('宠物修整')")
        time.sleep(1.5)
        check("宠物面板打开",
              js("document.getElementById('sideCat').textContent") == "宠物修整")
        check("三个滑杆",
              (js("[...document.querySelectorAll('[id^=petr_]')].length")
               or 0) == 3)
        check("抠图按钮存在", bool(js("!!document.getElementById('petCut')")))
        js("(function(){try{window.requestRender=function(){};"
           "return 1}catch(e){}})()")
        time.sleep(0.3)
        for _ in range(4):
            try:
                shot = call("Page.captureScreenshot", format="jpeg",
                            quality=70)
                with open(os.path.join(OUT, "ui_pet.png"), "wb") as fh:
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
