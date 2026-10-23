# -*- coding: utf-8 -*-
"""AI 祛路人按钮的 UI 链路（不真跑 LaMa，mock 掉耗时部分）."""
import base64
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CDP = "http://127.0.0.1:9223"
OUT = os.path.join(ROOT, "out")
FAIL = []


def check(n, ok, d=""):
    print("%-46s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
          if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not ts:
        print("浏览器未连接")
        return 1
    ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"], timeout=180)
    ni = [0]

    def call(m, **p):
        ni[0] += 1
        ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
        while True:
            r = json.loads(ws.recv())
            if r.get("id") == ni[0]:
                return r.get("result", {})

    def js(e):
        return call("Runtime.evaluate", expression=e, returnByValue=True,
                    awaitPromise=True).get("result", {}).get("value")

    call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
         deviceScaleFactor=1, mobile=False)
    call("Page.reload")
    time.sleep(4)

    # 载图（缩到小尺寸让 LaMa 快）
    with open(os.path.join(ROOT, "out", "passersby_demo.jpg"), "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        drawBase(); return true;})()""" % (
        json.dumps(up["session"]),
        json.dumps({"width": up["width"], "height": up["height"],
                    "preview_width": up["preview_width"],
                    "preview_height": up["preview_height"]}),
        json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(2)

    print("=== AI 祛路人按钮 ===")
    js("selectCat('背景消除')")
    time.sleep(2.0)
    check("面板打开", js("document.getElementById('sideCat').textContent")
          == "背景消除")
    check("祛路人按钮存在",
          bool(js("!!document.getElementById('bgcPassersby')")))
    check("说明文案存在",
          (js("document.querySelectorAll('#sliders .bstat').length") or 0) >= 3)

    # 点击并等待完成（图小，LaMa 应该在 60s 内）
    # 用 !! 转布尔：裸元素经 CDP 序列化的形态不稳定，直接判真值会误判
    info0 = None
    for _ in range(10):
        v = js("!!document.getElementById('bgcPbInfo')")
        if v is True:
            info0 = True
            break
        time.sleep(0.3)
    check("状态栏初始存在", info0 is True)
    js("document.getElementById('bgcPassersby').click()")
    done = False
    logtxt = ""
    for _ in range(90):
        time.sleep(1)
        logtxt = js("(function(){var e=document.getElementById('log');"
                    "return e ? e.innerText : '';})()") or ""
        if ("祛路人" in logtxt and
                ("消除" in logtxt or "没有" in logtxt or "失败" in logtxt)):
            done = True
            break
    check("点击后流程完成", done, logtxt[-120:].replace("\n", " / "))
    last = [l for l in logtxt.split("\n") if "祛路人" in l]
    check("日志报告结果", bool(last), (last[-1] if last else "")[:100])
    # requestRender 忙会让截图挂起（verify_levels 实测）——patch 后再截
    js("(function(){try{window.requestRender=function(){};"
       "return 1}catch(e){}})()")
    shot = None
    for _ in range(4):
        try:
            shot = call("Page.captureScreenshot", format="jpeg", quality=70)
            with open(os.path.join(OUT, "ui_passersby.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            break
        except Exception:
            time.sleep(1.0)

    call("Emulation.clearDeviceMetricsOverride")
    ws.close()
    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
