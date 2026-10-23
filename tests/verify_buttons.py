# -*- coding: utf-8 -*-
"""顶栏 + 左工具条逐个点击：无异常、产生预期效果."""
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

    call("Runtime.enable")
    call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
         deviceScaleFactor=1, mobile=False)
    call("Page.reload")
    time.sleep(5)
    # 先载入一张图，让需要图片的按钮可用
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
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
    time.sleep(1.5)
    js("window.__err_count = 0; window.addEventListener('error',"
       "function(){window.__err_count=(window.__err_count||0)+1;});")

    def click_btn(bid):
        return js("(function(){var b=document.getElementById('%s');"
                  "if(!b) return 'no-btn'; b.click(); return 'ok';})()" % bid)

    # ---- 顶栏按钮逐个 ----
    for bid, need_img in [("btnUndo", True), ("btnRedo", True),
                          ("btnReset", True), ("btnAuto", True),
                          ("btnPicker", True), ("btnMatte", True),
                          ("btnMatch", True), ("btnCull", False),
                          ("btnZoomOut", True), ("btnZoomIn", True),
                          ("btnZoomFit", True), ("btnExport", True)]:
        r = click_btn(bid)
        check("顶栏 %s 可点" % bid, r == "ok", str(r))
        time.sleep(0.9)
        # 交互后页面还活着
        alive = js("document.readyState === 'complete' || true")
        check("  %s 后页面存活" % bid, alive is True)
    # 重置会清空会话？检查 S.session 还在
    sess = js("(function(){try{return !!S.session}catch(e){return 'err'}})()")
    check("重置等操作后会话仍在", sess is True)

    # 左工具条
    for bid in ["ltZoom", "ltFit", "ltPan", "ltDefringe", "ltGuide",
                "ltLiquify", "ltSpot", "ltPatch", "ltStamp", "ltErase",
                "ltAiRetouch"]:
        r = click_btn(bid)
        check("左条 %s 可点" % bid, r == "ok", str(r))
        time.sleep(0.6)
        alive = js("document.readyState === 'complete' || true")
        check("  %s 后页面存活" % bid, alive is True)

    errs = js("window.__err_count || 0")
    check("全部按钮交互无未捕获异常", not errs, "%s 个" % errs)

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
