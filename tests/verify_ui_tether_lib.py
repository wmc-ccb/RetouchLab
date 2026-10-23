# -*- coding: utf-8 -*-
"""联机面板 + 图库 + 背景消除截图与断言."""
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


class C:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=180)
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
    ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
          if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not ts:
        print("浏览器未连接")
        return 1
    c = C(ts[0])
    c.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    c.call("Page.reload")
    time.sleep(4)

    # 载一张图
    with open(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
              "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    c.js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch','btnAuto',
         'btnHeal','btnStamp','btnUpscale'].forEach(function(id){
           var e=document.getElementById(id); if(e) e.disabled=false;});
        drawBase(); refreshHistory(); return true;})()""" % (
        json.dumps(up["session"]),
        json.dumps({"width": up["width"], "height": up["height"],
                    "preview_width": up["preview_width"],
                    "preview_height": up["preview_height"]}),
        json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(2)

    print("=== 联机面板 ===")
    c.js("selectCat('联机')")
    time.sleep(3.0)
    check("联机面板打开", c.js("document.getElementById('sideCat')"
                               ".textContent") == "联机")
    check("有相机区块", bool(c.js("!!document.getElementById('tethCam')")))
    msg = c.js("document.getElementById('tethCamMsg').innerText") or ""
    check("相机状态有内容", len(msg) > 10, msg.replace("\n", " / ")[:70])
    check("如实说明不能控快门", "未提供拍照命令" in msg or "可以控制快门" in msg)
    check("给出替代方案", "文件夹监听" in msg)
    c.shot(os.path.join(OUT, "ui_tether2.png"))

    print("\n=== 背景消除面板 ===")
    c.js("selectCat('背景消除')")
    time.sleep(2.0)
    check("面板打开", c.js("document.getElementById('sideCat')"
                           ".textContent") == "背景消除")
    n = c.js("document.querySelectorAll('[id^=bgc_][type=checkbox]').length")
    check("四项开关", (n or 0) >= 4, "%s 个" % n)
    check("换底色色板",
          (c.js("document.querySelectorAll('.bgcSw').length") or 0) >= 5)
    c.shot(os.path.join(OUT, "ui_bgclean2.png"))

    print("\n=== 图库 ===")
    c.js("setMode('library')")
    time.sleep(2.5)
    check("图库模块", c.js("(document.querySelector('.mode.active')||{})"
                           ".dataset && document.querySelector('.mode.active')"
                           ".dataset.mode") == "library")
    check("图库分组按钮",
          (c.js("document.querySelectorAll('.libF').length") or 0) >= 5)
    c.shot(os.path.join(OUT, "ui_library2.png"))

    c.call("Emulation.clearDeviceMetricsOverride")
    c.ws.close()
    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
