# -*- coding: utf-8 -*-
"""证件照面板 UI 链路."""
import base64
import json
import os
import sys
import time

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

    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch','btnAuto',
         'btnHeal','btnStamp','btnUpscale'].forEach(function(id){
           var e=document.getElementById(id); if(e) e.disabled=false;});
        drawBase(); return true;})()""" % (
        json.dumps(up["session"]),
        json.dumps({"width": up["width"], "height": up["height"],
                    "preview_width": up["preview_width"],
                    "preview_height": up["preview_height"]}),
        json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(2)

    print("=== 证件照面板 ===")
    js("selectCat('证件照')")
    time.sleep(2.0)
    check("面板打开", js("document.getElementById('sideCat').textContent")
          == "证件照")
    specs = js("document.querySelectorAll('.idSpec').length")
    check("5 种规格按钮", specs == 5, "%s 个" % specs)
    cols = js("document.querySelectorAll('.idCol').length")
    check("底色谱", (cols or 0) >= 3, "%s 个" % cols)
    check("生成按钮存在", bool(js("!!document.getElementById('idMake')")))
    check("导出按钮存在", bool(js("!!document.getElementById('idSave')")))

    # 选二寸 + 蓝底，点生成
    js("(function(){var b=document.querySelector('.idSpec[data-s=\"two_inch\"]');"
       "if(b) b.click(); return true;})()")
    time.sleep(0.4)
    js("(function(){var bs=document.querySelectorAll('.idCol');"
       "if(bs[1]) bs[1].click(); return true;})()")
    time.sleep(0.4)
    js("document.getElementById('idMake').click()")
    img = None
    for _ in range(60):
        time.sleep(1)
        img = js("(function(){var v=document.getElementById('idView');"
                 "var i=v?v.querySelector('img'):null;"
                 "return i?i.src.length:0;})()")
        if img and img > 1000:
            break
    check("生成并显示预览", bool(img) and img > 1000, "%s 字符" % img)
    dim = js("(function(){var v=document.getElementById('idView');"
             "return v?v.innerText:'';})()")
    check("显示尺寸信息", isinstance(dim, str) and "413" in dim, str(dim)[:40])
    logt = js("(function(){var e=document.getElementById('log');"
              "return e?e.innerText:'';})()") or ""
    check("日志报告生成", "证件照已生成" in logt,
          [l for l in logt.split("\n") if "证件照" in l][-1:][0][:60]
          if "证件照" in logt else "")

    shot = call("Page.captureScreenshot", format="png")
    with open(os.path.join(OUT, "ui_idphoto.png"), "wb") as fh:
        fh.write(base64.b64decode(shot["data"]))
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
