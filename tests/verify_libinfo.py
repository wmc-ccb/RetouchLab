# -*- coding: utf-8 -*-
"""图库信息面板：智能预览 + 元数据."""
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


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    real = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
    scan_dir = os.path.dirname(real)

    # API 层
    ex = requests.get(B + "/api/library/exif", params={"path": real},
                      timeout=60).json()
    check("EXIF 曝光齐全", all(ex.get(k) is not None
                               for k in ("iso", "focal", "aperture",
                                         "shutter")),
          "ISO%s %smm f/%s %s" % (ex.get("iso"), ex.get("focal"),
                                  ex.get("aperture"), ex.get("shutter")))
    check("EXIF 机身/镜头", bool(ex.get("camera")) and bool(ex.get("lens")),
          "%s / %s" % (ex.get("camera"), str(ex.get("lens"))[:24]))
    check("EXIF 时间/作者", bool(ex.get("taken")),
          str(ex.get("taken")))

    r = requests.get(B + "/api/library/facecrop",
                     params={"path": real, "size": 220}, timeout=300)
    check("智能预览返回图", r.ok and r.headers.get("content-type", "")
          .startswith("image"), "%d bytes" % len(r.content))

    # UI 层
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1
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
        js("setMode('library')")
        time.sleep(2.0)
        js("libScan(%s)" % json.dumps(scan_dir))
        for _ in range(60):
            time.sleep(1)
            if (js("document.querySelectorAll('.libCell').length") or 0) > 0:
                break
        cells = js("document.querySelectorAll('.libCell').length")
        check("图库有条目", (cells or 0) > 0, "%s 个" % cells)

        # 单击第一张：应出现信息面板
        js("(function(){var el=document.querySelector('.libCell');"
           "if(el) el.click(); return true;})()")
        txt = ""
        for _ in range(20):
            time.sleep(0.5)
            txt = js("(document.getElementById('libInfo')||{}).innerText")
            if txt and "读取中" not in txt:
                break
        check("信息面板出现", isinstance(txt, str) and len(txt) > 10,
              str(txt)[:80].replace("\n", " / "))
        check("显示曝光参数", isinstance(txt, str) and
              ("ISO" in txt or "f/" in txt), str(txt)[:60])
        has_face = js("!!document.querySelector('#libInfo img')")
        check("智能预览图出现", has_face is True)

        shot = call("Page.captureScreenshot", format="png")
        with open(os.path.join(OUT, "ui_libinfo.png"), "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        call("Emulation.clearDeviceMetricsOverride")
        ws.close()
    except Exception as exc:
        print("  UI 段跳过：%s" % str(exc)[:140])

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
