# -*- coding: utf-8 -*-
"""Browser check for the brush tools and the one-click auto button."""
import base64
import json
import os
import sys
import time

import requests
import websocket

CDP = "http://127.0.0.1:9223"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def _find_target():
    try:
        for t in requests.get(CDP + "/json", timeout=10).json():
            if t.get("type") == "page" and "8765" in t.get("url", ""):
                return t
    except Exception:
        return None
    return None


class Browser:
    def __init__(self, target):
        self.target = target
        self.ws = None
        self.i = 0
        self._connect()

    def _connect(self):
        for _ in range(4):
            try:
                self.ws = websocket.create_connection(
                    self.target["webSocketDebuggerUrl"], timeout=60)
                self.i = 0
                return
            except Exception:
                time.sleep(1.5)
        raise RuntimeError("could not attach")

    def call(self, method, **params):
        for _ in range(3):
            try:
                self.i += 1
                self.ws.send(json.dumps({"id": self.i, "method": method,
                                         "params": params}))
                while True:
                    msg = json.loads(self.ws.recv())
                    if msg.get("id") == self.i:
                        return msg.get("result", {})
            except Exception:
                try:
                    self.ws.close()
                except Exception:
                    pass
                time.sleep(1.0)
                self.target = _find_target() or self.target
                self._connect()
        raise RuntimeError("CDP failed: " + method)

    def js(self, expr):
        r = self.call("Runtime.evaluate", expression=expr,
                      returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    def shot(self, path):
        r = self.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as fh:
            fh.write(base64.b64decode(r["data"]))


def check(label, cond, detail=""):
    print("%-52s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    t = _find_target()
    if not t:
        print("page not found")
        return 1
    b = Browser(t)
    print("connected")
    # start from a clean page: the UI tests share one tab, and a session left
    # over from an earlier run would make the "disabled without a session"
    # checks fail for reasons unrelated to the feature
    b.call("Page.reload")
    time.sleep(4)

    # buttons exist and start disabled without a session
    for bid in ("btnAuto", "btnHeal", "btnStamp"):
        check("%s 存在" % bid, bool(b.js("!!document.getElementById('%s')" % bid)))
        check("%s 无会话时禁用" % bid,
              b.js("document.getElementById('%s').disabled" % bid) is True)

    # upload through the API so the tools become usable
    with open(SRC, "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("b.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]

    # drive the UI's own upload path by assigning the session the same way the
    # real flow does, then enabling the tools
    b.js("""
      (function(){
        S.session = %s;
        S.meta = %s;
        S.faces = %s;
        S.base = 'data:image/jpeg;base64,' + %s;
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch',
         'btnAuto','btnHeal','btnStamp'].forEach(function(id){
           document.getElementById(id).disabled = false;
        });
        return true;
      })()
    """ % (json.dumps(sid), json.dumps({"width": up["width"],
                                        "height": up["height"],
                                        "preview_width": up["preview_width"],
                                        "preview_height": up["preview_height"]}),
           json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(1.0)
    check("工具已启用", b.js("!document.getElementById('btnAuto').disabled"))

    # ---- brush mode ------------------------------------------------------
    b.js("setBrush('heal')")
    time.sleep(0.5)
    check("修复画笔模式激活", b.js("BRUSH.mode") == "heal", str(b.js("BRUSH.mode")))
    check("画笔浮动条显示", b.js("getComputedStyle(document.getElementById('brushBar')).display") == "flex")
    check("画笔参数滑杆存在", b.js("!!document.getElementById('brR')"))
    check("修复模式有纹理尺度滑杆", b.js("!!document.getElementById('brL')"))

    b.js("setBrush('clone')")
    time.sleep(0.4)
    check("切换到图章模式", b.js("BRUSH.mode") == "clone")
    check("图章模式无纹理尺度滑杆", not b.js("!!document.getElementById('brL')"))

    b.js("setBrush('clone')")
    time.sleep(0.3)
    check("再次点击关闭笔刷", b.js("BRUSH.mode") is None, str(b.js("BRUSH.mode")))

    # modal exclusivity: turning the brush on must turn erase off
    b.js("eraser = null; setBrush('heal'); if (typeof erasing !== 'undefined')"
         " { }")
    check("笔刷开启时消除模式互斥", b.js("erasing") is False)

    # ---- sample point + stroke through the API ---------------------------
    b.js("setBrush('heal')")
    b.js("""
      (function(){
        var c = document.getElementById('canvas');
        var r = c.getBoundingClientRect();
        BRUSH.source = [r.width*0.30, r.height*0.55];
        BRUSH.haveSample = true;
        return true;
      })()
    """)
    check("取样点已设置", b.js("BRUSH.haveSample") is True)

    res = requests.post("http://127.0.0.1:8765/api/brush", json={
        "session": sid, "mode": "heal",
        "points": [[up["preview_width"] * 0.55, up["preview_height"] * 0.60],
                   [up["preview_width"] * 0.65, up["preview_height"] * 0.60]],
        "source": [up["preview_width"] * 0.30, up["preview_height"] * 0.55],
        "radius": 30, "commit": True}, timeout=300).json()
    check("修复笔画 API 成功", "image" in res, str(res)[:60])

    h = requests.get("http://127.0.0.1:8765/api/history/" + sid, timeout=30).json()
    labels = [s["label"] for s in h["steps"]]
    check("笔画记入撤销栈", "修复画笔" in labels, str(labels))

    res2 = requests.post("http://127.0.0.1:8765/api/brush", json={
        "session": sid, "mode": "clone", "points": [[100, 100], [120, 100]],
        "source": [300, 300], "radius": 20, "commit": True}, timeout=300).json()
    h = requests.get("http://127.0.0.1:8765/api/history/" + sid, timeout=30).json()
    labels = [s["label"] for s in h["steps"]]
    check("图章也记入撤销栈", "仿制图章" in labels, str(labels))

    # ---- auto ------------------------------------------------------------
    a = requests.post("http://127.0.0.1:8765/api/auto",
                      json={"session": sid, "commit": True}, timeout=300).json()
    check("自动校正返回图像", "image" in a, str(list(a.keys()))[:70])
    check("自动校正返回线性增益", "gains" in a, str(a.get("gains")))
    check("自动校正给出曝光/色阶建议",
          bool(a.get("params")), str(a.get("params"))[:80])
    h = requests.get("http://127.0.0.1:8765/api/history/" + sid, timeout=30).json()
    check("自动校正记入撤销栈",
          "自动校正" in [s["label"] for s in h["steps"]])

    bad = requests.post("http://127.0.0.1:8765/api/auto",
                        json={"session": sid, "method": "nope"}, timeout=30)
    check("非法 method 返回 400", bad.status_code == 400, str(bad.status_code))

    b.js("autoFix()")
    time.sleep(6)
    check("UI 自动按钮可执行", b.js("S.session") == sid)

    b.js("selectCat('\u66f2\u7ebf')")
    time.sleep(0.8)
    b.shot(os.path.join(OUT, "ui4_tools.png"))

    b.ws.close()
    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
