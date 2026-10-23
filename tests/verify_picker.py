# -*- coding: utf-8 -*-
"""选图条与右侧图标栏的布局断言 + 截图."""
import base64
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
CDP = "http://127.0.0.1:9223"
failures = []


def check(label, cond, detail=""):
    print("%-50s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


class B:
    def __init__(self, t):
        self.ws = websocket.create_connection(t["webSocketDebuggerUrl"],
                                              timeout=90)
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
    t = [x for x in requests.get(CDP + "/json", timeout=10).json()
         if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not t:
        print("page not found")
        return 1
    b = B(t[0])
    b.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    b.call("Page.reload")
    time.sleep(4)

    # ---- 右侧图标栏 -----------------------------------------------------
    print("=== 图标栏位置 ===")
    pos = b.js("""
      (function(){
        var r=document.getElementById('rail').getBoundingClientRect();
        var s=document.getElementById('side').getBoundingClientRect();
        var c=document.getElementById('center').getBoundingClientRect();
        return {railL:Math.round(r.left), railR:Math.round(r.right),
                sideL:Math.round(s.left), sideR:Math.round(s.right),
                cL:Math.round(c.left), cR:Math.round(c.right),
                vw:window.innerWidth};
      })()
    """)
    if pos:
        print("   center %d-%d  side %d-%d  rail %d-%d  (vw %d)"
              % (pos["cL"], pos["cR"], pos["sideL"], pos["sideR"],
                 pos["railL"], pos["railR"], pos["vw"]))
        check("图标栏在参数面板右侧", pos["railL"] >= pos["sideR"] - 2,
              "railL=%d sideR=%d" % (pos["railL"], pos["sideR"]))
        check("图标栏贴最右", abs(pos["railR"] - pos["vw"]) <= 2,
              "railR=%d vw=%d" % (pos["railR"], pos["vw"]))
        check("参数面板在预览右侧", pos["sideL"] >= pos["cR"] - 2,
              "sideL=%d cR=%d" % (pos["sideL"], pos["cR"]))
        check("预览仍占主要宽度", (pos["cR"] - pos["cL"]) > pos["vw"] * 0.5,
              "%d px" % (pos["cR"] - pos["cL"]))
        check("图标栏可滚动（21 项）",
              b.js("getComputedStyle(document.getElementById('rail'))"
                   ".overflowY") in ("auto", "scroll"))

    # ---- 选图条 ----------------------------------------------------------
    print("\n=== 选图条 ===")
    with open(SRC, "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("c.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    b.js("""
      (function(){
        S.session=%s; S.meta=%s; S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        addThumb(S.base, 'DSC7550.NEF');
        addThumb(S.base, 'DSC7551.NEF');
        addThumb(S.base, 'DSC7552.jpg');
        drawBase(); return true; })()
    """ % (json.dumps(up["session"]),
           json.dumps({"width": up["width"], "height": up["height"],
                       "preview_width": up["preview_width"],
                       "preview_height": up["preview_height"]}),
           json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(1.5)

    check("选图条可见", b.js("getComputedStyle(document.getElementById("
                             "'filmbox')).display") != "none")
    check("缩略图数量 = 3",
          b.js("document.querySelectorAll('#fbStrip .shot').length") == 3,
          str(b.js("document.querySelectorAll('#fbStrip .shot').length")))
    check("顶栏有未筛选", b.js("(document.body.innerText||'')"
                               ".indexOf('未筛选')>=0") is True)
    check("顶栏有去挑图", b.js("(document.body.innerText||'')"
                               ".indexOf('去挑图')>=0") is True)
    check("顶栏有共 3 张计数",
          b.js("(document.getElementById('filmbox').innerText||'')"
               ".indexOf('共 3 张')>=0") is True,
          (b.js("document.getElementById('filmbox').innerText") or "")[:70])
    check("有 5 个星级", b.js("document.querySelectorAll('[data-star]')"
                              ".length") == 5)
    check("有色标", (b.js("document.querySelectorAll('[data-sw]').length") or 0) >= 5)
    check("有单视图/网格", b.js("!!document.getElementById('fbSingle') && "
                                "!!document.getElementById('fbGrid')"))
    check("RAW 标记已显示（.NEF）",
          (b.js("document.querySelectorAll('#fbStrip .raw').length") or 0) == 2,
          str(b.js("document.querySelectorAll('#fbStrip .raw').length")))

    # 标星：点第 3 颗
    b.js("document.querySelectorAll('[data-star]')[2].click()")
    time.sleep(0.4)
    check("点星标后 picked 为真", b.js("FILM.items[FILM.current].picked") is True,
          "stars=%s" % b.js("FILM.items[FILM.current].stars"))
    check("已选计数变 1", b.js("document.getElementById('filmbox')"
                               ".innerText.indexOf('已选 1')>=0") is True)

    # 双击缩略图标星
    b.js("document.querySelectorAll('#fbStrip .shot')[0].ondblclick()")
    time.sleep(0.4)
    check("双击缩略图标星", b.js("FILM.items[0].picked") is True)

    # 网格视图
    b.js("document.getElementById('fbGrid').click()")
    time.sleep(0.8)
    check("网格视图出现", b.js("!!document.getElementById('filmGrid')"))
    b.js("document.getElementById('fbSingle').click()")
    time.sleep(0.5)
    check("切回单视图后网格隐藏",
          b.js("(function(){var g=document.getElementById('filmGrid');"
               "return !g || g.style.display==='none';})()") is True)

    shot = b.shot(os.path.join(OUT, "ui_picker.png"))
    print("ui_picker.png %d KB" % shot)

    # 移动端：rail 隐藏、抽屉仍是浮层
    b.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
           deviceScaleFactor=2, mobile=True)
    b.call("Page.reload")
    time.sleep(3.5)
    check("移动端图标栏隐藏",
          b.js("getComputedStyle(document.getElementById('rail'))"
               ".display") == "none")
    check("移动端底部 tab 仍在",
          b.js("getComputedStyle(document.getElementById('mbar'))"
               ".display") == "flex")

    b.call("Emulation.clearDeviceMetricsOverride")
    b.ws.close()
    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
