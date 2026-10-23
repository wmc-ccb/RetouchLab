# -*- coding: utf-8 -*-
"""左侧工具条：结构、行为、与已有功能的联动."""
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

    print("=== 结构 ===")
    n = c.js("document.querySelectorAll('#ltools .ltBtn').length")
    check("左侧工具条有按钮", (n or 0) >= 10, "%s 个" % n)
    labels = c.js("[...document.querySelectorAll('#ltools .l')]"
                  ".map(function(e){return e.textContent})")
    print("   标签:", labels)
    for want in ("缩放", "适合", "抓手", "祛边", "参考线", "液化",
                 "污点修复", "修补", "仿制图章", "智能消除", "AI点修"):
        check("有「%s」" % want, isinstance(labels, list) and want in labels)
    # 不做免费/付费这类商业分级：工具条上不应出现 Free 徽章
    nbadge = c.js("document.querySelectorAll('#ltools .badge').length") or 0
    check("无 Free 徽章", nbadge == 0, "%s 个 badge" % nbadge)
    check("页面无 Free 字样",
          c.js("(document.body.innerText.indexOf('Free') < 0)") is True)
    # 5 个功能组之间是 4 条分隔线，这是正确数量
    check("有分隔线",
          (c.js("document.querySelectorAll('#ltools .ltSep').length") or 0) >= 4,
          "%s 条" % c.js("document.querySelectorAll('#ltools .ltSep').length"))
    geo = c.js("""
      (function(){
        var t=document.getElementById('ltools').getBoundingClientRect();
        var c=document.getElementById('center').getBoundingClientRect();
        return {tRight:Math.round(t.right), cLeft:Math.round(c.left),
                tWidth:Math.round(t.width)};
      })()
    """)
    check("工具条在画布左侧", geo and geo["tRight"] <= geo["cLeft"] + 2,
          "工具条右 %s / 画布左 %s" % (geo["tRight"], geo["cLeft"]))

    print("\n=== 行为 ===")
    # 抓手
    c.js("document.getElementById('ltPan').click()")
    time.sleep(0.4)
    check("抓手打开后高亮",
          c.js("document.getElementById('ltPan').classList.contains('active')")
          is True)
    c.js("document.getElementById('ltPan').click()")
    time.sleep(0.3)

    # 参考线
    c.js("document.getElementById('ltGuide').click()")
    time.sleep(0.5)
    check("参考线出现",
          c.js("(function(){var g=document.getElementById('guideLayer');"
               "return !!g && getComputedStyle(g).display!=='none';})()")
          is True)
    c.js("document.getElementById('ltGuide').click()")
    time.sleep(0.3)

    # 仿制图章 → 应进入 clone 模式
    c.js("document.getElementById('ltStamp').click()")
    time.sleep(0.6)
    check("仿制图章进入 clone 模式",
          c.js("(typeof BRUSH!=='undefined' && BRUSH.mode==='clone')") is True,
          str(c.js("typeof BRUSH!=='undefined' ? BRUSH.mode : 'no BRUSH'")))
    check("工具条按钮与模式同步高亮",
          c.js("document.getElementById('ltStamp')"
               ".classList.contains('active')") is True)
    c.js("document.getElementById('ltStamp').click()")
    time.sleep(0.3)

    # 祛边 → 切到祛边面板
    c.js("document.getElementById('ltDefringe').click()")
    time.sleep(1.5)
    check("祛边按钮切到祛边面板",
          c.js("document.getElementById('sideCat').textContent") == "祛边",
          str(c.js("document.getElementById('sideCat').textContent")))

    c.js("document.getElementById('ltGuide').click()")
    time.sleep(0.4)
    c.shot(os.path.join(OUT, "ui_ltools.png"))
    c.js("document.getElementById('ltGuide').click()")

    print("\n=== 移动端 ===")
    c.call("Emulation.setDeviceMetricsOverride", width=390, height=844,
           deviceScaleFactor=2, mobile=True)
    c.call("Page.reload")
    time.sleep(3.5)
    check("移动端隐藏左工具栏",
          c.js("getComputedStyle(document.getElementById('ltools')).display")
          == "none")
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
