# -*- coding: utf-8 -*-
"""双栏参数面板 + 预设缩略图的布局断言与截图."""
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
                                              timeout=120)
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
        print("no page")
        return 1
    b = B(ts[0])
    b.call("Runtime.enable")
    b.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    b.call("Page.reload")
    time.sleep(4)

    with open(SRC, "rb") as fh:
        up = requests.post("http://127.0.0.1:8765/api/upload",
                           files={"file": ("c.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    b.js("""
      (function(){
        S.session=%s; S.meta=%s; S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch','btnAuto',
         'btnHeal','btnStamp','btnUpscale'].forEach(function(id){
           var e=document.getElementById(id); if(e) e.disabled=false; });
        addThumb(S.base,'DSC7550.NEF'); drawBase(); refreshHistory();
        return true; })()
    """ % (json.dumps(up["session"]),
           json.dumps({"width": up["width"], "height": up["height"],
                       "preview_width": up["preview_width"],
                       "preview_height": up["preview_height"]}),
           json.dumps(up["faces"]), json.dumps(up["preview"])))
    time.sleep(1.5)

    print("=== 双栏结构 ===")
    b.js("selectCat('\u9884\u8bbe')")
    time.sleep(4.0)          # 等缩略图渲染
    geo = b.js("""
      (function(){
        var pc=document.getElementById('presetCol'),
            pm=document.getElementById('paramCol'),
            sd=document.getElementById('side');
        var r1=pc.getBoundingClientRect(), r2=pm.getBoundingClientRect(),
            r3=sd.getBoundingClientRect();
        return {pcL:Math.round(r1.left), pcR:Math.round(r1.right),
                pmL:Math.round(r2.left), pmR:Math.round(r2.right),
                sdL:Math.round(r3.left), sdR:Math.round(r3.right),
                pcVis:getComputedStyle(pc).display};
      })()
    """)
    if geo:
        print("   预设栏 %d-%d  参数栏 %d-%d  面板 %d-%d"
              % (geo["pcL"], geo["pcR"], geo["pmL"], geo["pmR"],
                 geo["sdL"], geo["sdR"]))
        check("预设栏在参数栏左侧", geo["pcL"] <= geo["pmL"],
              "%d <= %d" % (geo["pcL"], geo["pmL"]))
        check("参数栏紧接预设栏", abs(geo["pmL"] - geo["pcR"]) <= 2,
              "pmL=%d pcR=%d" % (geo["pmL"], geo["pcR"]))
        check("两栏都在面板内", geo["pcL"] >= geo["sdL"] - 2 and
              geo["pmR"] <= geo["sdR"] + 2,
              "%d..%d in %d..%d" % (geo["pcL"], geo["pmR"],
                                    geo["sdL"], geo["sdR"]))
        check("预设栏可见", geo["pcVis"] != "none")

    ncards = b.js("document.querySelectorAll('.pcard').length")
    check("预设卡片已渲染", (ncards or 0) >= 7, "%s 张" % ncards)
    nimg = b.js("[...document.querySelectorAll('[data-thumb]')]"
                ".filter(e => e.src && e.src.indexOf('data:image')===0).length")
    check("预设缩略图已生成", (nimg or 0) >= 5, "%s/%s 张" % (nimg, ncards))

    # 缩略图必须是不同画面（不是同一张图重复）
    uniq = b.js("""
      (function(){
        var s=new Set();
        document.querySelectorAll('[data-thumb]').forEach(function(e){
          if(e.src && e.src.indexOf('data:image')===0) s.add(e.src.length+':'+e.src.slice(-40));
        });
        return s.size; })()
    """)
    check("缩略图各不相同（确实套了不同预设）", (uniq or 0) >= 5,
          "%s 个不同" % uniq)

    # 点一个预设，等它真正完成。
    # `applyPresetByName` 是 async：card.onclick() 立刻返回一个 Promise，
    # 日志是渲染接口回来之后才写的。之前固定 sleep 后就读日志，读到的
    # 永远是上一条，于是误判成"点击没生效"。
    ready = 0
    for _ in range(40):
        ready = b.js("(function(){var c=document.querySelectorAll('.pcard');"
                     "if(!c.length) return 0;"
                     "return [...c].filter(function(x){var i=x.querySelector('img');"
                     "return i&&i.src&&i.src.indexOf('data:image')===0;}).length;})()")
        if (ready or 0) >= 7:
            break
        time.sleep(0.5)
    check("等缩略图全部就绪", (ready or 0) >= 7, "%s/7" % ready)

    card = b.js("(function(){var c=document.querySelectorAll('.pcard')[3];"
                "return c ? c.dataset.p : null;})()")
    before_log = b.js("document.getElementById('log').innerText") or ""
    b.js("window.__applied = null; document.querySelectorAll('.pcard')[3]"
         ".onclick()")
    logtxt = ""
    for _ in range(40):
        time.sleep(0.5)
        logtxt = b.js("document.getElementById('log').innerText") or ""
        if "预设已应用" in logtxt and logtxt != before_log:
            break
    check("点预设后日志报告应用项数", "预设已应用" in logtxt,
          (logtxt.strip().splitlines() or [""])[0][:60])
    check("预设参数被管线识别（applied 非空）", "（0 项）" not in logtxt
          and "预设已应用" in logtxt, "卡片=%s" % card)
    check("被点的卡片高亮",
          b.js("!!document.querySelector('.pcard.active')"))

    # 非参数分类应隐藏预设栏，参数区占满
    b.js("selectCat('\u66f2\u7ebf')")
    time.sleep(1.2)
    check("曲线分类下隐藏预设栏",
          b.js("getComputedStyle(document.getElementById('presetCol'))"
               ".display") == "none")
    wid = b.js("Math.round(document.getElementById('paramCol')"
               ".getBoundingClientRect().width)")
    check("隐藏后参数区变宽", (wid or 0) > 380, "%s px" % wid)

    b.js("selectCat('\u9884\u8bbe')")
    time.sleep(1.5)
    print("ui_dualcol.png %d KB" % b.shot(os.path.join(OUT, "ui_dualcol.png")))

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
