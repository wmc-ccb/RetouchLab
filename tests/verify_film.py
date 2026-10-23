# -*- coding: utf-8 -*-
"""胶片栏交互：星级 / 选片 / 拒片 / 视图切换."""
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

    # 载 3 张（同一张反复用，重点在交互）
    js("(function(){window.prompt=function(){return null;};"
       "window.confirm=function(){return true;};return true;})()")
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("a.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
        S.base='data:image/jpeg;base64,'+%s;
        document.getElementById('dropzone').style.display='none';
        ['btnExport','btnErase','btnPicker','btnMatte','btnMatch','btnAuto',
         'btnHeal','btnStamp','btnUpscale'].forEach(function(id){
           var e=document.getElementById(id); if(e) e.disabled=false;});
        drawBase();
        ['a.jpg','b.jpg','c.jpg'].forEach(function(n){
          addThumb(S.base, n);});
        return true;})()""" % (
        json.dumps(up["session"]),
        json.dumps({"width": up["width"], "height": up["height"],
                    "preview_width": up["preview_width"],
                    "preview_height": up["preview_height"]}),
        json.dumps(up["faces"]),
        json.dumps(up["preview"])))
    time.sleep(2)

    print("=== 胶片栏结构 ===")
    n = js("document.querySelectorAll('#fbStrip .shot').length")
    check("缩略图条 3 张", n == 3, "%s 张" % n)
    check("星级控件 5 个",
          (js("document.querySelectorAll('#filmbox [data-star]').length") or 0)
          == 5)
    check("当前文件名显示",
          (js("(document.querySelector('#filmbox .curName')||{}).textContent")
           == "c.jpg"),
          str(js("(document.querySelector('#filmbox .curName')||{})"
                 ".textContent")))

    print("\n=== 星级 ===")
    # 合成点击用 dispatchEvent（带 bubbles）。选择器内层用双引号：
    # 外层 Python 字符串已用单引号包裹，再嵌 \' 会传给 JS 一个语法错误
    # 表达式（实测返回 null，点击静默无效）。
    js("var el=document.querySelector(\"#filmbox [data-star='4']\");"
       "el.dispatchEvent(new MouseEvent('click',{bubbles:true}));")
    time.sleep(0.5)
    st = js("FILM.items[FILM.current].stars")
    pk = js("FILM.items[FILM.current].picked")
    check("点 4 星 -> stars=4 picked=true", st == 4 and pk is True,
          "stars=%s picked=%s" % (st, pk))
    on = js("document.querySelectorAll('#filmbox .star.on').length")
    check("4 颗星高亮", on == 4, "%s 颗" % on)
    js("var el=document.querySelector(\"#filmbox [data-star='4']\");"
       "el.dispatchEvent(new MouseEvent('click',{bubbles:true}));")
    time.sleep(0.4)
    check("再点同星 -> 清零",
          js("FILM.items[FILM.current].stars") == 0)

    print("\n=== 选片/拒片 ===")
    js("(function(){FILM.items[0].picked=true;FILM.items[0].stars=5;"
       "FILM.items[1].rejected=true;filmInit();return true;})()")
    time.sleep(0.5)
    check("已选计数 1",
          (js("(document.querySelector('#filmbox b')||{}).textContent") == "1"))
    rej = js("document.querySelectorAll('#fbStrip .shot.rej').length")
    check("拒片标记", rej == 1, "%s 个" % rej)

    print("\n=== 过滤 ===")
    # 三态循环：all -> pick（1张已选） -> unpicked（2张） -> all
    js("document.getElementById('fbUnfilter').click()")   # -> pick
    time.sleep(0.5)
    vis = js("document.querySelectorAll('#fbStrip .shot').length")
    check("已选过滤只显示 1 张", vis == 1, "%s 张" % vis)
    js("document.getElementById('fbUnfilter').click()")   # -> unpicked
    time.sleep(0.5)
    vis = js("document.querySelectorAll('#fbStrip .shot').length")
    check("未选过滤显示 2 张", vis == 2, "%s 张" % vis)
    js("document.getElementById('fbUnfilter').click()")   # -> all
    time.sleep(0.4)
    vis = js("document.querySelectorAll('#fbStrip .shot').length")
    check("回到全部 3 张", vis == 3, "%s 张" % vis)

    print("\n=== 视图切换 ===")
    js("document.getElementById('fbGrid').click()")
    time.sleep(0.8)
    check("网格视图出现",
          js("!!document.getElementById('filmGrid')") is True)
    js("document.getElementById('fbGrid').click()")
    time.sleep(0.5)

    shot = call("Page.captureScreenshot", format="png")
    with open(os.path.join(OUT, "ui_film.png"), "wb") as fh:
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
