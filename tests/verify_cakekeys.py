# -*- coding: utf-8 -*-
"""蛋糕式快捷键：←/→ 选片、x 拒片、p 选片、Shift+1..5 星级、\ 对比."""
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
    ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"], timeout=60)
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

    def key(k, mods=None):
        mods = mods or {}
        js("""(function(){
          document.dispatchEvent(new KeyboardEvent('keydown',
            {key: %s, bubbles: true, %s}));
          return true;})()""" % (json.dumps(k),
                                 ", ".join("%s: true" % m for m in mods)))

    call("Page.reload")
    time.sleep(5)
    js("window.alert=function(){};window.confirm=function(){return false;};")

    # 造 3 个胶片条目
    js("""(function(){
      FILM.items = [
        {src:'data:image/gif;base64,R0lGODlhAQABAAAAACw=', name:'a.jpg', session:null, picked:false, stars:0},
        {src:'data:image/gif;base64,R0lGODlhAQABAAAAACw=', name:'b.jpg', session:null, picked:false, stars:0},
        {src:'data:image/gif;base64,R0lGODlhAQABAAAAACw=', name:'c.jpg', session:null, picked:false, stars:0}];
      FILM.current = 0;
      return true;})()""")

    # → 下一张 / ← 上一张
    key("ArrowRight")
    time.sleep(0.3)
    check("→ 下一张", js("FILM.current") == 1)
    key("ArrowRight")
    time.sleep(0.3)
    check("→ 再下一张", js("FILM.current") == 2)
    key("ArrowLeft")
    time.sleep(0.3)
    check("← 上一张", js("FILM.current") == 1)

    # p 选片
    key("p")
    time.sleep(0.3)
    check("p 选片", js("FILM.items[1].picked") is True)
    key("p")
    time.sleep(0.3)
    check("再按 p 取消", js("FILM.items[1].picked") is False)

    # x 拒片
    key("x")
    time.sleep(0.3)
    check("x 拒片", js("FILM.items[1].rejected") is True)
    # p 与拒片互斥
    key("p")
    time.sleep(0.3)
    check("选片取消拒片", js("FILM.items[1].rejected") is False and
          js("FILM.items[1].picked") is True)

    # Shift+1..5 星级
    key("3", ["shiftKey"])
    time.sleep(0.3)
    check("Shift+3 星级", js("FILM.items[1].stars") == 3)
    key("5", ["shiftKey"])
    time.sleep(0.3)
    check("Shift+5 星级", js("FILM.items[1].stars") == 5)

    # \ 对比（无会话不崩；有会话画原图）
    with open(SRC, "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    # 提交一步使历史有 2 步
    requests.post(B + "/api/wheel", json={
        "session": up["session"], "commit": True,
        "family": "red", "hue": 10}, timeout=600)
    js("(function(){S.session=%s; return true;})()" %
       json.dumps(up["session"]))
    key("\\")
    time.sleep(1.2)
    check("\\ 显示原图（无异常）", js(
        "beforeShowing") is True)
    js("""(function(){document.dispatchEvent(new KeyboardEvent('keyup',
      {key:'\\\\', bubbles:true})); return true;})()""")
    time.sleep(0.8)
    check("松开恢复", js("beforeShowing") is False)

    ws.close()
    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
