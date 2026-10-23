# -*- coding: utf-8 -*-
"""胶片栏已修标记：done 表按 basename 匹配 → 条目 ✓ 角标."""
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
DONEF = os.path.join(ROOT, "server", "_library_done.json")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    if os.path.exists(DONEF):
        os.remove(DONEF)

    # 预置 done 表（文件名匹配即可命中）
    r = requests.post(B + "/api/library/mark_done", json={
        "src": SRC, "out": os.path.join(OUT, "_x.jpg")}, timeout=60)
    check("预置 done 记录", r.ok, str(r.status_code))

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

    js("window.alert = function(){};")
    call("Page.reload")
    time.sleep(5)

    # 拉 done 表 + 模拟胶片栏条目（名字与 done 的 basename 相同）
    r = js("""(async function(){
      const d = await fetch('/api/library/done').then(r => r.json());
      LIB.done = d.done || {};
      FILM.items = [{ src: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
                      name: '_DSC7550.jpg', session: null,
                      picked: false, stars: 0 },
                    { src: 'data:image/gif;base64,R0lGODlhAQABAAAAACw=',
                      name: '_DSC9999.jpg', session: null,
                      picked: false, stars: 0 }];
      FILM.current = 0;
      filmInit();
      const strip = document.getElementById('fbStrip');
      return {html: strip.innerHTML,
              hasDone: strip.innerHTML.indexOf('✓') >= 0,
              doneCount: (strip.innerHTML.match(/✓/g) || []).length};
    })()""")
    check("done 表已载入", r and r.get("doneCount") is not None)
    check("已修条目显示 ✓", r and r.get("doneCount") == 1,
          "%s 个 ✓（应为 1：仅 _DSC7550 已修）" % (r or {}).get("doneCount"))

    # 未修条目无 ✓
    only2 = js("""(function(){
      const shots = [...document.querySelectorAll('#fbStrip .shot')];
      return shots.map(function(s){
        return {i: s.dataset.i, done: s.innerHTML.indexOf('✓') >= 0};
      });
    })()""")
    check("条目 0 有 ✓", isinstance(only2, list) and only2[0].get("done") is True)
    check("条目 1 无 ✓", isinstance(only2, list) and
          only2[1].get("done") is False)

    try:
        ws.close()
    except Exception:
        pass
    if os.path.exists(DONEF):
        os.remove(DONEF)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
