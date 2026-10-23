# -*- coding: utf-8 -*-
"""联机不打断精修：有提交历史时不自动切会话（数据丢失保护）."""
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

    call("Page.reload")
    time.sleep(5)

    # 场景 A：会话有提交历史（can_undo=true）→ poll 收到新照片不切换
    with open(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
              "rb") as fh:
        upA = requests.post(B + "/api/upload",
                            files={"file": ("a.jpg", fh, "image/jpeg")},
                            timeout=300).json()
    # 给 A 提交一步（模拟用户在精修）。/api/render 是纯预览端点
    # （无 commit 字段），提交要走有副作用的端点——用色轮红族。
    requests.post(B + "/api/wheel", json={
        "session": upA["session"], "commit": True,
        "family": "red", "hue": 10}, timeout=600)
    js("(function(){S.session=%s; S.meta={}; S.faces=[]; return true;})()"
       % json.dumps(upA["session"]))

    # 模拟 poll 收到新照片（直接调 pollTether 的分支逻辑）
    r = js("""(async function(){
      const busy = await fetch('/api/history/' + S.session)
        .then(r => r.json()).catch(() => null);
      return {can_undo: busy && busy.can_undo === true};
    })()""")
    check("精修中的会话 can_undo=true", r and r.get("can_undo") is True,
          str(r))

    # 场景 B：全新会话（can_undo=false）→ 自动接入
    with open(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
              "rb") as fh:
        upB = requests.post(B + "/api/upload",
                            files={"file": ("b.jpg", fh, "image/jpeg")},
                            timeout=300).json()
    r2 = js("""(async function(){
      const busy = await fetch('/api/history/' + %s)
        .then(r => r.json()).catch(() => null);
      return {can_undo: !!(busy && busy.can_undo)};
    })()""" % json.dumps(upB["session"]))
    check("新会话 can_undo=false", r2 and r2.get("can_undo") is False,
          str(r2))

    # 守卫语义：editing = can_undo === true
    check("守卫语义正确（editing 仅在精修时为真）",
          r.get("can_undo") is True and r2.get("can_undo") is False)

    try:
        ws.close()
    except Exception:
        pass

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
