# -*- coding: utf-8 -*-
"""Debug pass: console errors, failed requests, and dead panels.

Reads what the running app actually does rather than what the code intends:
every category is opened in turn with the console and network watched, so a
panel that throws on render or a request that 500s is caught by name.
"""
import json
import os
import sys
import time

import requests
import websocket

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDP = "http://127.0.0.1:9223"
BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


class B:
    def __init__(self, url):
        self.ws = websocket.create_connection(url, timeout=90)
        self.i = 0
        self.events = []

    def call(self, m, **p):
        self.i += 1
        self.ws.send(json.dumps({"id": self.i, "method": m, "params": p}))
        while True:
            r = json.loads(self.ws.recv())
            if "id" in r and r["id"] == self.i:
                return r.get("result", {})
            self.events.append(r)

    def js(self, e):
        return self.call("Runtime.evaluate", expression=e, returnByValue=True,
                         awaitPromise=True).get("result", {}).get("value")

    def drain(self):
        """Collect console/exception events that arrived without a request."""
        out = list(self.events)
        self.events = []
        self.ws.settimeout(0.25)
        try:
            while True:
                try:
                    out.append(json.loads(self.ws.recv()))
                except Exception:
                    break
        finally:
            self.ws.settimeout(90)
        msgs = []
        for ev in out:
            m = ev.get("method", "")
            p = ev.get("params", {})
            if m == "Runtime.consoleAPICalled":
                if p.get("type") in ("error", "warning", "assert"):
                    txt = " ".join(
                        str(a.get("value", a.get("description", "")))
                        for a in p.get("args", []))
                    msgs.append((p["type"], txt[:220]))
            elif m == "Runtime.exceptionThrown":
                d = p.get("exceptionDetails", {})
                msgs.append(("exception",
                             (d.get("text", "") + " " +
                              str(d.get("exception", {}).get("description", ""))
                              )[:220]))
            elif m == "Network.loadingFailed":
                msgs.append(("net-fail", p.get("errorText", "")))
            elif m == "Network.responseReceived":
                resp = p.get("response", {})
                if resp.get("status", 0) >= 400:
                    msgs.append(("http-%d" % resp["status"],
                                 resp.get("url", "")[:160]))
        return msgs


def main():
    ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
          if x.get("type") == "page" and "8765" in x.get("url", "")]
    if not ts:
        print("no page")
        return 1
    b = B(ts[0]["webSocketDebuggerUrl"])
    b.call("Runtime.enable")
    b.call("Network.enable")
    b.call("Page.enable")
    b.call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
           deviceScaleFactor=1, mobile=False)
    b.call("Page.reload")
    time.sleep(4)
    b.drain()

    # 载入一张图，让所有面板都有数据可渲染
    with open(SRC, "rb") as fh:
        up = requests.post(BASE + "/api/upload",
                           files={"file": ("d.jpg", fh, "image/jpeg")},
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
    b.drain()

    cats = b.js("[...document.querySelectorAll('#rail .rbtn .lbl')]"
                ".map(e => e.textContent.trim())")
    print("图标栏分类（%d）: %s" % (len(cats or []), cats))

    print("\n=== 逐个打开分类，收集控制台错误 ===")
    problems = []
    for i, name in enumerate(cats or []):
        b.js("document.querySelectorAll('#rail .rbtn')[%d].click()" % i)
        time.sleep(0.85)
        issues = b.drain()
        if issues:
            problems.append((name, issues))
            print("  %-8s %d 条" % (name, len(issues)))
            for kind, txt in issues[:3]:
                print("      [%s] %s" % (kind, txt))
    if not problems:
        print("  无控制台错误")

    # 空状态：不载图时打开每个分类
    print("\n=== 空状态（未载图）逐个打开 ===")
    b.call("Page.reload")
    time.sleep(3.5)
    b.drain()
    empty_problems = []
    for i, name in enumerate(cats or []):
        b.js("document.querySelectorAll('#rail .rbtn')[%d].click()" % i)
        time.sleep(0.6)
        issues = b.drain()
        if issues:
            empty_problems.append((name, issues))
            print("  %-8s %d 条" % (name, len(issues)))
            for kind, txt in issues[:2]:
                print("      [%s] %s" % (kind, txt))
    if not empty_problems:
        print("  无异常")

    # 端点巡检
    print("\n=== 端点巡检 ===")
    sid = up["session"]
    eps = [
        ("GET", "/api/health", None),
        ("GET", "/api/params", None),
        ("GET", "/api/presets", None),
        ("GET", "/api/faces/" + sid, None),
        ("GET", "/api/history/" + sid, None),
        ("GET", "/api/masks/" + sid, None),
        ("GET", "/api/histogram/" + sid, None),
        ("GET", "/api/auto_levels/" + sid, None),
        ("GET", "/api/formulas", None),
        ("GET", "/api/tether/status", None),
        ("GET", "/api/semantic/" + sid, None),
        ("GET", "/api/matte/" + sid, None),
        ("POST", "/api/render", {"session": sid, "params": {}, "quality": 85}),
        ("POST", "/api/auto", {"session": sid}),
        ("POST", "/api/masked", {"session": sid, "masks": ["subject"],
                                 "controls": {"exposure": 10},
                                 "commit": False}),
        ("POST", "/api/tether/poll", None),
        ("POST", "/api/undo", {"session": sid}),
    ]
    bad = 0
    for method, path, body in eps:
        t0 = time.time()
        try:
            if method == "GET":
                r = requests.get(BASE + path, timeout=600)
            else:
                r = requests.post(BASE + path, json=body or {}, timeout=600)
            dt = time.time() - t0
            flag = "OK " if r.status_code < 400 else "BAD"
            if r.status_code >= 400:
                bad += 1
            print("  %s %-34s %3d  %6.2fs" % (flag, path, r.status_code, dt))
        except Exception as exc:
            bad += 1
            print("  BAD %-34s %s" % (path, str(exc)[:80]))

    print("\n结论: 控制台问题 %d 处 / 空状态问题 %d 处 / 端点失败 %d 个"
          % (len(problems), len(empty_problems), bad))
    b.ws.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
