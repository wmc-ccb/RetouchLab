# -*- coding: utf-8 -*-
"""验证打包版的界面能完整加载（双击可用性的最后一道关）。

只用一个标签页，全程超时保护，避免把机器拖死。
"""
import base64
import json
import os
import sys
import time

import requests
import websocket

CDP = "http://127.0.0.1:9223"
PORT = 8766
B = "http://127.0.0.1:%d" % PORT
OUT = r"C:\Users\Rana\ZhaJinWenQianCake\out"
FAIL = []


def check(name, ok, detail=""):
    print("%-30s %s %s" % (name, "OK  " if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def main():
    try:
        requests.get(CDP + "/json/version", timeout=5)
    except Exception:
        print("Chrome CDP 未运行，跳过界面验证")
        return 0

    # 打包服务必须先起（assemble 后手动启动，或启动.bat）。
    # 探不到就明确报告——绝不能继续导航标签页：这里曾经征用 tabs[0]
    # 导到 8766，测完不还原，把同批跑的其它 UI 测试全部毒死
    # （它们都在找 8765 的标签页）。
    try:
        requests.get(B + "/api/health", timeout=3)
    except Exception:
        print("RESULT: SKIPPED（%s 无打包服务，先启动 dist\\ZhaJinWenQianCake\\启动.bat）" % B)
        return 0

    # 用自己的标签页，用完即关：源码版的 8765 标签页不能动
    r = requests.put(CDP + "/json/new?" + B + "/", timeout=15)
    if not r.ok:
        print("建标签失败:", r.status_code)
        return 1
    target = r.json()

    ws = websocket.create_connection(target["webSocketDebuggerUrl"],
                                     timeout=60)
    n = [0]
    events = []

    def call(m, **p):
        n[0] += 1
        ws.send(json.dumps({"id": n[0], "method": m, "params": p}))
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                r = json.loads(ws.recv())
            except Exception:
                continue
            if r.get("method") in ("Runtime.exceptionThrown",
                                   "Runtime.consoleAPICalled",
                                   "Network.loadingFailed"):
                events.append(r)
            if r.get("id") == n[0]:
                return r.get("result", {})
        raise RuntimeError("CDP 超时: %s" % m)

    def js(expr):
        try:
            r = call("Runtime.evaluate", expression=expr, returnByValue=True,
                     awaitPromise=True).get("result", {})
            return r.get("value")
        except Exception as e:
            print("  js 失败:", str(e)[:60])
            return None

    call("Runtime.enable")
    call("Network.enable")
    call("Page.enable")
    call("Emulation.setDeviceMetricsOverride", width=1600, height=1000,
         deviceScaleFactor=1, mobile=False)
    call("Page.navigate", url=B + "/")
    time.sleep(8)

    print("\n=== 打包版界面 ===")
    title = js("document.title")
    check("页面标题", "ZhaJinWenQianCake" in (title or ""), str(title))

    rail = js("document.querySelectorAll('#rail .rbtn').length")
    check("左栏工具按钮", (rail or 0) >= 40, "%s 个" % rail)

    check("画布存在", js("!!document.getElementById('canvas')") is True)
    check("右栏存在", js("!!document.getElementById('side')") is True)

    cats = js("fetch('/api/cats').then(r=>r.status).catch(e=>'ERR')")
    check("/api/cats 可用", cats == 200, str(cats))

    extern = js("""
      (function(){
        var bad=[];
        document.querySelectorAll('link[href],script[src]')
          .forEach(function(e){
            var u=e.getAttribute('href')||e.getAttribute('src')||'';
            if(/^https?:/i.test(u)) bad.push(u);
          });
        return bad;
      })()
    """)
    check("无外部资源依赖", not (extern or []), str((extern or [])[:2]))

    exc = len([e for e in events
               if e.get("method") == "Runtime.exceptionThrown"])
    check("无 JS 异常", exc == 0, "%d 个" % exc)

    try:
        shot = call("Page.captureScreenshot", format="png")
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, "_packaged_ui.png")
        with open(p, "wb") as f:
            f.write(base64.b64decode(shot["data"]))
        print("\n截图:", p)
    except Exception as e:
        print("截图失败:", str(e)[:60])

    ws.close()
    # 关掉自己的标签页：留着会占 CDP，也会让"找 8765 标签页"的
    # 其它测试多一次无关匹配
    try:
        requests.get(CDP + "/json/close/" + target["id"], timeout=10)
    except Exception:
        pass
    print("\nRESULT: %s" % ("FAILED -> %s" % FAIL if FAIL else "ALL PASS"))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
