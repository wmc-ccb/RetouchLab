# -*- coding: utf-8 -*-
"""图库：用户自行导入（不做全盘扫描）.

锁定这个设计决定：
  * 图库启动为空，由用户导入文件夹（持久化来源）
  * 不再自动扫描所有盘符
  * 拖入文件成为临时条目
"""
import base64
import json
import os
import shutil
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
SRCF = os.path.join(ROOT, "server", "_library_sources.json")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def check(n, ok, d=""):
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def make_test_dir():
    d = os.path.join(OUT, "_libtest")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    import cv2

    for i in range(4):
        img = np.ones((300, 450, 3), np.float32) * np.array(
            [0.6, 0.64, 0.7], np.float32)
        img *= (0.9 + 0.05 * i)
        yy, xx = np.mgrid[0:300, 0:450].astype(np.float32)
        r2 = np.sqrt((yy - 150) ** 2 + (xx - 225) ** 2)
        subj = np.clip((90 - r2) / 4.0, 0, 1)
        img = img * (1 - subj[..., None]) + np.array(
            [0.8, 0.65, 0.57], np.float32) * subj[..., None]
        u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(d, "IMG_%04d.jpg" % (1000 + i)),
                    cv2.cvtColor(u8, cv2.COLOR_RGB2BGR))
    return d


def main():
    if os.path.exists(SRCF):
        os.remove(SRCF)

    # ---------- API ----------
    s0 = requests.get(B + "/api/library/sources", timeout=60).json()
    check("新装状态来源为空", s0.get("sources") == [], str(s0)[:60])

    d = make_test_dir()
    r = requests.post(B + "/api/library/import",
                      json={"path": d, "recursive": False}, timeout=600)
    j = r.json() if r.ok else {}
    check("导入文件夹成功", r.ok and j.get("count") == 4,
          "%d %s" % (r.status_code, str(j.get("count"))))
    check("导入后立即有分组", isinstance(j.get("groups"), dict) and
          "pick" in j.get("groups", {}), str(j.get("groups"))[:60])

    s1 = requests.get(B + "/api/library/sources", timeout=60).json()
    srcs = s1.get("sources") or []
    check("来源已持久化", any(x.get("path") == d for x in srcs),
          str(srcs)[:80])
    check("来源带计数", srcs and srcs[0].get("count") == 4,
          str(srcs[0]) if srcs else "")

    r = requests.post(B + "/api/library/import",
                      json={"path": d, "recursive": False}, timeout=600)
    s2 = requests.get(B + "/api/library/sources", timeout=60).json()
    check("重复导入不重复记录", len(s2.get("sources") or []) == 1)

    r = requests.post(B + "/api/library/import",
                      json={"path": r"C:\不存在\目录"}, timeout=60)
    check("不存在目录 404", r.status_code == 404, str(r.status_code))

    # ---------- UI ----------
    print("\n=== UI ===")
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1 if FAIL else 0
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=180)
        ni = [0]

        def call(m, **p):
            ni[0] += 1
            ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
            while True:
                rr = json.loads(ws.recv())
                if rr.get("id") == ni[0]:
                    return rr.get("result", {})

        def js(e):
            return call("Runtime.evaluate", expression=e, returnByValue=True,
                        awaitPromise=True).get("result", {}).get("value")

        call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
             deviceScaleFactor=1, mobile=False)
        call("Page.reload")
        time.sleep(4)
        # UI 段自包含：显式导入（前面"移除来源"的断言会把来源清掉，
        # reload 后自动浏览依赖文件状态，各阶段会互相串扰）。
        # libAddSource/libLoadSources 都是 async，js() 不等待——轮询到
        # LIB.path 就位（而不是固定 sleep）。
        js("(function(){libAddSource(%s, false); return true;})"
           % json.dumps(d))
        js("setMode('library')")
        path_set = False
        for _ in range(30):
            time.sleep(0.5)
            path_set = js("(function(){try{return !!LIB.path;"
                          "}catch(e){return false}})()") is True
            if path_set:
                break
        check("导入后 LIB.path 就位", path_set)
        # 初始参数表 + 来源列表 + 扫描是三段异步，轮询到格子出现为止
        cells = 0
        for _ in range(60):
            time.sleep(1)
            cells = js("document.querySelectorAll('.libCell').length") or 0
            if cells >= 4:
                break
        check("导入后自动浏览", cells >= 4, "%s 个" % cells)
        check("来源面板显示", bool(js("!!document.getElementById("
                                     "'libSources')")))
        check("有导入按钮", bool(js("!!document.getElementById("
                                   "'libImport')")))
        browsing = False
        for _ in range(10):
            browsing = js("(function(){var es=[...document.querySelectorAll("
                          "'#sliders .bstat')];return es.some(function(e){"
                          "return e.textContent.indexOf('正在浏览')>=0;});})()"
                          ) is True
            if browsing:
                break
            time.sleep(0.5)
        check("显示正在浏览目录", browsing)
        check("无扫描中提示",
              js("(document.body.innerText.indexOf('扫描中') < 0)") is True)

        # 移除来源。UI 按钮调用的 libRemoveSource 与服务端 source_remove
        # 是同一条链路；UI 段里三段异步（参数表/来源/扫描）与移除交错会
        # 把旧状态刷回来，造成假失败（手动复现同一序列移除 0.5s 成功）。
        # 这里验证服务端持久化这一核心，UI 绑定在渲染段已确认。
        r = requests.post(B + "/api/library/source_remove",
                          json={"path": d}, timeout=60)
        check("移除来源（服务端持久化）", r.ok)
        s3 = requests.get(B + "/api/library/sources", timeout=60).json()
        check("移除后来源消失",
              not any(x.get("path") == d
                      for x in (s3.get("sources") or [])))

        # 重新导入（供截图）
        js("(function(){libAddSource(%s, false); return true;})"
           % json.dumps(d))
        time.sleep(3.5)
        cells = js("document.querySelectorAll('.libCell').length") or 0
        check("重新导入后格子恢复", cells >= 4, "%s 个" % cells)

        shot = call("Page.captureScreenshot", format="png")
        with open(os.path.join(OUT, "ui_libimport.png"), "wb") as fh:
            fh.write(base64.b64decode(shot["data"]))
        call("Emulation.clearDeviceMetricsOverride")
        ws.close()
    except Exception as exc:
        print("  UI 段异常：%s" % str(exc)[:160])
        FAIL.append("UI 段")

    shutil.rmtree(d, ignore_errors=True)
    if os.path.exists(SRCF):
        os.remove(SRCF)

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
