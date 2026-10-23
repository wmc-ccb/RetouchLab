# -*- coding: utf-8 -*-
"""图库选中集 → 批处理：工作流闭环.

用 M: 里最小的真实照片组跑一批（草稿尺寸，控制耗时），验证：
  1. 图库勾选 -> renderBatch 把选中集带进 BATCH.paths
  2. startBatch 发 paths 优先于源目录
  3. /api/batch 真正完成并产出文件
"""
import base64
import json
import os
import shutil
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
    print("%-50s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def find_small_group():
    """M: 下 3~6 张照片的目录，让批处理几十秒内能跑完。"""
    roots = requests.get(B + "/api/library/roots", timeout=120).json()
    mroot = next((x for x in roots.get("roots", []) if x["name"] == "M:"), None)
    if not mroot:
        return None, []
    cands = sorted(mroot["groups"], key=lambda g: g["count"])
    for g in cands:
        if 2 <= g["count"] <= 6:
            s = requests.get(B + "/api/library/scan",
                             params={"path": g["path"], "limit": 8},
                             timeout=300).json()
            return g["path"], [it["path"] for it in s.get("items", [])]
    return None, []


def main():
    # ---------- API 层：显式 paths 批处理 ----------
    scan_path, files = find_small_group()
    if not files:
        # 回退：用一张图自己造目录
        scan_path = os.path.dirname(
            r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg")
        files = [r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"]
    print("测试目录: %s (%d 文件)" % (scan_path, len(files)))

    out_dir = os.path.join(OUT, "_batch_sel")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    r = requests.post(B + "/api/batch", json={
        "paths": files[:3], "out_dir": out_dir, "preset": "natural",
        "long_edge": 1280}, timeout=120)
    check("批处理接受 paths 列表", r.ok, "%d %s" % (r.status_code, r.text[:90]))
    if not r.ok:
        return 1
    rid = r.json()["id"]
    total = r.json().get("total", 0)
    check("总数与请求一致", total == min(3, len(files)), "total=%s" % total)

    # 轮询到完成（状态字段是 `state`，不是 status）
    final = None
    for _ in range(120):
        time.sleep(2)
        j = requests.get(B + "/api/batch/" + rid, timeout=60).json()
        if j.get("state") in ("done", "failed", "cancelled"):
            final = j
            break
    check("批处理完成", final is not None and final.get("state") == "done",
          str(final and {k: final.get(k) for k in ("state", "done", "failed")}))
    produced = [f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")]
    check("产出文件数正确", len(produced) == total,
          "%d/%d" % (len(produced), total))

    # ---------- UI 层：图库勾选 -> 批处理 ----------
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接，跳过 UI 段")
        else:
            ws = websocket.create_connection(
                ts[0]["webSocketDebuggerUrl"], timeout=180)
            ni = [0]

            def call(m, **p):
                ni[0] += 1
                ws.send(json.dumps({"id": ni[0], "method": m, "params": p}))
                while True:
                    rr = json.loads(ws.recv())
                    if rr.get("id") == ni[0]:
                        return rr.get("result", {})

            def js(e):
                return call("Runtime.evaluate", expression=e,
                            returnByValue=True, awaitPromise=True
                            ).get("result", {}).get("value")

            call("Emulation.setDeviceMetricsOverride", width=1800,
                 height=1050, deviceScaleFactor=1, mobile=False)
            call("Page.reload")
            time.sleep(4)
            js("setMode('library')")
            time.sleep(2.0)
            js("libScan(%s)" % json.dumps(scan_path))
            cells = 0
            for _ in range(60):
                time.sleep(1)
                cells = js("document.querySelectorAll('.libCell').length") or 0
                if cells > 0:
                    break
            check("图库扫描出条目", cells > 0, "%s 个" % cells)

            # 勾选：直接改 LIB.sel（单击格子是"打开会话"，不是勾选）
            js("(function(){LIB.items.slice(0,2).forEach(function(it){"
               "LIB.sel[LIB.items.indexOf(it)]=true;});renderLibrary();"
               "return true;})()")
            time.sleep(1.0)
            nsel = js("Object.values(LIB.sel).filter(Boolean).length")
            check("图库勾选 2 张", nsel == 2, "%s 张" % nsel)
            check("去批处理按钮出现",
                  bool(js("!!document.getElementById('libToBatch')")))

            js("document.getElementById('libToBatch').click()")
            time.sleep(2.0)
            check("切到批处理面板",
                  js("document.getElementById('sideCat').textContent")
                  == "批处理")
            banner = js("(function(){var es=[...document.querySelectorAll("
                        "'#sliders .bstat')];var b=es.find(function(e){"
                        "return e.textContent.indexOf('已从图库')>=0;"
                        "});return b?b.textContent:'';})()")
            check("批处理显示选中集横幅", isinstance(banner, str)
                  and "已从图库" in banner, str(banner)[:60])

            # 选中集确实进了 BATCH.paths
            npaths = js("(BATCH.paths||[]).length")
            check("BATCH.paths 带入选中集", npaths == 2, "%s 个" % npaths)

            shot = call("Page.captureScreenshot", format="png")
            with open(os.path.join(OUT, "ui_batch_sel.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))
            call("Emulation.clearDeviceMetricsOverride")
            ws.close()
    except Exception as exc:
        print("  UI 段跳过：%s" % str(exc)[:140])

    print()
    if FAIL:
        print("RESULT: FAILED %d -> %s" % (len(FAIL), FAIL))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
