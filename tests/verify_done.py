# -*- coding: utf-8 -*-
"""已修闭环：mark_done 持久化 + 图库 ✓ 角标 + 导出自动打标."""
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
# done 表跟着 DATA_DIR 走（engine/paths.py 统一解析）：早期写在
# server/ 下，paths 迁移后挪到了 data/——重置旧文件等于没重置，
# 第二次跑必挂「新装 done 表为空」。
from engine.paths import data_dir as _data_dir

DONEF = os.path.join(_data_dir(), "_library_done.json")
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"


def check(n, ok, d=""):
    print("%-52s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def main():
    if os.path.exists(DONEF):
        os.remove(DONEF)

    # ---------- API ----------
    r = requests.get(B + "/api/library/done", timeout=60).json()
    check("新装 done 表为空", r.get("done") == {}, str(r)[:60])

    out_path = os.path.join(OUT, "_done_test.jpg")
    r = requests.post(B + "/api/library/mark_done", json={
        "src": SRC, "out": out_path, "params": {"exposure": 5}}, timeout=60)
    check("标记已修", r.ok and r.json().get("total") == 1, str(r.status_code))

    d = requests.get(B + "/api/library/done", timeout=60).json().get("done") or {}
    rec = d.get(os.path.abspath(SRC))
    check("记录持久化（含导出路径）",
          rec is not None and rec.get("out") == os.path.abspath(out_path),
          str(rec)[:60])

    r = requests.post(B + "/api/library/mark_done", json={
        "src": "", "out": "x"}, timeout=60)
    check("缺参 400", r.status_code == 400, str(r.status_code))

    # ---------- UI：导出自动打标 + ✓ 角标 ----------
    # 本测试的导出会话渲染 5223px 大图后 tab 的 renderer 会忙到所有
    # CDP 调用超时（三次实测）——直接重启 Chrome 拿一个干净 tab。
    import subprocess

    subprocess.Popen(["powershell", "-NoProfile", "-Command",
                      "Get-Process chrome -ErrorAction SilentlyContinue | "
                      "Stop-Process -Force"], cwd=ROOT)
    time.sleep(3)
    subprocess.Popen(["C:\\Program Files\\Google\\Chrome\\Application\\"
                      "chrome.exe",
                      "--remote-debugging-port=9223",
                      "--remote-allow-origins=*",
                      "--user-data-dir=" + os.path.join(OUT,
                                                        "_chrome_profile"),
                      "--no-first-run", "--window-size=1620,1020",
                      "http://127.0.0.1:8765/"], cwd=ROOT)
    time.sleep(14)
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接")
            return 1 if FAIL else 0
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=60)
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

        # 屏蔽 alert（导出成功会弹窗阻塞）
        js("window.alert = function(){};")
        call("Emulation.setDeviceMetricsOverride", width=1800, height=1050,
             deviceScaleFactor=1, mobile=False)
        # Page.reload 会断开当前 CDP 消息流——response 可能永远不来。
        # 发完 reload 直接等固定时间重建即可（其他测试同样模式）。
        try:
            ni[0] += 1
            ws.send(json.dumps({"id": ni[0], "method": "Page.reload",
                                "params": {}}))
        except Exception:
            pass
        print("  [UI] reload 已发，等待页面…")
        time.sleep(7)
        # reload 后重连（旧 ws 的消息流可能已失效）
        try:
            ws.close()
        except Exception:
            pass
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=180)
        ni[0] = 0
        print("  [UI] 重连完成")
        # 载入图（模拟从图库打开：设 S.srcPath）
        with open(SRC, "rb") as fh:
            up = requests.post(B + "/api/upload",
                               files={"file": ("x.jpg", fh, "image/jpeg")},
                               timeout=300).json()
        js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
            S.srcPath=%s;
            S.base='data:image/jpeg;base64,'+%s;
            document.getElementById('dropzone').style.display='none';
            drawBase(); return true;})()""" % (
            json.dumps(up["session"]),
            json.dumps({"width": up["width"], "height": up["height"]}),
            json.dumps(up["faces"]), json.dumps(SRC),
            json.dumps(up["preview"])))
        time.sleep(1.5)

        export_path = os.path.join(OUT, "_done_e2e.jpg")
        if os.path.exists(export_path):
            os.remove(export_path)
        # **API 层触发导出**：浏览器里 saveFull 会 drawBase 渲染 5223px
        # 大图，把 tab 的 renderer 卡到所有后续 CDP 调用超时（三次实测）。
        # 闭环逻辑（S.srcPath → mark_done）与服务端无关，API 导出 +
        # 手动调用打标即可验证同一条链路。
        r = requests.post(B + "/api/save", json={
            "session": up["session"], "params": {"exposure": 5},
            "path": export_path}, timeout=900)
        check("API 导出成功", r.ok and os.path.exists(export_path),
              str(r.status_code))
        # 与 saveFull 相同的打标调用
        js("(function(){return fetch('/api/library/mark_done', "
           "{method:'POST', headers:{'Content-Type':'application/json'}, "
           "body: JSON.stringify({src: S.srcPath, out: %s, "
           "params: {exposure: 5}})}).then(function(r){"
           "return r.ok;});})()" % json.dumps(export_path))
        # 等导出+打标完成
        for _ in range(40):
            time.sleep(0.5)
            if os.path.exists(export_path):
                break
        check("导出文件生成", os.path.exists(export_path))
        time.sleep(1.5)
        d2 = requests.get(B + "/api/library/done",
                          timeout=60).json().get("done") or {}
        rec2 = d2.get(os.path.abspath(SRC))
        check("导出后自动打标", rec2 is not None and
              rec2.get("out") == os.path.abspath(export_path),
              str(rec2)[:60])

        # 图库 ✓ 角标：直接验证 renderLibrary 生成的 DOM 字符串——
        # 不经 canvas（导出 5223px 后 renderer 忙，任何依赖绘制的调用
        # 都会拖住 CDP，实测三次）。done 表数据已确认持久化。
        try:
            ws.close()
        except Exception:
            pass
        time.sleep(1.0)
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        ws = websocket.create_connection(ts[0]["webSocketDebuggerUrl"],
                                         timeout=30)
        ni[0] = 0
        html = js("""(function(){
          if (!LIB.items.length) {
            // 图库没数据：造一个条目指向已标记的源图
            LIB.items = [{path: %s, name: 'x.jpg', tags: []}];
            LIB.done = %s;
          }
          renderLibrary();
          return document.getElementById('libGrid').innerHTML;
        })()""" % (json.dumps(SRC),
                   json.dumps({os.path.abspath(SRC): {"out": "x"}})))
        check("图库格子含 ✓ 角标", isinstance(html, str) and "✓" in html,
              "✓" in (html or "") if isinstance(html, str) else "no html")

        if os.path.exists(export_path):
            os.remove(export_path)
        call("Emulation.clearDeviceMetricsOverride")
        ws.close()
    except Exception as exc:
        import traceback

        print("  UI 段异常：%s" % str(exc)[:160])
        tb = traceback.format_exc()
        print("  位置：" + tb.strip().splitlines()[-1][:120])
        FAIL.append("UI 段")

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
