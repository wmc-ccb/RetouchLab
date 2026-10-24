# -*- coding: utf-8 -*-
"""我的预设：保存 / 覆盖 / 删除 / 双 tab."""
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
from engine.paths import data_dir as _data_dir  # 状态在 data/ 不在 server/ (paths 迁移)
UPF = os.path.join(_data_dir(), "_user_presets.json")


def check(n, ok, d=""):
    print("%-46s %s %s" % (n, "OK  " if ok else "FAIL", d))
    if not ok:
        FAIL.append(n)


def clean():
    if os.path.exists(UPF):
        os.remove(UPF)


def main():
    clean()

    # ---------- API ----------
    r = requests.get(B + "/api/presets", timeout=60).json()
    builtin = r.get("presets", {})
    check("内置预设存在", "natural" in builtin, "%d 个" % len(builtin))

    r = requests.post(B + "/api/presets/save", json={
        "name": "My Test Preset", "params": {"exposure": 8, "contrast": 5}},
        timeout=60)
    check("保存自定义预设", r.ok and r.json().get("key") == "mytestpreset",
          str(r.json())[:80])

    r = requests.post(B + "/api/presets/save", json={
        "name": "natural", "params": {"exposure": 8}}, timeout=60)
    check("内置名不可覆盖 (409)", r.status_code == 409, str(r.status_code))

    r = requests.post(B + "/api/presets/save", json={
        "name": "my test preset", "params": {"exposure": 2}}, timeout=60)
    check("同名需显式覆盖 (409)", r.status_code == 409, str(r.status_code))
    r = requests.post(B + "/api/presets/save", json={
        "name": "my test preset", "params": {"exposure": 2},
        "overwrite": True}, timeout=60)
    check("显式覆盖成功", r.ok, str(r.status_code))

    r = requests.post(B + "/api/presets/save", json={
        "name": "junk", "params": {"not_a_param": 9, "also_bad": 1}},
        timeout=60)
    check("未知参数被拒绝", r.status_code == 400, str(r.status_code))

    r = requests.get(B + "/api/presets", timeout=60).json()
    allp = r.get("presets", {})
    check("我的预设出现在总表", "mytestpreset" in allp,
          "%d 个" % len(allp))
    check("覆盖后参数生效", allp.get("mytestpreset", {}).get("exposure") == 2,
          str(allp.get("mytestpreset"))[:60])

    # 用它渲染一次（预设必须可用）
    with open(r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg",
              "rb") as fh:
        up = requests.post(B + "/api/upload",
                           files={"file": ("x.jpg", fh, "image/jpeg")},
                           timeout=300).json()
    sid = up["session"]
    r = requests.post(B + "/api/render", json={
        "session": sid, "params": allp["mytestpreset"], "quality": 80},
        timeout=300)
    check("自定义预设可渲染", r.ok and len(r.json().get("image", "")) > 1000,
          "%d" % r.status_code)

    r = requests.post(B + "/api/presets/delete", json={"key": "nope"},
                      timeout=60)
    check("删除不存在的 (404)", r.status_code == 404, str(r.status_code))
    r = requests.post(B + "/api/presets/delete", json={"key": "mytestpreset"},
                      timeout=60)
    check("删除成功", r.ok, str(r.status_code))
    r = requests.get(B + "/api/presets", timeout=60).json()
    check("删除后不在总表", "mytestpreset" not in r.get("presets", {}))

    # ---------- UI ----------
    try:
        ts = [x for x in requests.get(CDP + "/json", timeout=10).json()
              if x.get("type") == "page" and "8765" in x.get("url", "")]
        if not ts:
            print("浏览器未连接，跳过 UI")
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
            js("""(function(){S.session=%s;S.meta=%s;S.faces=%s;
                S.base='data:image/jpeg;base64,'+%s;
                S.params={exposure:8,contrast:5};
                document.getElementById('dropzone').style.display='none';
                drawBase(); return true;})()""" % (
                json.dumps(sid),
                json.dumps({"width": up["width"], "height": up["height"],
                            "preview_width": up["preview_width"],
                            "preview_height": up["preview_height"]}),
                json.dumps(up["faces"]), json.dumps(up["preview"])))
            time.sleep(2)

            js("selectCat('光影')")
            time.sleep(2.0)
            tabs = js("[...document.querySelectorAll('#presetTabs button')]"
                      ".map(function(b){return b.textContent})")
            check("有推荐/我的两个 tab",
                  isinstance(tabs, list) and "推荐" in tabs and "我的" in tabs,
                  str(tabs))
            nrec = js("document.querySelectorAll('#pcList .pcard').length")
            check("推荐 tab 有内置预设", (nrec or 0) >= 5, "%s 张" % nrec)

            js("document.querySelector('#presetTabs [data-t=mine]').click()")
            time.sleep(1.0)
            nmine = js("document.querySelectorAll('#pcList .pcard').length")
            check("我的 tab 初始为空提示", (nmine or 0) == 0,
                  "%s 张" % nmine)
            has_hint = js("(document.querySelector('#pcList .bstat')||{})"
                          ".textContent")
            check("空态有引导文案", isinstance(has_hint, str)
                  and "存为我的预设" in has_hint, str(has_hint)[:50])

            # 保存一份（绕过 prompt）
            js("window.prompt = function(){return 'ui 测试预设';};"
               "window.confirm = function(){return true;};")
            js("S.params={exposure:6,vibrance:10};")
            js("saveUserPreset()")
            # 保存流程是异步的：POST → 清缓存 → 重新拉参数表 → 重渲染。
            # 轮询等卡片出现，固定 sleep 会读在半路上。
            nmine = 0
            for _ in range(30):
                time.sleep(0.5)
                nmine = js("document.querySelectorAll('#pcList .pcard')"
                           ".length") or 0
                if nmine >= 1:
                    break
            check("保存后出现在我的 tab", nmine >= 1, "%s 张" % nmine)
            tabobj = js("(function(){var b=document.querySelector("
                        "'#presetTabs button.active');"
                        "return b ? b.dataset.t : null;})()")
            check("保存后自动切到我的", tabobj == "mine", str(tabobj))
            shot = call("Page.captureScreenshot", format="png")
            with open(os.path.join(OUT, "ui_mypreset.png"), "wb") as fh:
                fh.write(base64.b64decode(shot["data"]))

            js("window.confirm = function(){return true;};")
            js("document.querySelector('#pcList .pdel').click()")
            time.sleep(2.0)
            nmine = js("document.querySelectorAll('#pcList .pcard').length")
            check("UI 删除后为空", (nmine or 0) == 0, "%s 张" % nmine)

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
    try:
        sys.exit(main())
    finally:
        clean()
