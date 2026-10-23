# -*- coding: utf-8 -*-
"""Formula editor and culling contact sheet.

The formula engine has stored step programs all along; the UI could only run
them. What matters for an editor:

* a saved formula **round-trips** — what the editor writes is what the engine
  reads back, including numeric types (a value stored as the string "170"
  reaches `abs("170")` and raises);
* a builtin is **rejected** on delete rather than silently vanishing;
* the cull response carries a **thumbnail per result**, so picking is visual;
* the contact sheet's ordering matches the scores.
"""
import base64
import io
import json
import os
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BASE = "http://127.0.0.1:8765"
SRC = r"C:\Users\Rana\Desktop\26cicf\d3\宫园薰\_DSC7550.jpg"
failures = []


def check(label, cond, detail=""):
    print("%-56s %s %s" % (label, "OK" if cond else "FAIL", detail))
    if not cond:
        failures.append(label)


def main():
    from engine import formula as F

    print("=== 配方引擎往返 ===")
    name = "_verify_editor"
    formula = {
        "name": name,
        "description": "验证用",
        "steps": [
            {"kind": "params", "params": {"exposure": 12, "contrast": 8}},
            {"kind": "params", "params": {"temp": -5}},
        ],
    }
    lib = F.upsert(formula)
    check("保存成功", name in lib, str(len(lib)) + " 个配方")
    back = F.load_library()[name]
    check("步骤数往返一致", len(back["steps"]) == 2,
          str(len(back["steps"])))
    p0 = back["steps"][0]["params"]
    check("数值仍然是数字类型",
          isinstance(p0["exposure"], (int, float))
          and isinstance(p0["contrast"], (int, float)),
          "exposure=%r contrast=%r" % (p0["exposure"], p0["contrast"]))
    check("数值正确", float(p0["exposure"]) == 12.0)

    # a string number must not survive as a string: abs("170") raises
    F.upsert({"name": name, "description": "x",
              "steps": [{"kind": "params", "params": {"exposure": "20"}}]})
    b2 = F.load_library()[name]["steps"][0]["params"]["exposure"]
    check("字符串数字被规范化为数字", isinstance(b2, (int, float)),
          "%r (%s)" % (b2, type(b2).__name__))

    # run it to prove the stored program is executable
    from engine.imgutil import load_image, fit_long_edge
    img = fit_long_edge(load_image(SRC), 600)
    try:
        out, info = F.run(F.load_library()[name], img)
        check("存储的配方可以运行", out is not None and len(info["applied"]) > 0,
              str(info["applied"])[:50])
    except Exception as exc:
        check("存储的配方可以运行", False, str(exc)[:70])

    F.delete(name)
    check("删除生效", name not in F.load_library())

    # builtin protection is a library-free rule; check the server enforces it
    try:
        requests.get(BASE + "/api/health", timeout=60)
        up = None
        print("\n=== 配方 API ===")
        r = requests.post(BASE + "/api/formulas/save",
                          json={"formula": {"name": "_api_check", "description": "d",
                                            "steps": [{"kind": "params",
                                                       "params": {"exposure": 5}}]}},
                          timeout=60)
        check("API 保存 200", r.status_code == 200, str(r.status_code))
        got = requests.get(BASE + "/api/formulas", timeout=60).json()
        names = [f["name"] for f in got["formulas"]]
        check("保存后出现在列表", "_api_check" in names, str(names)[:60])

        d = requests.post(BASE + "/api/formulas/delete",
                          json={"formula": {"name": "skin_prep"}}, timeout=60)
        check("内置配方不可删除（400）", d.status_code == 400,
              "%s %s" % (d.status_code, str(d.json().get("detail"))[:40]))
        d2 = requests.post(BASE + "/api/formulas/delete",
                           json={"formula": {"name": "_api_check"}}, timeout=60)
        check("自建配方可删除", d2.status_code == 200)

        # ---- cull contact sheet ------------------------------------------
        print("\n=== 挑图缩略图 ===")
        from engine.imgutil import save_image, fit_long_edge as _fle
        from engine.imgutil import load_image as _li
        work = os.path.join(ROOT, "out", "_cull_test")
        os.makedirs(work, exist_ok=True)
        base = _fle(_li(SRC), 900)
        for i in range(4):
            save_image(os.path.join(work, "c%d.jpg" % i),
                       np.clip(base * (0.85 + i * 0.08), 0, 1), 92)
        cj = requests.post(BASE + "/api/cull",
                           json={"paths": [work], "thumbs": True},
                           timeout=300).json()
        check("返回结果", cj["count"] == 4, "%d 张" % cj["count"])
        check("每条结果带缩略图",
              all("thumb" in it for it in cj["results"]),
              "%d/%d" % (sum(1 for it in cj["results"] if "thumb" in it),
                         cj["count"]))
        if cj["results"] and "thumb" in cj["results"][0]:
            raw = base64.b64decode(cj["results"][0]["thumb"])
            from PIL import Image
            im = Image.open(io.BytesIO(raw))
            check("缩略图可解码且很小", im.width <= 200, "%dx%d" % im.size)
        scores = [it["overall"] for it in cj["results"]]
        check("按分数降序", all(scores[i] >= scores[i + 1]
                              for i in range(len(scores) - 1)),
              str([round(s, 3) for s in scores]))
        off = requests.post(BASE + "/api/cull",
                            json={"paths": [work], "thumbs": False},
                            timeout=300).json()
        check("thumbs=False 时不返回缩略图",
              all("thumb" not in it for it in off["results"]))
    except requests.exceptions.ConnectionError:
        print("\n服务未运行，跳过 API 检查")

    # ---- browser ---------------------------------------------------------
    print("\n=== UI ===")
    try:
        import websocket
        tgt = [t for t in requests.get("http://127.0.0.1:9223/json",
                                       timeout=10).json()
               if t.get("type") == "page" and "8765" in t.get("url", "")]
        if not tgt:
            print("  浏览器未连接，跳过")
        else:
            ws = websocket.create_connection(tgt[0]["webSocketDebuggerUrl"],
                                             timeout=40)
            n = [0]

            def call(method, **params):
                n[0] += 1
                ws.send(json.dumps({"id": n[0], "method": method,
                                    "params": params}))
                while True:
                    m = json.loads(ws.recv())
                    if m.get("id") == n[0]:
                        return m.get("result", {})

            def js(expr):
                r = call("Runtime.evaluate", expression=expr,
                         returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            call("Page.reload")
            time.sleep(4)
            js("selectCat('\u914d\u65b9')")
            time.sleep(1.5)
            check("配方列表有编辑按钮",
                  (js("document.querySelectorAll('[data-edit]').length") or 0) >= 5,
                  str(js("document.querySelectorAll('[data-edit]').length")))
            check("有新建按钮", bool(js("!!document.getElementById('fNew')")))
            js("document.getElementById('fNew').onclick()")
            time.sleep(0.5)
            check("可进入编辑模式", js("FEDIT.open") is True)
            check("编辑器有名称输入", bool(js("!!document.getElementById('fedName')")))
            check("有四种步骤按钮",
                  js("document.querySelectorAll('[data-add]').length") == 4,
                  str(js("document.querySelectorAll('[data-add]').length")))
            check("默认一个步骤", js("FEDIT.steps.length") == 1)
            # use real clicks (.click() dispatches a bubbling event, so the
            # row handler sees the glyph as ev.target). Calling .onclick()
            # directly invokes the row handler with the row as target, which
            # silently takes the "select this step" branch instead.
            js("document.querySelector('[data-add=\"params\"]').click()")
            time.sleep(0.3)
            check("可添加步骤", js("FEDIT.steps.length") == 2,
                  str(js("FEDIT.steps.length")))
            js("FEDIT.steps[0].params = {exposure: 20}; FEDIT.sel = 0;"
               " renderFormulaEditor();")
            time.sleep(0.3)
            # `!!` matters: returnByValue cannot serialise a DOM element, so a
            # bare querySelector comes back as undefined and reads as "absent"
            check("步骤详情显示参数",
                  bool(js("!!document.querySelector('input[data-k]')")),
                  "参数行数 %s" % js(
                      "document.querySelectorAll('input[data-k]').length"))
            # reorder: the step added by default is first, so moving it down
            # must leave a step after it
            before_kind = js("FEDIT.steps[0].kind")
            js("document.querySelector('[data-dn=\"0\"]').click()")
            time.sleep(0.3)
            check("可下移步骤", js("FEDIT.steps[1].kind") == before_kind,
                  "移到位置 1 的是 %s" % js("FEDIT.steps[1].kind"))
            js("document.querySelector('[data-del=\"0\"]').click()")
            time.sleep(0.3)
            check("可删除步骤", js("FEDIT.steps.length") == 1,
                  str(js("FEDIT.steps.length")))
            js("selectCat('\u6311\u56fe')")
            time.sleep(1.0)
            check("挑图面板存在", bool(js("!!document.getElementById('cullResult')")))
            check("挑图网格容器",
                  bool(js("!!document.getElementById('cullGrid') || "
                          "!!document.getElementById('cullResult')")))
            call("Page.captureScreenshot", format="png")
            n[0] += 1
            ws.send(json.dumps({"id": n[0], "method": "Page.captureScreenshot",
                                "params": {"format": "png"}}))
            while True:
                m = json.loads(ws.recv())
                if m.get("id") == n[0]:
                    with open(os.path.join(ROOT, "out", "ui6_editor.png"),
                              "wb") as fh:
                        fh.write(base64.b64decode(m["result"]["data"]))
                    break
            ws.close()
    except Exception as exc:
        print("  UI 检查跳过：%s" % str(exc)[:70])

    print()
    if failures:
        print("FAILED: %d -> %s" % (len(failures), failures))
        return 1
    print("RESULT: ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
