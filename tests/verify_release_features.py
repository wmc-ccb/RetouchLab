# -*- coding: utf-8 -*-
"""发布支持三件套的回归：版本接口、崩溃上报接口、关于面板、首次引导。

Round 3 (2026-10-24) 新增：
- /api/version    版本/构建/更新源 —— 关于面板与更新检查的本地基准
- /api/errors     out/error.log 的结构化视图 —— 复制诊断报告的数据源
- 关于面板 (?)    版本 + 更新检查 + 诊断 + 最近错误 + 一键复制
- 首次启动引导卡  只出现一次，任何交互收起，绝不挡自动化测试

环境：源码服务 8765；UI 段需要 Chrome CDP 9223 且有 8765 标签页。
"""
import json
import os
import sys
import time

import requests

BASE = "http://127.0.0.1:8765"
CDP_HTTP = "http://127.0.0.1:9223"

results = []


def check(label, cond, detail=""):
    print("%-46s %s %s" % (label, "OK  " if cond else "FAIL", detail))
    results.append((label, bool(cond)))


def api_checks():
    print("\n=== API：/api/version ===")
    v = requests.get(BASE + "/api/version", timeout=10).json()
    check("返回版本号", bool(v.get("version", "").count(".") >= 1),
          v.get("version", "?"))
    check("带构建标识", bool(v.get("build")), v.get("build", "?"))
    check("带更新源", str(v.get("update_url", "")).startswith("http"))
    check("带运行形态", isinstance(v.get("frozen"), bool))

    print("\n=== API：/api/errors ===")
    e = requests.get(BASE + "/api/errors", timeout=10).json()
    check("返回 errors 数组", isinstance(e.get("errors"), list))
    for item in e.get("errors", [])[:5]:
        ok = ("time" in item and "request" in item
              and "exception" in item)
        if not ok:
            check("条目结构 time/request/exception", False, str(item)[:80])
            break
    else:
        check("条目结构 time/request/exception", True,
              "%d 条" % len(e.get("errors", [])))
    lim = requests.get(BASE + "/api/errors?limit=1", timeout=10).json()
    check("limit 参数生效", len(lim.get("errors", [])) <= 1)


def ui_checks():
    print("\n=== UI：关于面板 / 首次引导 ===")
    import websocket
    tabs = requests.get(CDP_HTTP + "/json", timeout=10).json()
    ws_url = next(t["webSocketDebuggerUrl"] for t in tabs
                  if t.get("type") == "page" and "8765" in t.get("url", ""))
    ws = websocket.create_connection(ws_url, timeout=30)
    state = {"i": 0}

    def call(method, **params):
        state["i"] += 1
        ws.send(json.dumps({"id": state["i"], "method": method,
                            "params": params}))
        while True:
            msg = json.loads(ws.recv())
            if msg.get("id") == state["i"]:
                return msg.get("result", {})

    def js(expr):
        r = call("Runtime.evaluate", expression=expr,
                 returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")

    call("Page.reload")
    time.sleep(3.5)

    check("顶栏 ? 按钮存在", js(
        "!!document.getElementById('btnAbout')") is True)

    # 版本号一致性：关于面板显示 = /api/version
    ver_api = requests.get(BASE + "/api/version", timeout=10).json()
    js("ABOUT.show()")
    time.sleep(1.2)
    shown = js("(document.getElementById('aboutCard')||{})"
               ".innerHTML || ''")
    check("关于面板打开并含版本", ver_api["version"] in (shown or ""),
          ver_api["version"])
    check("关于面板含快速上手", "快速上手" in (shown or ""))
    check("关于面板含复制按钮", "复制诊断报告" in (shown or ""))

    # 更新比较逻辑：纯函数，四种序
    cmp_ok = js("""(function(){
      const n = ABOUT.newer;
      return n('1.2.0','1.1.0') === true && n('1.1.0','1.1.0') === false
          && n('1.1.9','1.2.0') === false && n('2.0','1.9.9') === true
          && n('v1.2.0','1.1.0') === false ? 'ok' : 'bad';
    })()""")
    check("版本序比较 newer()", cmp_ok == "ok", str(cmp_ok))

    # Esc 关闭
    js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape'}))")
    time.sleep(0.3)
    check("Esc 关闭关于面板", js(
        "!document.getElementById('aboutOv')") is True)

    # 首次引导：flag 置位时不出现；清掉后出现且可收起
    # （先清掉页面载入时可能留下的卡，保证每步从已知状态出发）
    js("var w0=document.getElementById('welcome'); if(w0) w0.remove();"
       "localStorage.setItem('rl_onboarded','1'); showWelcomeOnce()")
    check("已引导过不再出现", js(
        "!document.getElementById('welcome')") is True)
    js("localStorage.removeItem('rl_onboarded'); showWelcomeOnce()")
    time.sleep(0.2)
    check("未引导时出现引导卡", js(
        "!!document.getElementById('welcome')") is True)
    # 程序化点「开始使用」= 收起 + 置 flag（测试不派发 pointerdown）
    js("document.querySelector('#welcome .go').click()")
    check("「开始使用」收起并记 flag", js(
        "!document.getElementById('welcome') && "
        "localStorage.getItem('rl_onboarded') === '1'") is True)

    # 页面重载后一切照旧（flag 持久）
    call("Page.reload")
    time.sleep(3.0)
    check("重载后不再出现引导卡", js(
        "!document.getElementById('welcome')") is True)
    check("页面功能正常(spec 就绪)", (js(
        "(typeof S!=='undefined'&&S.spec)?S.spec.length:0") or 0) > 50)

    ws.close()


def main():
    api_checks()
    try:
        ui_checks()
    except Exception as exc:               # 浏览器不在时报告为失败而不是崩
        check("UI 段", False, "%s: %s" % (type(exc).__name__, exc)[:80])

    n_pass = sum(1 for _, r in results if r)
    print("\n%d/%d 通过" % (n_pass, len(results)))
    if n_pass == len(results):
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILED %d -> %s" % (
        len(results) - n_pass,
        [l for l, r in results if not r][:6]))
    return 1


if __name__ == "__main__":
    sys.exit(main())
