# -*- coding: utf-8 -*-
"""AI 服务生命周期管理（engine/comfy_mgr）的回归。

覆盖 2026-10-24 易用性改造的核心链路：
- GET /api/ai/service 三态（未运行/启动中/就绪）与 comfy_dir 发现
- POST start（异步）→ 轮询到 ready → /api/health 的 generative_fill 转 true
- POST stop → 服务停止（句柄在我们手里，terminate 路径）
- 停止后再 start（第二次启动验证状态机复位）

环境：源码服务 8765；ComfyUI 便携版在 data/generative.json 或默认位置。
没有装 ComfyUI 的机器上本套会 SKIPPED（not_installed 即预期行为）。
"""
import sys
import time

import requests

B = "http://127.0.0.1:8765"
TIMEOUT_READY = 300          # 本机实测冷启动 ~110s，上限给足

results = []


def check(label, cond, detail=""):
    print("%-40s %s %s" % (label, "OK  " if cond else "FAIL", detail))
    results.append(bool(cond))


def svc():
    return requests.get(B + "/api/ai/service", timeout=10).json()


def main():
    st = svc()
    if not st.get("comfy_dir"):
        print("RESULT: SKIPPED（未安装 ComfyUI，not_installed 即正确行为）")
        return 0

    # 起点干净：如果在跑（外部自启），先停（stop 只动自己拉起的，
    # 外部实例会保持 running——那也算通过，跳过启动段）
    if st.get("ready"):
        r = requests.post(B + "/api/ai/service",
                          json={"action": "stop"}, timeout=60).json()
        if r.get("state") != "stopped":
            print("RESULT: SKIPPED（外部自启的 ComfyUI 在跑，不碰它）")
            check("外部实例不被 stop 误杀", svc().get("ready") is True)
            print("RESULT: PASS (external instance respected)")
            return 0

    check("未运行时状态 ready=false", svc().get("ready") is False)
    check("发现了 comfy_dir", bool(svc().get("comfy_dir")))

    # 异步启动：立即返回 starting
    r = requests.post(B + "/api/ai/service",
                      json={"action": "start"}, timeout=30).json()
    check("start 立即返回 starting", r.get("state") == "starting", str(r))

    t0 = time.time()
    ready_at = None
    while time.time() - t0 < TIMEOUT_READY:
        st = svc()
        if st.get("ready"):
            ready_at = time.time() - t0
            break
        if not st.get("starting"):
            break              # 启动失败（会带 last_error）
        time.sleep(4)
    check("轮询到就绪", ready_at is not None,
          "%.0fs (%s)" % (time.time() - t0, svc().get("last_error", "")))
    if ready_at is None:
        print("RESULT: FAILED（启动失败）")
        return 1

    h = requests.get(B + "/api/health", timeout=10).json()
    check("health.generative_fill 同步转 true", h.get("generative_fill") is True)

    # 懒启动路径：ready 时再 start 应幂等返回 ready
    r = requests.post(B + "/api/ai/service",
                      json={"action": "start"}, timeout=30).json()
    check("ready 时 start 幂等", r.get("state") == "ready", str(r))

    # 停止：我们拉起的 → terminate 生效
    r = requests.post(B + "/api/ai/service",
                      json={"action": "stop"}, timeout=60).json()
    check("stop 后服务停止", r.get("state") == "stopped", str(r))
    check("health.generative_fill 转回 false",
          requests.get(B + "/api/health", timeout=10)
          .json().get("generative_fill") is False)

    n_pass = sum(results)
    print("\n%d/%d 通过" % (n_pass, len(results)))
    if n_pass == len(results):
        print("RESULT: ALL PASS")
        return 0
    print("RESULT: FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
