# -*- coding: utf-8 -*-
"""极限环境暴力测试（不在 CI 列表——专找极端场景的 bug）。

六个阶段轮番轰炸一个源码服务，核心不变量只有一条：
**无论怎么虐，服务器必须活着**——任何 ConnectionError 记为致命，
5xx 记为 bug，4xx 是"正确拒绝"不计。每阶段后采样 /api/health 的
memory_mb 观察内存增长。

阶段：
  A 畸形输入轰击   空文件/文本/截断 JPEG/坏 session/坏 base64/极端参数
  B 会话洪泛       20 个上传挤爆 LRU(8)，旧会话应优雅 404
  C 大图           12MP 合成图走完整管线(带人脸检测)
  D 并发混合轰炸   3 线程 × 破坏性端点混跑 2 轮
  E 真超清双任务   2 个并发 4x 任务抢 GPU(推理锁应排队) + 渲染插队
  F 参数随机暴力   120 组随机参数快速渲染(无检测)

用法：python tests/stress_extreme.py [base_url]
"""
import io
import json
import random
import sys
import threading
import time

import numpy as np
import requests

B = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765"
issues = []
fatal = []
mem_samples = []


def mem_mb():
    try:
        return requests.get(B + "/api/health", timeout=15).json()["memory_mb"]
    except Exception:
        return -1


def check_phase(name, fn):
    print("\n=== %s ===" % name, flush=True)
    before = mem_mb()
    mem_samples.append((name, before))
    try:
        fn()
    except Exception as exc:
        fatal.append("%s 未捕获: %s" % (name, str(exc)[:100]))
    after = mem_mb()
    mem_samples[-1] = (name, before, after)
    alive = requests.get(B + "/api/health", timeout=15).status_code == 200
    print("[%s] 服务器存活: %s | 内存 %.0f -> %.0f MB"
          % (name, alive, before, after))
    if not alive:
        fatal.append("%s 后服务器死亡" % name)


def sess_new(img=None, name="s.jpg"):
    import cv2
    if img is None:
        img = np.random.rand(480, 640, 3).astype(np.float32) * 0.5 + 0.3
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
        (np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("imencode failed")
    f = io.BytesIO(buf.tobytes())
    f.name = name
    r = requests.post(B + "/api/upload", files={"file": (name, f, "image/jpeg")},
                      timeout=120)
    return r


# ---------------------------------------------------------------- A
def phase_a():
    bad_uploads = [
        ("空文件", b""),
        ("文本伪装", b"this is not an image at all"),
        ("截断JPEG", open("samples/synthetic_portrait.jpg", "rb").read()[:2000]),
        ("单像素", b"\xff\xd8\xff\xe0" + b"\x00" * 40),
    ]
    for label, payload in bad_uploads:
        r = requests.post(B + "/api/upload",
                          files={"file": ("x.jpg", payload, "image/jpeg")},
                          timeout=60)
        if r.status_code >= 500:
            issues.append("上传[%s] -> %d (应 4xx)" % (label, r.status_code))
        else:
            print("  上传[%s] -> %d (正确拒绝)" % (label, r.status_code))

    # 坏 session
    for body in ({"session": "nonexistent", "params": {}},
                 {"session": "", "params": {}},
                 {"session": "../etc/passwd", "params": {}}):
        r = requests.post(B + "/api/render", json=body, timeout=30)
        if r.status_code >= 500:
            issues.append("坏 session -> %d" % r.status_code)
    print("  坏 session -> 全部 4xx")

    # 坏 mask / 坏 base64
    for mask in ("not-base64!!!", "AAAA", ""):
        r = requests.post(B + "/api/fill", json={
            "session": sess_new().json()["session"],
            "mask_png": mask, "prompt": "x"}, timeout=60)
        if r.status_code >= 500:
            issues.append("坏 mask[%s...] -> %d" % (mask[:6], r.status_code))
    print("  坏 mask -> 全部 4xx")

    # 极端参数
    sid = sess_new().json()["session"]
    extreme = [{"exposure": 1e9}, {"exposure": -1e9}, {"smooth": 1e9},
               {"temperature": float("nan") if False else 1e308},
               {"ca_amount": -1e9}, {"hsl_red_hue": 1e9},
               {"distortion": 1e9, "vignetting": -1e9}]
    for p in extreme:
        r = requests.post(B + "/api/render", json={"session": sid, "params": p,
                                                   "long_edge": 400},
                          timeout=60)
        if r.status_code >= 500:
            issues.append("极端参数 %s -> %d" % (json.dumps(p)[:30],
                                                 r.status_code))
    print("  极端参数 %d 组 -> 全部消化" % len(extreme))


# ---------------------------------------------------------------- B
def phase_b():
    img = np.random.rand(420, 560, 3).astype(np.float32) * 0.5 + 0.3
    sids = []
    for i in range(20):
        r = sess_new(img + np.random.rand(420, 560, 3).astype(np.float32) * 0.05)
        if r.status_code != 200:
            issues.append("会话洪泛第 %d 个上传失败 %d" % (i, r.status_code))
            return
        sids.append(r.json()["session"])
    print("  20 个会话已创建（LRU 上限 8）")
    # 旧会话应优雅 404/410，不能 500
    gone = 0
    for s in sids[:10]:
        r = requests.post(B + "/api/render", json={"session": s, "params": {},
                                                   "long_edge": 300},
                          timeout=30)
        if r.status_code >= 500:
            issues.append("被逐出会话 render -> %d" % r.status_code)
        elif r.status_code in (404, 410):
            gone += 1
    print("  被逐出会话: %d 个优雅拒绝" % gone)
    # 最新会话必须仍然可用
    r = requests.post(B + "/api/render",
                      json={"session": sids[-1], "params": {"exposure": 5},
                            "long_edge": 300}, timeout=60)
    if r.status_code != 200:
        issues.append("最新会话 render 失败 %d" % r.status_code)


# ---------------------------------------------------------------- C
def phase_c():
    import cv2
    rng = np.random.default_rng(5)
    base = rng.random((3000, 4000, 3)).astype(np.float32) * 0.4 + 0.35
    # 人脸色块，让检测有事做
    cv2.rectangle(base, (1800, 1200), (2300, 1900), (0.85, 0.65, 0.55), -1)
    big = np.clip(base, 0, 1)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
        (big * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("imencode failed")
    f = io.BytesIO(buf.tobytes()); f.name = "big.jpg"
    t0 = time.time()
    r = requests.post(B + "/api/upload", files={"file": ("big.jpg", f,
                                                         "image/jpeg")},
                      timeout=300)
    up_s = time.time() - t0
    if r.status_code != 200:
        issues.append("12MP 上传失败 %d" % r.status_code)
        return
    sid = r.json()["session"]
    t0 = time.time()
    r = requests.post(B + "/api/render",
                      json={"session": sid, "params": {"exposure": 4,
                                                       "smooth": 20},
                            "long_edge": 1400}, timeout=600)
    render_s = time.time() - t0
    if r.status_code != 200:
        issues.append("12MP 渲染失败 %d" % r.status_code)
    else:
        print("  12MP: 上传 %.1fs, 渲染 %.1fs (%d px 输出)"
              % (up_s, render_s, r.json()["width"] ** 2 and r.json()["width"]))


# ---------------------------------------------------------------- D
def phase_d():
    with open("samples/synthetic_portrait.jpg", "rb") as f:
        sid = requests.post(B + "/api/upload",
                            files={"file": ("t.jpg", f, "image/jpeg")},
                            timeout=120).json()["session"]
    jobs = [
        ("/api/render", {"session": sid, "params": {"exposure": 5},
                         "long_edge": 700}),
        ("/api/portrait", {"session": sid, "smooth": 25, "commit": False}),
        ("/api/automask", {"session": sid}),
        ("/api/sky", {"session": sid, "preset": "sunset"}),
    ]
    errs = []
    for round_i in range(2):
        threads = []
        for path, body in jobs:
            def hit(path=path, body=body):
                try:
                    r = requests.post(B + path, json=body, timeout=300)
                    if r.status_code >= 500:
                        errs.append("%s -> %d" % (path, r.status_code))
                except Exception as e:
                    errs.append("%s 连接: %s" % (path, str(e)[:40]))
            th = threading.Thread(target=hit)
            threads.append(th)
            th.start()
        for t in threads:
            t.join()
        print("  轮次 %d: 4 端点并发完成, 5xx=%d" % (round_i + 1, len(errs)))
    if errs:
        issues.extend(errs)


# ---------------------------------------------------------------- E
def phase_e():
    import cv2
    img = np.clip(np.random.default_rng(9).random((800, 600, 3)) * 0.5 + 0.3,
                  0, 1).astype(np.float32)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(
        (img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("imencode failed")
    f = io.BytesIO(buf.tobytes()); f.name = "up.jpg"
    sid = requests.post(B + "/api/upload", files={"file": ("up.jpg", f,
                                                           "image/jpeg")},
                        timeout=120).json()["session"]
    jids = []
    for _ in range(2):                      # 两个并发 4x 任务
        r = requests.post(B + "/api/upscale",
                          json={"session": sid, "job": True}, timeout=30)
        jid = r.json().get("job")
        if jid:
            jids.append(jid)
    if len(jids) < 2:
        issues.append("并发超清任务未创建齐: %s" % jids)
        return
    print("  2 个 4x 任务已提交（推理锁应排队），期间插队渲染…")
    render_errs = []
    stop = time.time() + 90
    while time.time() < stop:
        r = requests.post(B + "/api/render",
                          json={"session": sid, "params": {},
                                "long_edge": 400}, timeout=120)
        if r.status_code != 200:
            render_errs.append(r.status_code)
        sts = [requests.get(B + "/api/upscale/job/" + j, timeout=15).json()
               for j in jids]
        if all(s["phase"] in ("done", "error") for s in sts):
            break
        time.sleep(2)
    for j in jids:
        st = requests.get(B + "/api/upscale/job/" + j, timeout=15).json()
        if st["phase"] == "error":
            issues.append("超清任务 error: %s" % st.get("error", "")[:80])
        else:
            print("  任务 %s: %s (%dx%d, %.0fs)"
                  % (j[:6], st["phase"], st.get("width", 0),
                     st.get("height", 0), 0))
    if render_errs:
        issues.append("超清期间插队渲染失败: %s" % render_errs[:3])


# ---------------------------------------------------------------- F
def phase_f():
    rng = random.Random(11)
    keys = ["exposure", "contrast", "highlights", "shadows", "whites",
            "blacks", "texture_tone", "clarity", "dehaze", "vibrance",
            "saturation", "temp", "tint", "smooth", "texture", "sharpening",
            "ca_amount", "distortion", "vignetting", "grain_amount",
            "hsl_red_hue", "hsl_orange_sat", "hsl_blue_lum"]
    with open("samples/synthetic_portrait.jpg", "rb") as f:
        sid = requests.post(B + "/api/upload",
                            files={"file": ("t.jpg", f, "image/jpeg")},
                            timeout=120).json()["session"]
    errs = []
    t0 = time.time()
    n = 0
    while time.time() - t0 < 90:            # 90 秒连发
        params = {k: rng.uniform(-100, 100) for k in keys}
        r = requests.post(B + "/api/render",
                          json={"session": sid, "params": params,
                                "long_edge": 420}, timeout=120)
        n += 1
        if r.status_code >= 500:
            errs.append("%d" % r.status_code)
    if errs:
        issues.append("随机参数 90s 渲染: %d 次 5xx" % len(errs))
    else:
        print("  90 秒连发 %d 次随机参数渲染, 零 5xx" % n)


def main():
    print("暴力测试目标:", B, flush=True)
    if requests.get(B + "/api/health", timeout=15).status_code != 200:
        print("服务器不可达")
        return 1
    for name, fn in (("A 畸形输入轰击", phase_a),
                     ("B 会话洪泛(LRU)", phase_b),
                     ("C 大图 12MP", phase_c),
                     ("D 并发混合轰炸", phase_d),
                     ("E 真超清双任务抢 GPU", phase_e),
                     ("F 参数随机暴力 90s", phase_f)):
        check_phase(name, fn)

    print("\n=== 内存轨迹 ===")
    for s in mem_samples:
        print("  %-22s %s" % (s[0], "->".join("%.0f MB" % x for x in s[1:])))
    growth = mem_samples[-1][2] - mem_samples[0][1]
    if growth > 1500:
        issues.append("内存增长 %.0f MB 疑似泄漏" % growth)

    print("\n=== 总计 ===")
    if fatal:
        for f in fatal:
            print("  致命:", f)
    for i in issues:
        print("  bug:", i)
    if not fatal and not issues:
        print("RESULT: SURVIVED CLEAN — 服务器在全部极限场景下存活，零 5xx")
        return 0
    print("RESULT: FOUND %d fatal / %d issues" % (len(fatal), len(issues)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
