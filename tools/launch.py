# -*- coding: utf-8 -*-
"""RetouchLab 启动器。

双击 `RetouchLab.bat`（或打包后的 exe）会走到这里，负责：

  1. 单实例检查 —— 端口已占用就直接开浏览器，不重复起进程
  2. 挑一个可用端口（8765 被别的程序占了就往后找）
  3. 起 uvicorn，等服务真的能响应再打开浏览器
  4. 日志写到 out/launcher.log，出错时把原因显示给用户
  5. Ctrl+C / 关窗口后干净退出

设计取舍：
  - 不隐藏控制台。用户想关掉服务时需要看得到窗口；打包成 exe 后
    用 --noconsole 也可以，但那样出错时用户只能看到"闪退"。
  - 浏览器打开失败不算致命（可能是没有默认浏览器），继续跑服务，
    只是提示用户手敲地址。
"""
import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# **必须在任何 torch import 之前注册 CUDA DLL 目录。**
# 打包版里 torch/lib 不在默认 DLL 搜索路径上，torch 会因为找不到
# cublasLt64_12.dll 而静默回退 CPU（实测超清从 9s 变 109s）。
try:
    from engine.paths import register_cuda_dlls
    register_cuda_dlls()
except Exception:
    pass

LOG_DIR = os.path.join(ROOT, "out")
LOG_PATH = os.path.join(LOG_DIR, "launcher.log")
DEFAULT_PORT = 8765


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


def port_in_use(port, host="127.0.0.1", timeout=0.4):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def is_our_service(port):
    """端口上跑的是不是 RetouchLab（而不是别的程序）。

    判据要**足够严**：只看 "ok" 字段会把任何返回 {"ok":true} 的服务
    误认成自己，进而"复用"到一个根本不是 RetouchLab 的端口上——用户
    双击后既没起服务、也没报错，只看到浏览器打开一个陌生页面。
    这里同时要求 3 个 RetouchLab 特有字段。
    """
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(
                "http://127.0.0.1:%d/api/health" % port,
                timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (bool(data.get("ok"))
                and "max_sessions" in data
                and "face_model" in data)
    except Exception:
        return False


def pick_port(preferred):
    """优先用首选端口；被别的程序占了就往后找。

    返回 (port, reusable)：reusable 为 True 表示该端口上已经是
    RetouchLab 自己，调用方应直接开浏览器而不是再起一个实例。
    """
    if not port_in_use(preferred):
        return preferred, False
    if is_our_service(preferred):
        return preferred, True
    for p in range(preferred + 1, preferred + 20):
        if not port_in_use(p):
            return p, False
    # 全被占：让系统随机分配一个空闲端口
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1], False
    finally:
        s.close()


def wait_until_ready(port, timeout=90.0):
    """等服务能应答 /api/health。返回 True 表示就绪。"""
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/api/health" % port,
                    timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def open_browser_later(url, port, delay_hint=0.0):
    def work():
        if delay_hint:
            time.sleep(delay_hint)
        if wait_until_ready(port):
            log("服务就绪，打开浏览器: %s" % url)
            try:
                webbrowser.open(url)
            except Exception as e:
                log("打开浏览器失败（%s），请手动访问 %s" % (e, url))
        else:
            log("等待服务就绪超时；请手动访问 %s" % url)
    threading.Thread(target=work, daemon=True).start()


def banner(port, reused):
    host_line = "http://127.0.0.1:%d" % port
    try:
        lan_ip = _lan_ip()
    except Exception:
        lan_ip = None
    print()
    print("  " + "=" * 56)
    print("   RetouchLab   人像精修引擎")
    print("  " + "-" * 56)
    print("   本机:  %s" % host_line)
    if lan_ip:
        print("   局域网: http://%s:%d" % (lan_ip, port))
    if reused:
        print("   （检测到服务已在运行，直接复用）")
    else:
        print("   关闭这个窗口即停止服务")
    print("  " + "=" * 56)
    print()


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 80))
        return s.getsockname()[0]
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser(description="RetouchLab launcher")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--lan", action="store_true",
                    help="监听 0.0.0.0，允许局域网访问")
    args = ap.parse_args()

    host = "0.0.0.0" if args.lan else args.host
    port, reusable = pick_port(args.port)
    url = "http://127.0.0.1:%d/" % port

    log("RetouchLab 启动中（Python %s）" % sys.version.split()[0])
    log("目录: %s" % ROOT)

    # ---- 已经有实例在跑：直接开浏览器，不重复起 ----
    if reusable:
        banner(port, reused=True)
        log("复用已运行的服务 %s" % url)
        if not args.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        return 0

    if port != args.port:
        log("端口 %d 被占用，改用 %d" % (args.port, port))

    # ---- 起服务 ----
    try:
        import uvicorn
    except Exception as e:
        log("依赖缺失：%s" % e)
        print("\n启动失败：没找到 uvicorn。请先运行")
        print("  python -m pip install -r requirements.txt\n")
        return 2

    if not args.no_browser:
        open_browser_later(url, port)

    banner(port, reused=False)

    try:
        import server.app as APP
        config = uvicorn.Config(APP.app, host=host, port=port,
                                log_level="warning", access_log=False)
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        log("收到中断，退出")
    except Exception as e:
        log("服务崩溃：%r" % (e,))
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
