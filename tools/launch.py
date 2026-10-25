# -*- coding: utf-8 -*-
"""ZhaJinWenQianCake 启动器（桌面应用模式）。

双击 exe（或 `python tools/launch.py`）的完整流程：

  1. 单实例检查 —— 已有实例在跑就直接开一个新窗口贴上去，不重复起服务
  2. 挑一个可用端口（8765 被占就往后找）
  3. 起服务（后台线程）
  4. **弹原生应用窗口**（pywebview / WebView2，Win10/11 自带运行时），
     先显示加载页，服务就绪后自动载入界面
  5. 关窗口 = 退应用：停服务、顺手停掉自己拉起的 ComfyUI、进程干净退出

兼容与兜底：
  - 没有 WebView2 的老系统（或 pywebview 加载失败）自动退回系统浏览器，
    行为与旧版一致，绝不让用户面对"闪退"。
  - --browser 强制浏览器模式；--lan 监听局域网（手机/平板仍用浏览器访问）。
  - 无控制台（打包 windowed）时，致命错误写 out/launcher.log 并弹
    MessageBox 告知，不搞无声闪退。
"""
import argparse
import os
import socket
import sys
import threading
import time

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
WINDOW_TITLE = "炸金文谦蛋糕 — 人像精修"

LOADING_HTML = """<!doctype html><html><head><meta charset="utf-8">
<style>
body{margin:0;background:#0a0a0c;color:#ddd;display:flex;flex-direction:column;
align-items:center;justify-content:center;height:100vh;font:15px/1.8 system-ui,
"Microsoft YaHei"}
.logo{font-size:34px;font-weight:600;color:#e8b341;letter-spacing:1px}
.logo span{color:#e8e9ec}
.tip{color:#8a8c93;font-size:13px;margin-top:10px}
.spin{width:34px;height:34px;border:3px solid #26272c;border-top-color:#e8b341;
border-radius:50%;margin-bottom:22px;animation:r 1s linear infinite}
@keyframes r{to{transform:rotate(360deg)}}
</style></head><body>
<div class="spin"></div>
<div class="logo">炸金文谦蛋糕<span>Lab</span></div>
<div class="tip">正在启动引擎与模型，首次约 10~30 秒…</div>
</body></html>"""


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


def fatal(msg):
    """无控制台场景的致命错误：日志 + 消息框，拒绝无声闪退。"""
    log("致命错误：%s" % msg)
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "ZhaJinWenQianCake 启动失败：\n\n%s\n\n详情见 out\\launcher.log"
                % msg, "ZhaJinWenQianCake", 0x10)
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
    """端口上跑的是不是 ZhaJinWenQianCake（而不是别的程序）。

    判据要**足够严**：只看 "ok" 字段会把任何返回 {"ok":true} 的服务
    误认成自己。同时要求 3 个 ZhaJinWenQianCake 特有字段。
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
    ZhaJinWenQianCake 自己，调用方应直接开窗口而不是再起一个实例。
    """
    if not port_in_use(preferred):
        return preferred, False
    if is_our_service(preferred):
        return preferred, True
    for p in range(preferred + 1, preferred + 20):
        if not port_in_use(p):
            return p, False
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1], False
    finally:
        s.close()


def wait_until_ready(port, timeout=90.0):
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


def open_window(url, port, own_service, server=None):
    """弹桌面窗口；返回 True 表示窗口正常打开并已关闭（正常退出路径）。

    own_service=True：本进程起了服务，关窗口时要负责停服务 + 清理
    ComfyUI 子进程。False：贴到已有实例，关窗口直接走人。
    """
    try:
        import webview
    except Exception as e:
        log("pywebview 不可用（%s），退回浏览器模式" % e)
        return False

    win = webview.create_window(
        WINDOW_TITLE, html=LOADING_HTML,
        width=1440, height=900, min_size=(1100, 700))

    def _navigate():
        # 加载页先撑住画面，服务就绪后无缝切过去
        if wait_until_ready(port, timeout=120):
            log("服务就绪，载入界面: %s" % url)
            try:
                win.load_url(url)
            except Exception:
                pass
        else:
            try:
                win.load_html("<h3 style='font-family:system-ui'>"
                              "启动超时，详见 out\\launcher.log</h3>")
            except Exception:
                pass

    threading.Thread(target=_navigate, daemon=True).start()
    try:
        # webview.start 必须在主线程（Win32 消息循环），返回 = 窗口已关
        webview.start(gui="edgechromium")
    except Exception as e:
        log("窗口创建失败（%s），退回浏览器模式" % e)
        return False

    # ---- 窗口关闭：干净收尾 ----
    log("窗口已关闭，正在退出")
    if own_service:
        try:
            from engine import comfy_mgr
            comfy_mgr.stop()          # 只停我们拉起的 ComfyUI（8GB）
        except Exception:
            pass
        if server is not None:
            server.should_exit = True
            time.sleep(0.5)
    # 模型预热线程等仍持有 torch 上下文，正常退出可能挂住；
    # 收尾已做完，硬退是打包应用的标准做法。
    os._exit(0)
    return True


def open_browser(url, port, delay_hint=0.0):
    import webbrowser

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


def main():
    ap = argparse.ArgumentParser(description="ZhaJinWenQianCake launcher")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--browser", action="store_true",
                    help="用系统浏览器而不是应用窗口")
    ap.add_argument("--lan", action="store_true",
                    help="监听 0.0.0.0，允许局域网访问")
    args = ap.parse_args()

    host = "0.0.0.0" if args.lan else args.host
    port, reusable = pick_port(args.port)
    url = "http://127.0.0.1:%d/" % port

    log("ZhaJinWenQianCake 启动中（Python %s）" % sys.version.split()[0])
    log("目录: %s" % ROOT)

    # ---- 已经有实例在跑：贴一个新窗口/开浏览器，不重复起服务 ----
    if reusable:
        log("复用已运行的服务 %s" % url)
        if not args.browser and open_window(url, port, own_service=False):
            return 0
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass
        return 0

    if port != args.port:
        log("端口 %d 被占用，改用 %d" % (args.port, port))

    # ---- 起服务（后台线程；窗口/浏览器在前面走） ----
    try:
        import uvicorn
    except Exception as e:
        fatal("依赖缺失 uvicorn（%s）" % e)
        return 2

    server = None

    def _serve():
        nonlocal server
        try:
            import server.app as APP
            config = uvicorn.Config(APP.app, host=host, port=port,
                                    log_level="warning", access_log=False)
            server = uvicorn.Server(config)
            # 非主线程：uvicorn 会跳过信号捕获（源码已处理），Ctrl+C
            # 语义由窗口关闭代替
            server.run()
        except Exception as e:
            log("服务崩溃：%r" % (e,))
            import traceback
            log(traceback.format_exc()[-1500:])
            fatal("服务线程崩溃：%s" % str(e)[:120])
            os._exit(1)

    threading.Thread(target=_serve, daemon=True).start()

    if not args.browser and open_window(url, port, own_service=True,
                                        server=server):
        return 0
    # 浏览器兜底路径（无 WebView2 / --browser）
    open_browser(url, port)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("收到中断，退出")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        fatal(str(e)[:200])
        sys.exit(1)
