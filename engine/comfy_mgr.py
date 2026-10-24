# -*- coding: utf-8 -*-
"""ComfyUI 服务的生命周期管理：按需启动、就绪探测、空闲自动停止。

为什么存在：生成式三件套（AI 布景 / 创成式填充 / 道具修复）跑在外挂
ComfyUI 上，此前用户要自己开终端跑脚本启动它——打包发行版的用户机器
上甚至没有 Python，这条路是死的。本模块把启动/停止搬进服务进程：
前端按钮 → /api/ai/service → 这里 subprocess 拉起 ComfyUI 自带的
python_embeded，用户全程零命令。

约定：
- 「就绪」的判据与 engine/genfill.available() 一致（HTTP 应答且注册了
  inpaint checkpoint），这里直接复用它，保证面板状态与功能可用性永远
  说同一句话。
- 只主动停「我们自己拉起」的进程（句柄在手的 terminate），外部自己
  启动的 ComfyUI 绝不碰；taskkill 兜底仅用于清理我们拉起但句柄丢失
  的半死进程。
- 空闲 15 分钟自动停止：ComfyUI 常驻吃 ~8GB 内存，修图主流程不该
  为偶尔用一次的 AI 功能一直买单。任何一次成功使用都会刷新时间戳。
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from typing import Optional

PORT = 8388
IDLE_STOP_SECONDS = 15 * 60
START_TIMEOUT = 300          # 装满自定义节点的便携版冷启动实测可到 2 分钟+

_STATE = {
    "proc": None,            # 我们拉起的 Popen（可能是 None 而服务还活着）
    "comfy_dir": None,
    "starting": False,
    "started_at": 0.0,
    "last_used": 0.0,
    "last_error": "",
}
_LOCK = threading.Lock()


def _root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gen_config_path() -> str:
    from .paths import data_dir
    return os.path.join(data_dir(), "generative.json")


def find_comfy_dir() -> Optional[str]:
    """定位 ComfyUI 便携版根目录（含 python_embeded 与 ComfyUI/main.py）。

    优先级与 tools/install_generative.py 保持一致：环境变量 >
    data/generative.json（install 写入）> 常见路径。
    """
    cands = []
    if os.environ.get("RETOUCHLAB_COMFYUI_DIR"):
        cands.append(os.environ["RETOUCHLAB_COMFYUI_DIR"])
    try:
        with open(_gen_config_path(), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("comfy_dir"):
            cands.append(d["comfy_dir"])
    except Exception:
        pass
    home = os.path.expanduser("~")
    cands += [os.path.join(home, "Desktop", "ComfyUI"),
              r"C:\ComfyUI", os.path.join(home, "ComfyUI")]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "python_embeded", "python.exe")) \
                and os.path.isfile(os.path.join(c, "ComfyUI", "main.py")):
            return c
    return None


def is_ready() -> bool:
    """ComfyUI 在跑且模型注册（与 genfill 同判据，不缓存）。"""
    from . import genfill
    return genfill.available(cache=False)


def start(background: bool = True, timeout: float = START_TIMEOUT) -> dict:
    """启动 ComfyUI。background=True 立即返回由调用方轮询；False 同步等就绪。

    返回 {state}: ready / starting / not_installed / error。
    """
    with _LOCK:
        if is_ready():
            _STATE["last_used"] = time.time()
            return {"state": "ready"}
        if _STATE["starting"]:
            return {"state": "starting"}
        d = find_comfy_dir()
        if not d:
            return {"state": "not_installed"}
        # 模型齐不齐先查（缺模型拉起来也白搭，还占 8GB 内存）
        ck = os.path.join(d, "ComfyUI", "models", "checkpoints",
                          "sd15_inpaint.ckpt")
        cn = os.path.join(d, "ComfyUI", "models", "controlnet",
                          "control_v11p_sd15_inpaint_fp16.safetensors")
        missing = [n for n, p in (("主模型", ck), ("ControlNet", cn))
                   if not os.path.isfile(p)]
        if missing:
            return {"state": "not_installed",
                    "detail": "缺 " + "、".join(missing) +
                              "（安装方法见 GENERATIVE.md）"}
        _STATE["starting"] = True
        _STATE["comfy_dir"] = d
        _STATE["started_at"] = time.time()

    log_path = os.path.join(d, "retouchlab_comfyui.log")
    cmd = [os.path.join(d, "python_embeded", "python.exe"), "-s",
           os.path.join("ComfyUI", "main.py"),
           "--windows-standalone-build", "--lowvram", "--disable-assets",
           "--listen", "127.0.0.1", "--port", str(PORT)]
    try:
        log = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, cwd=d, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt"
            else 0)
    except Exception as exc:                       # pragma: no cover
        with _LOCK:
            _STATE["starting"] = False
            _STATE["last_error"] = str(exc)[:200]
        return {"state": "error", "detail": str(exc)[:200]}
    with _LOCK:
        _STATE["proc"] = proc
        _STATE["last_error"] = ""

    def _wait_ready():
        deadline = time.time() + timeout
        while time.time() < deadline:
            if is_ready():
                with _LOCK:
                    _STATE["starting"] = False
                    _STATE["last_used"] = time.time()
                return
            if proc.poll() is not None:            # 进程自己退了
                tail = ""
                try:
                    with open(log_path, encoding="utf-8",
                              errors="replace") as f:
                        tail = "\n".join(f.read().splitlines()[-6:])
                except Exception:
                    pass
                with _LOCK:
                    _STATE["starting"] = False
                    _STATE["last_error"] = tail[:400] or "进程提前退出"
                return
            time.sleep(2)
        with _LOCK:
            _STATE["starting"] = False
            _STATE["last_error"] = "等待就绪超时（%ds）" % timeout

    if background:
        threading.Thread(target=_wait_ready, daemon=True).start()
        return {"state": "starting"}
    _wait_ready()
    return {"state": "ready" if is_ready() else "error",
            "detail": _STATE["last_error"]}


def stop() -> dict:
    """停掉 ComfyUI：只动我们拉起的进程；外部自启的一律不碰。"""
    with _LOCK:
        proc = _STATE["proc"]
        _STATE["proc"] = None
        _STATE["starting"] = False
    if proc is not None and proc.poll() is None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        except Exception:
            pass
    if not is_ready():
        return {"state": "stopped"}
    # 到这里说明 ready 但句柄不在手（不是我们拉起的，或句柄丢了）。
    # 前者不碰；后者是"我们拉起后服务重启"的半死场景，按命令行兜底清理。
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
         "| Where-Object {$_.CommandLine -like '*ComfyUI*main.py*--port %d*'} "
         "| Select-Object -ExpandProperty ProcessId" % PORT],
        capture_output=True).stdout.decode("utf-8", "replace")
    killed = []
    for pid in (x for x in out.split() if x.isdigit()):
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"],
                       capture_output=True)
        killed.append(pid)
    time.sleep(1)
    return {"state": "stopped" if not is_ready() else "running",
            "killed": killed}


def touch():
    """记一次使用，空闲自动停止从这里重新计时。"""
    with _LOCK:
        _STATE["last_used"] = time.time()


def status() -> dict:
    ready = is_ready()
    with _LOCK:
        return {
            "ready": ready,
            "starting": _STATE["starting"] and not ready,
            "comfy_dir": _STATE["comfy_dir"] or find_comfy_dir(),
            "seconds_since_start": round(time.time() - _STATE["started_at"], 1)
            if _STATE["started_at"] else None,
            "idle_stop_after_s": IDLE_STOP_SECONDS,
            "last_error": _STATE["last_error"],
        }


def _idle_watcher():
    """daemon：空闲超时自动停（仅停我们拉起的）。"""
    while True:
        time.sleep(60)
        with _LOCK:
            proc = _STATE["proc"]
            last = _STATE["last_used"]
            starting = _STATE["starting"]
        if proc is None or starting or not is_ready():
            continue
        if last and time.time() - last > IDLE_STOP_SECONDS:
            stop()


def install_watcher():
    threading.Thread(target=_idle_watcher, daemon=True).start()
