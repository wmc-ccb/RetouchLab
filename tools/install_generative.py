#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成式三件套（AI 布景 / 创成式填充 / 道具修复）的一键安装与启动。

RetouchLab 本体不含扩散模型：这三样功能跑在外挂 ComfyUI（默认
http://127.0.0.1:8388）上，需要两个模型文件：

    checkpoints/sd15_inpaint.ckpt                        (~2.0 GB)
    controlnet/control_v11p_sd15_inpaint_fp16.safetensors (~0.7 GB)

用法
----
    python tools/install_generative.py check     # 体检：缺什么报什么
    python tools/install_generative.py start     # 启动 ComfyUI(8388) 并自检
    python tools/install_generative.py stop      # 停掉 8388 上的 ComfyUI
    python tools/install_generative.py install   # 全新机器：下载并装齐
        --dir D:\\ComfyUI      安装目录（默认 C:\\ComfyUI）
        --skip-models         已有模型时跳过下载（省 2.7 GB）

check/start 只管"用起来"；install 才动硬盘。安装记录写在
data/generative.json，start 从那里找 ComfyUI 目录。

设计取舍：本机（开发机）已经装好 ComfyUI 便携版与两个模型，
check/start 直接可用；install 面向全新机器，下载走官方发布源
（ComfyUI GitHub Release 的 7z 便携包 + HuggingFace 直链），
不镜像、不打包进发行版——13GB 的东西塞进安装包只会拖死下载。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

PORT = 8388
BASE = "http://127.0.0.1:%d" % PORT
CHECKPOINT = "sd15_inpaint.ckpt"
CONTROLNET = "control_v11p_sd15_inpaint_fp16.safetensors"

# 全新机器下载源（官方发布，不经过第三方镜像）
URL_PORTABLE_7Z = ("https://github.com/comfyanonymous/ComfyUI/releases/"
                   "download/latest/ComfyUI_windows_portable_nvidia.7z")
URL_7ZR = "https://www.7-zip.org/a/7zr.exe"
URL_CHECKPOINT = ("https://huggingface.co/runwayml/"
                  "stable-diffusion-inpainting/resolve/main/"
                  "sd-v1-5-inpainting.ckpt")
URL_CONTROLNET = ("https://huggingface.co/ControlNet-v1-1/resolve/main/"
                  "control_v11p_sd15_inpaint_fp16.safetensors")

CONFIG = None  # data/generative.json，install 时写入


def _config_path():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "data", "generative.json")


def _load_config():
    global CONFIG
    try:
        with open(_config_path(), encoding="utf-8") as f:
            CONFIG = json.load(f)
    except Exception:
        CONFIG = {}
    return CONFIG


def find_comfy_dir(cli_dir=None):
    """定位 ComfyUI 便携版根目录（含 python_embeded 与 ComfyUI/main.py）。

    优先级：命令行 > 环境变量 > data/generative.json > 常见路径。
    """
    candidates = []
    if cli_dir:
        candidates.append(cli_dir)
    if os.environ.get("RETOUCHLAB_COMFYUI_DIR"):
        candidates.append(os.environ["RETOUCHLAB_COMFYUI_DIR"])
    cfg = CONFIG or _load_config()
    if cfg.get("comfy_dir"):
        candidates.append(cfg["comfy_dir"])
    home = os.path.expanduser("~")
    candidates += [
        os.path.join(home, "Desktop", "ComfyUI"),
        r"C:\ComfyUI",
        os.path.join(home, "ComfyUI"),
        os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "ComfyUI"),
    ]
    for c in candidates:
        if (c and os.path.isdir(os.path.join(c, "ComfyUI"))
                and os.path.isfile(os.path.join(
                    c, "python_embeded", "python.exe"))):
            return c
    return None


def _http_ok(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def comfy_running():
    """ComfyUI 活着且注册了 inpaint checkpoint（与 engine/genfill 同判据）。"""
    if not _http_ok(BASE + "/system_stats", timeout=2):
        return False
    try:
        with urllib.request.urlopen(BASE + "/object_info", timeout=5) as r:
            info = json.load(r)
        ckpts = info.get("CheckpointLoaderSimple", {}).get(
            "input", {}).get("required", [])[0]
        return any(CHECKPOINT in str(c) for c in ckpts)
    except Exception:
        return False


def _models_state(comfy_dir):
    ck = os.path.join(comfy_dir, "ComfyUI", "models", "checkpoints",
                      CHECKPOINT)
    cn = os.path.join(comfy_dir, "ComfyUI", "models", "controlnet",
                      CONTROLNET)
    return {"checkpoint": (os.path.isfile(ck), ck),
            "controlnet": (os.path.isfile(cn), cn)}


def cmd_check(args):
    print("生成式组件体检")
    print("-" * 56)
    d = find_comfy_dir(args.dir)
    if d:
        print("[OK]      ComfyUI 便携版: %s" % d)
    else:
        print("[缺失]    未找到 ComfyUI 便携版")
        print("          → python tools/install_generative.py install")
        return 1
    for name, (ok, path) in _models_state(d).items():
        size = ""
        if ok:
            size = " (%.1f GB)" % (os.path.getsize(path) / 1024 ** 3)
        print("[%s] %s%s: %s" % ("OK   " if ok else "缺失  ", name,
                                  size, path if ok else "→ install 补齐"))
    if comfy_running():
        print("[OK]      服务已在 %s 应答且模型注册" % BASE)
        print("\n全部就绪：RetouchLab 的 AI 布景 / 创成式填充 / 道具修复可用。")
        return 0
    print("[未启动]  %s 无应答" % BASE)
    print("          → python tools/install_generative.py start")
    return 2


def cmd_start(args):
    d = find_comfy_dir(args.dir)
    if not d:
        print("未找到 ComfyUI，先运行 install。")
        return 1
    missing = [k for k, (ok, _) in _models_state(d).items() if not ok]
    if missing:
        print("缺模型: %s → 先 install/check 补齐" % ", ".join(missing))
        return 1
    if comfy_running():
        print("ComfyUI 已在 %s 运行。" % BASE)
        return 0
    exe = os.path.join(d, "python_embeded", "python.exe")
    main = os.path.join("ComfyUI", "main.py")
    # --lowvram：与显存大的机器相比略慢，但保证 8GB 卡也能跑；
    # 8388 而非默认 8188：见 engine/genfill.py 头注（保留端口段冲突）；
    # --disable-assets：资产库只服务桌面 UI，API 模式用不到；旧版本
    # 升级后 comfyui.db 缺表会在启动时卡死，禁掉最稳。
    cmd = [exe, "-s", main, "--windows-standalone-build", "--lowvram",
           "--disable-assets",
           "--listen", "127.0.0.1", "--port", str(PORT)]
    log = open(os.path.join(d, "retouchlab_comfyui.log"), "w",
               encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=d, stdout=log, stderr=log,
                            creationflags=subprocess.CREATE_NO_WINDOW
                            if os.name == "nt" else 0)
    print("ComfyUI 启动中 (PID %d)，等待应答…" % proc.pid)
    # 装了很多自定义节点的 ComfyUI 光 import 就要一两分钟，90s 会误报
    for i in range(240):
        if comfy_running():
            print("[OK] 服务就绪: %s（日志: %s）"
                  % (BASE, os.path.join(d, "retouchlab_comfyui.log")))
            print("回到 RetouchLab 刷新即可使用生成式三件套。")
            return 0
        if proc.poll() is not None:
            break
        time.sleep(1)
    print("[失败] 240 秒内未就绪。日志尾部:")
    try:
        with open(os.path.join(d, "retouchlab_comfyui.log"),
                  encoding="utf-8", errors="replace") as f:
            tail = f.read().splitlines()[-12:]
        print("\n".join("    " + l for l in tail))
    except Exception:
        pass
    return 1


def cmd_stop(args):
    pids = set()
    # 1) 正常情况：8388 在监听
    out = subprocess.run(["netstat", "-ano"],
                         capture_output=True).stdout.decode(
                             "utf-8", "replace")
    for line in out.splitlines():
        if ":%d" % PORT in line and "LISTENING" in line:
            pids.add(line.split()[-1])
    # 2) 兜底：进程还在但没监听了（半死状态，netstat 找不到）。
    # 实测出现过：启动器 shim 被杀后 ComfyUI 本体继续占 8GB 内存。
    try:
        q = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter "
             "\"Name='python.exe'\" | Where-Object {$_.CommandLine "
             "-like '*ComfyUI*main.py*--port %d*'} | "
             "Select-Object -ExpandProperty ProcessId" % PORT],
            capture_output=True).stdout.decode("utf-8", "replace")
        pids.update(x for x in q.split() if x.isdigit())
    except Exception:
        pass
    if not pids:
        print("8388 上没有进程。")
        return 0
    for pid in pids:
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"],
                       capture_output=True)
        print("已结束 PID %s" % pid)
    return 0


def _download(url, dest, label):
    if os.path.exists(dest):
        print("  已存在，跳过: %s" % dest)
        return
    print("  下载 %s …" % label)
    print("    %s" % url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1024 * 512)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                sys.stdout.write("\r    %.1f / %.1f GB" %
                                 (got / 1024 ** 3, total / 1024 ** 3))
                sys.stdout.flush()
    print()
    os.replace(tmp, dest)


def cmd_install(args):
    target = args.dir or r"C:\ComfyUI"
    if find_comfy_dir(target if os.path.isdir(target) else None) \
            and os.path.isdir(os.path.join(target, "ComfyUI")):
        print("ComfyUI 已存在于 %s，跳过程序下载。" % target)
    else:
        os.makedirs(target, exist_ok=True)
        z7 = os.path.join(target, "7zr.exe")
        _download(URL_7ZR, z7, "7zr.exe (解压工具)")
        arc = os.path.join(target, "comfyui_portable.7z")
        _download(URL_PORTABLE_7Z, arc,
                  "ComfyUI 便携版 (~1.5 GB, 含 Python)")
        print("  解压中…")
        r = subprocess.run([z7, "x", "-y", "-o" + target, arc])
        if r.returncode != 0:
            print("解压失败（exit %d）" % r.returncode)
            return 1
        os.remove(arc)
    if not args.skip_models:
        _download(URL_CHECKPOINT,
                  os.path.join(target, "ComfyUI", "models", "checkpoints",
                               CHECKPOINT),
                  "SD1.5 inpainting 模型 (~2.0 GB)")
        _download(URL_CONTROLNET,
                  os.path.join(target, "ComfyUI", "models", "controlnet",
                               CONTROLNET),
                  "inpaint ControlNet (~0.7 GB)")
    cfg = CONFIG or _load_config()
    cfg["comfy_dir"] = target
    os.makedirs(os.path.dirname(_config_path()), exist_ok=True)
    with open(_config_path(), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    print("\n安装完成 → python tools/install_generative.py start")
    return cmd_check(argparse.Namespace(dir=None))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["check", "start", "stop", "install"])
    ap.add_argument("--dir", help="ComfyUI 目录（check/start）或安装目录"
                                  "（install，默认 C:\\ComfyUI）")
    ap.add_argument("--skip-models", action="store_true",
                    help="install 时跳过模型下载")
    args = ap.parse_args()
    _load_config()
    return {"check": cmd_check, "start": cmd_start, "stop": cmd_stop,
            "install": cmd_install}[args.command](args)


if __name__ == "__main__":
    sys.exit(main() or 0)
