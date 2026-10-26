# -*- coding: utf-8 -*-
"""首次运行时的模型下载器（瘦发行版的核心）。

发行包不再内置 980MB 模型；服务启动后发现模型缺失，前端显示
「首次初始化」引导层，调用 /api/models 接口驱动这里的逻辑从
GitHub Release（models-v1，清单见 tools/models_manifest.json）
下载并 sha256 校验。

与 tools/fetch_models.py 的关系：那份是命令行版（源码用户随
「一键安装环境.bat」用）；这里是进程内版，被 server 调用、可查进度、
可在 UI 上重试。清单文件两处共用：源码在 tools/ 下，打包后由
assemble_dist 摆到 exe 旁（manifest_path() 负责双形态解析）。

线程模型：fetch() 在后台线程里逐文件下载；_STATE 记录进度，
status() 供 API 轮询。同一时刻只允许一个下载任务。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

from .paths import models_dir

# 允许的下载源：本仓库 Release 及其重定向落地域（GitHub 资产 CDN）
_ALLOWED_HOSTS = {"github.com", "objects.githubusercontent.com",
                  "codeload.github.com", "api.github.com"}

# 下载状态：idle / running / done / error
_STATE = {
    "phase": "idle",            # idle|running|done|error
    "current": "",              # 正在下载的文件
    "done_files": 0,
    "total_files": 0,
    "bytes_done": 0,
    "bytes_total": 0,
    "error": "",
    "started_at": 0.0,
}
_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None


def manifest_path() -> str:
    """清单位置：源码在 tools/，打包后在 exe 旁（assemble 负责摆放）。"""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
        p = os.path.join(base, "models_manifest.json")
        if os.path.isfile(p):
            return p
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "tools", "models_manifest.json")


def _load_manifest() -> dict:
    with open(manifest_path(), encoding="utf-8") as f:
        return json.load(f)


def _safe_url(url: str) -> str:
    """只放行 https + GitHub 域。清单被篡改时在这里拦住，而不是任由
    服务端向任意目标发请求。"""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in _ALLOWED_HOSTS:
        raise ValueError("非法下载源: %s" % url[:120])
    return url


def _safe_dest(rel_path: str) -> str:
    """把清单里的相对路径钉死在 models/ 目录内（防路径穿越）。"""
    if ".." in rel_path.replace("\\", "/").split("/"):
        raise ValueError("非法路径: " + rel_path)
    base = os.path.realpath(models_dir())
    dest = os.path.realpath(os.path.join(base, *rel_path.split("/")))
    if not dest.startswith(base + os.sep):
        raise ValueError("路径越界: " + rel_path)
    return dest


def _verify(path: str, sha256: str) -> bool:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return False
    return h.hexdigest() == sha256


def missing_files() -> list:
    """缺失/损坏的模型文件清单（manifest 顺序）。"""
    out = []
    for item in _load_manifest()["files"]:
        dest = _safe_dest(item["path"])
        if not (os.path.exists(dest)
                and os.path.getsize(dest) == item["size"]
                and _verify(dest, item["sha256"])):
            out.append(item["path"])
    return out


def _asset_name(rel_path: str) -> str:
    return rel_path.replace("/", "_")


def _download_one(url: str, dest: str, progress: dict, retries: int = 12):
    """单文件断点续传下载；progress 直接累加进共享 _STATE。

    retries 给得宽（12 次 × 递增退避）：这台机器到 GitHub 的链路
    实测会连续超时几次才通，重试预算太紧会让 980MB 的首次下载
    轻易整体失败。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    for attempt in range(retries):
        try:
            resume = os.path.getsize(tmp) if os.path.exists(tmp) else 0
            req = urllib.request.Request(_safe_url(url))
            if resume:
                req.add_header("Range", "bytes=%d-" % resume)
            with urllib.request.urlopen(req, timeout=60) as r, \
                    open(tmp, "ab" if resume else "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    progress["bytes_done"] += len(chunk)
                    # 逐 MB 刷新共享进度：大文件（300MB+）如果等整文件
                    # 完成才更新，UI 进度条会卡 0% 好几分钟像死机。
                    with _LOCK:
                        _STATE["bytes_done"] = progress["bytes_done"]
            os.replace(tmp, dest)
            return
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def _worker(paths: list):
    m = _load_manifest()
    by_path = {x["path"]: x for x in m["files"]}
    progress = {"bytes_done": 0}
    try:
        for rel in paths:
            item = by_path[rel]
            dest = _safe_dest(rel)
            url = "%s/%s" % (m["base_url"], _asset_name(rel))
            with _LOCK:
                _STATE["current"] = rel
            _download_one(url, dest, progress)
            if not _verify(dest, item["sha256"]):
                os.remove(dest)
                raise RuntimeError("sha256 校验失败: " + rel)
            with _LOCK:
                _STATE["done_files"] += 1
                _STATE["bytes_done"] = progress["bytes_done"]
        with _LOCK:
            _STATE["phase"] = "done"
            _STATE["current"] = ""
    except Exception as exc:
        with _LOCK:
            _STATE["phase"] = "error"
            _STATE["error"] = str(exc)[:300]


def fetch() -> dict:
    """启动（或继续）下载缺失模型。已在跑时直接返回当前进度。"""
    global _THREAD
    with _LOCK:
        if _STATE["phase"] == "running":
            return status()
        need = missing_files()
        m = _load_manifest()
        if not need:
            _STATE.update(phase="done", current="", error="",
                          total_files=len(m["files"]))
            return status()
        by_path = {x["path"]: x for x in m["files"]}
        have = len(m["files"]) - len(need)
        _STATE.update(phase="running", current=need[0], error="",
                      done_files=0, total_files=len(need),
                      bytes_done=0,
                      bytes_total=sum(by_path[p]["size"] for p in need),
                      started_at=time.time(),
                      files_ready=have)
    _THREAD = threading.Thread(target=_worker, args=(need,), daemon=True)
    _THREAD.start()
    return status()


def status() -> dict:
    m = _load_manifest()
    with _LOCK:
        s = dict(_STATE)
        running = s["phase"] == "running"
        ready_now = s.get("files_ready", 0) + s["done_files"]
    if not running:
        ready_now = len(m["files"]) - len(missing_files())
    s["installed"] = ready_now
    s["total"] = len(m["files"])
    return s
