# -*- coding: utf-8 -*-
"""资源路径解析：同时支持「源码运行」和「PyInstaller 打包运行」。

两种运行方式的目录结构完全不同：

    源码运行
      <root>/engine/paths.py      <- __file__
      <root>/models/*.onnx
      <root>/web/index.html

    PyInstaller onedir
      <dist>/RetouchLab/RetouchLab.exe
      <dist>/RetouchLab/_internal/engine/...   (打包进去的代码)
      <dist>/RetouchLab/models/...              (运行时目录，可写)
      <dist>/RetouchLab/web/...                 (运行时目录，可写)

所以**模型和 web 资源一律放在 exe 旁边的可写目录**（不要打进 _internal），
理由：
  1. 模型近 1GB，放进 _internal 每次都要跟 exe 一起被校验/复制
  2. 生成式组件包是后装的，必须能往这个目录里加东西
  3. 用户想换模型/加模型时直接拷文件就行，不用重新打包

环境变量 RETOUCHLAB_HOME 可覆盖运行时根目录（便携 U 盘场景）。
"""
from __future__ import annotations

import os
import sys


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _exe_dir() -> str:
    """打包后：exe 所在目录。源码运行：项目根目录。"""
    if is_frozen():
        return os.path.dirname(os.path.abspath(sys.executable))
    # 源码：本文件在 <root>/engine/paths.py
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def home() -> str:
    """运行时根目录（放 models/ web/ data/ out/ 的地方）。"""
    env = os.environ.get("RETOUCHLAB_HOME", "").strip()
    if env:
        return os.path.abspath(env)
    return _exe_dir()


def models_dir() -> str:
    return os.path.join(home(), "models")


def web_dir() -> str:
    return os.path.join(home(), "web")


def data_dir() -> str:
    return os.path.join(home(), "data")


def out_dir() -> str:
    return os.path.join(home(), "out")


def ensure_dirs() -> None:
    """首次启动时把该有的目录建出来。"""
    for d in (models_dir(), data_dir(), out_dir()):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass


def register_cuda_dlls() -> bool:
    """把 torch 的 CUDA 运行库加进 DLL 搜索路径（**必须在 import torch 之前**）。

    打包版踩过的坑：PyInstaller 会把 torch/lib 里的 cublas/cudnn 等
    DLL 一起收进 _internal/torch/lib，但 Windows 的默认 DLL 搜索顺序里
    没有这个目录。结果是 `torch.cuda.is_available()` 返回 True，而
    `torch.cuda.init()` 在第一次真正分配显存时找不到 cublasLt64_12.dll，
    torch 内部把异常吞掉后静默回退 CPU——表现就是"GPU 明明可用，超清
    却慢了 12 倍"（实测打包版 109s vs 源码版 9.2s）。

    add_dll_directory 是 3.8+ 的官方做法；同时补 PATH，因为少数扩展
    模块仍按 PATH 找依赖。
    """
    if os.name != "nt":
        return False
    added = False
    candidates = []
    try:
        import torch  # noqa: F401  只为取路径
        candidates.append(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except Exception:
        pass
    # 打包版：_internal/torch/lib
    if is_frozen():
        candidates.append(os.path.join(os.path.dirname(sys.executable),
                                       "_internal", "torch", "lib"))
    for d in candidates:
        if not os.path.isdir(d):
            continue
        try:
            os.add_dll_directory(d)
            added = True
        except (AttributeError, OSError):
            pass
        try:
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
            added = True
        except Exception:
            pass
    return added
