# -*- coding: utf-8 -*-
"""打包后的收尾：把运行时资源摆到 exe 旁边，生成发行目录。

PyInstaller 只负责「代码 + 依赖」，运行时资源按 engine/paths.py 的约定
放在 exe 同级的可写目录里。本脚本做这些事：

    dist/RetouchLab/
        RetouchLab.exe           <- PyInstaller 产物
        _internal/               <- PyInstaller 内部依赖
        models/                  <- 从项目里拷（0.95GB）
        web/                     <- 从项目里拷
        data/                    <- 空目录（用户数据）
        out/                     <- 空目录（日志/导出）
        启动.bat                 <- 给用户双击的入口
        README-安装说明.txt

用法：
    python tools/assemble_dist.py
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "RetouchLab")


def copytree(src, dst, skip=()):
    if not os.path.isdir(src):
        print("  跳过（不存在）:", src)
        return 0
    n = 0
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames[:] = [d for d in dirnames if d not in skip
                       and not d.startswith("__pycache__")]
        rel = os.path.relpath(dirpath, src)
        target_dir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(target_dir, exist_ok=True)
        for f in filenames:
            if f.startswith("."):
                continue
            try:
                shutil.copy2(os.path.join(dirpath, f),
                             os.path.join(target_dir, f))
                n += 1
            except OSError as e:
                print("  拷贝失败 %s: %s" % (f, e))
    return n


BAT = """@echo off
chcp 65001 >nul
title RetouchLab
rem 路径含空格/中文也能用：全程用 %~dp0 并加引号
cd /d "%~dp0"

if not exist "%~dp0RetouchLab.exe" (
  echo.
  echo   找不到 RetouchLab.exe。
  echo   请确认这个 bat 和 RetouchLab.exe 在同一个文件夹里。
  echo.
  pause
  exit /b 1
)

echo.
echo   ============================================
echo    RetouchLab   人像精修引擎
echo   ============================================
echo.
echo   正在启动，浏览器会自动打开...
echo   第一次启动要读模型，需要十几秒，请稍等。
echo   关闭这个窗口即停止服务。
echo.

"%~dp0RetouchLab.exe" %*
set RC=%ERRORLEVEL%

echo.
if not "%RC%"=="0" (
  echo   启动失败（退出码 %RC%）。
  echo   详细信息看 out 目录下的 error.log / launcher.log
  echo.
) else (
  echo   服务已停止。
  echo.
)
pause
"""

README = """RetouchLab 使用说明
=========================

【启动】
  双击「启动.bat」即可。浏览器会自动打开 http://127.0.0.1:8765

  第一次启动要读取模型，需要十几秒，属正常现象。
  关闭那个黑色窗口 = 停止服务。

【目录说明】
  models\\    模型文件（不可删除；想换模型直接替换同名文件）
  web\\       前端页面
  data\\      你的数据（预设、图库来源、导出记录）
  out\\       日志和临时输出

【生成式功能（AI 布景 / 创成式填充 / 道具修复）】
  这三个功能需要额外在后台运行 ComfyUI（默认 127.0.0.1:8388）
  并准备好 SD inpainting 模型，本安装包不含这部分。

  没有 ComfyUI 时，其余全部功能正常，只是这三个工具会提示不可用。
  默认的「祛路人」「消除」走 LaMa，不需要 ComfyUI。

【局域网 / 手机访问】
  用记事本打开「启动.bat」，把这一行
      "%~dp0RetouchLab.exe" %*
  改成
      "%~dp0RetouchLab.exe" --lan %*
  然后手机连同一个 Wi-Fi，访问提示里显示的局域网地址。

【卸载】
  删除整个文件夹即可，不写注册表。

【遇到问题】
  1. 看 out\\launcher.log —— 启动阶段的问题
  2. 看 out\\error.log   —— 运行中接口报错（含完整堆栈）
  3. 浏览器打开 http://127.0.0.1:8765/api/diagnose
     —— 一次性列出运行环境：是否打包运行、Python 版本、
        torch/CUDA 是否真的能用（会实际跑一次 GPU 运算）、
        8 个模型文件是否就位。报bug时把这段贴出来最快。

  常见的「超清很慢」：如果 diagnose 里 gpu_works 是 false，
  说明显卡没被用上，会退回 CPU（慢 10 倍以上）。
"""


def main():
    if not os.path.isdir(DIST):
        print("找不到 %s" % DIST)
        print("请先运行：pyinstaller RetouchLab.spec --noconfirm")
        return 1

    print("组装发行目录:", DIST)

    # 1) 运行时资源
    n = copytree(os.path.join(ROOT, "web"), os.path.join(DIST, "web"))
    print("  web/      %d 个文件" % n)

    n = copytree(os.path.join(ROOT, "models"), os.path.join(DIST, "models"),
                 skip=("__pycache__",))
    print("  models/   %d 个文件" % n)

    # 2) 空目录
    for d in ("data", "out"):
        os.makedirs(os.path.join(DIST, d), exist_ok=True)
    print("  data/ out/ 已建")

    # 3) 启动脚本 + 说明
    with open(os.path.join(DIST, "启动.bat"), "w",
              encoding="utf-8-sig", newline="\r\n") as f:
        f.write(BAT)
    with open(os.path.join(DIST, "README-使用说明.txt"), "w",
              encoding="utf-8-sig", newline="\r\n") as f:
        f.write(README)
    print("  启动.bat / README 已写")

    # 4) 体积统计
    total = 0
    for dp, _, fs in os.walk(DIST):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    print("\n发行目录总大小: %.2f GB" % (total / 1024 ** 3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
