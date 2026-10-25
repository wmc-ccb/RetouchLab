# -*- coding: utf-8 -*-
"""打包后的收尾：把运行时资源摆到 exe 旁边，生成发行目录。

PyInstaller 只负责「代码 + 依赖」，运行时资源按 engine/paths.py 的约定
放在 exe 同级的可写目录里。本脚本做这些事：

    dist/ZhaJinWenQianCake/
        ZhaJinWenQianCake.exe           <- PyInstaller 产物
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
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist", "ZhaJinWenQianCake")


def read_version():
    """从 server/app.py 取 APP_VERSION——版本号只有一处事实来源。"""
    with open(os.path.join(ROOT, "server", "app.py"),
              encoding="utf-8") as f:
        for line in f:
            if line.startswith("APP_VERSION"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return "0.0.0"


def find_signtool():
    """Windows SDK 里的 signtool.exe，找不到返回 None。"""
    hits = []
    for base in (r"C:\Program Files (x86)\Windows Kits\10\bin",
                 r"C:\Program Files\Windows Kits\10\bin"):
        if not os.path.isdir(base):
            continue
        for ver in sorted(os.listdir(base), reverse=True):
            p = os.path.join(base, ver, "x64", "signtool.exe")
            if os.path.isfile(p):
                hits.append(((0, 0), p) if not ver[0].isdigit()
                            else (tuple(int(x) for x in ver.split(".")[:2]), p))
    return max(hits)[1] if hits else None


def sign_dist():
    """可选代码签名：没配证书就跳过并说明，绝不把打包流程卡死。

    配置（环境变量）：
        RETOUCHLAB_CERT_PFX   证书文件路径（.pfx）
        RETOUCHLAB_CERT_PASS  证书密码
    为什么放在 assemble 而不是 spec：签名是发布动作不是构建动作，
    本地自用打包不应该被"没买证书"挡住。
    """
    pfx = os.environ.get("RETOUCHLAB_CERT_PFX", "").strip()
    if not pfx:
        print("  代码签名：未配置 RETOUCHLAB_CERT_PFX，跳过"
              "（发行给外部用户前建议签名，见 README「发布」章节）")
        return False
    if not os.path.isfile(pfx):
        print("  代码签名：证书文件不存在 %s，跳过" % pfx)
        return False
    tool = find_signtool()
    if not tool:
        print("  代码签名：找不到 signtool.exe（需要 Windows SDK），跳过")
        return False
    exe = os.path.join(DIST, "ZhaJinWenQianCake.exe")
    cmd = [tool, "sign", "/fd", "SHA256", "/tr",
           "http://timestamp.digicert.com", "/td", "SHA256",
           "/f", pfx]
    pwd = os.environ.get("RETOUCHLAB_CERT_PASS", "")
    if pwd:
        cmd += ["/p", pwd]
    cmd.append(exe)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print("  代码签名：完成（%s）" % os.path.basename(tool))
        subprocess.run([tool, "verify", "/pa", "/v", exe],
                       capture_output=True)
        return True
    print("  代码签名失败: %s" % (r.stderr or r.stdout).strip()[:300])
    return False


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
title ZhaJinWenQianCake
rem 路径含空格/中文也能用：全程用 %~dp0 并加引号
cd /d "%~dp0"

if not exist "%~dp0ZhaJinWenQianCake.exe" (
  echo.
  echo   找不到 ZhaJinWenQianCake.exe。
  echo   请确认这个 bat 和 ZhaJinWenQianCake.exe 在同一个文件夹里。
  echo.
  pause
  exit /b 1
)

echo.
echo   ============================================
echo    ZhaJinWenQianCake   人像精修引擎
echo   ============================================
echo.
echo   正在启动，应用窗口会自动打开...
echo   第一次启动要读模型，需要十几秒，请稍等。
echo   关闭应用窗口即退出。
echo.

"%~dp0ZhaJinWenQianCake.exe" %*
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

README = """ZhaJinWenQianCake 使用说明
=========================

【启动】
  双击 ZhaJinWenQianCake.exe —— 会打开独立的应用窗口（不是浏览器）。

  第一次启动要读取模型，先显示启动页十几秒，属正常现象。
  关闭应用窗口 = 退出程序。

  兼容说明：应用窗口用的是系统自带的 WebView2（Win10/11 内置）。
  极老的系统没有它时会自动改用浏览器打开，功能完全一样。

【目录说明】
  models\\    模型文件（不可删除；想换模型直接替换同名文件）
  web\\       前端页面
  data\\      你的数据（预设、图库来源、导出记录）
  out\\       日志和临时输出

【生成式功能（AI 布景 / 创成式填充 / 道具修复）】
  这三个功能需要额外的 ComfyUI 组件（约 13GB），本安装包不含。
  装好之后日常零命令：打开这三个工具的面板，点「启动 AI 服务」
  按钮即可（直接点生成也会自动启动），空闲 15 分钟自动停止。
  首次安装方法见 GENERATIVE-安装指南.txt。

  没有 ComfyUI 时，其余全部功能正常，只是这三个工具会提示未安装。
  默认的「祛路人」「消除」走 LaMa，不需要 ComfyUI。

【局域网 / 手机访问】
  建一个快捷方式，目标后面加上 --lan（注意前面有空格）：
      ...\\ZhaJinWenQianCake.exe --lan
  然后手机连同一个 Wi-Fi，浏览器访问启动日志(out\\launcher.log)
  里的局域网地址。电脑上仍是应用窗口，不受影响。

【卸载】
  删除整个文件夹即可，不写注册表。

【遇到问题】
  界面右上角「?」→ 复制诊断报告，把内容发给维护者最快。
  或：
  1. 看 out\\launcher.log —— 启动阶段的问题
  2. 看 out\\error.log   —— 运行中接口报错（含完整堆栈）
  3. 浏览器打开 http://127.0.0.1:8765/api/diagnose
     —— 一次性列出运行环境：是否打包运行、Python 版本、
        torch/CUDA 是否真的能用（会实际跑一次 GPU 运算）、
        8 个模型文件是否就位。

  常见的「超清很慢」：如果 diagnose 里 gpu_works 是 false，
  说明显卡没被用上，会退回 CPU（慢 10 倍以上）。
"""


def main():
    if not os.path.isdir(DIST):
        print("找不到 %s" % DIST)
        print("请先运行：pyinstaller ZhaJinWenQianCake.spec --noconfirm")
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

    # 3.5) 构建戳 + 生成式安装器随包
    # build-stamp.txt 放 exe 旁（data/ 会被用户数据占用，_internal/ 不可见），
    # /api/version 会读它：用户报障时能对上"哪一次打包"。
    stamp = time.strftime("%Y-%m-%d") + ".1"
    with open(os.path.join(DIST, "build-stamp.txt"), "w",
              encoding="utf-8") as f:
        f.write(stamp)
    print("  build-stamp.txt = %s (版本 %s)" % (stamp, read_version()))
    gi = os.path.join(ROOT, "tools", "install_generative.py")
    if os.path.isfile(gi):
        shutil.copy2(gi, os.path.join(DIST, "install_generative.py"))
        gd = os.path.join(ROOT, "GENERATIVE.md")
        if os.path.isfile(gd):
            shutil.copy2(gd, os.path.join(DIST, "GENERATIVE-安装指南.txt"))
        print("  install_generative.py 已随包（生成式一键安装）")

    # 4) 可选代码签名（未配置证书时静默跳过，见 sign_dist 注释）
    sign_dist()

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
