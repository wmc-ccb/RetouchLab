@echo off
chcp 65001 >nul
title ZhaJinWenQianCake
setlocal

rem ============================================================
rem  ZhaJinWenQianCake 启动（双击这个文件即可）
rem
rem  做的事：
rem    1. 找 Python（优先用打包时记录的路径，其次 PATH 里的 python）
rem    2. 跑 tools\launch.py —— 它会自动挑端口、等服务就绪后开浏览器
rem    3. 日志在 out\launcher.log
rem
rem  想允许局域网访问：在这个文件里把 --lan 加到最后一行
rem ============================================================

set "ROOT=%~dp0"
cd /d "%ROOT%"

rem ---- 找 Python（优先「一键安装环境.bat」建好的 .venv）----
set "PY="
if exist "%ROOT%.venv\Scripts\python.exe" set "PY=%ROOT%.venv\Scripts\python.exe"
if not defined PY if exist "%ROOT%python_embed\python.exe" set "PY=%ROOT%python_embed\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo.
  echo  没有找到 Python。请先双击「一键安装环境.bat」，
  echo  它会自动搞定一切（需要已安装 Python 3.11-3.13）。
  echo.
  pause
  exit /b 1
)

echo.
echo   正在启动 ZhaJinWenQianCake ...
echo   （首次启动要读模型，需要十几秒）
echo.

"%PY%" -X utf8 "%ROOT%tools\launch.py" %*

echo.
echo   服务已停止。
pause
