@echo off
chcp 65001 >nul
title ZhaJinWenQianCake 打包
setlocal
cd /d "%~dp0"

rem ============================================================
rem  一键打包：PyInstaller 出 exe + 组装运行时资源
rem
rem  产物在 dist\ZhaJinWenQianCake\ ，直接压缩发给用户即可。
rem  前置：python -m pip install pyinstaller
rem ============================================================

set "PY="
if exist "%~dp0python_embed\python.exe" set "PY=%~dp0python_embed\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo   找不到 Python，无法打包。
  pause & exit /b 1
)

echo.
echo   [1/3] 检查 PyInstaller
"%PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo   未安装，正在安装...
  "%PY%" -m pip install pyinstaller --quiet || (echo   安装失败 & pause & exit /b 1)
)

echo   [2/3] PyInstaller 打包
"%PY%" -m PyInstaller ZhaJinWenQianCake.spec --noconfirm --clean
if errorlevel 1 (
  echo.
  echo   打包失败。常见原因：
  echo     - dist\ZhaJinWenQianCake 里有进程在跑（先关掉）
  echo     - 磁盘空间不足（需要约 8GB）
  echo.
  pause & exit /b 1
)

echo   [3/3] 组装运行时资源（models / web / 启动脚本）
"%PY%" -X utf8 tools\assemble_dist.py
if errorlevel 1 (echo   组装失败 & pause & exit /b 1)

echo.
echo   ============================================
echo    打包完成
echo   ============================================
echo    产物: %~dp0dist\ZhaJinWenQianCake\
echo    发给用户后，双击里面的「启动.bat」即可。
echo.
pause
