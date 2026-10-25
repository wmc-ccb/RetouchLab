@echo off
chcp 65001 >nul
title ZhaJinWenQianCake - 安装环境
setlocal
cd /d "%~dp0"

rem ============================================================
rem  一键安装环境（拿到源码后双击这个，跑一次就行）
rem
rem  做的事：
rem    1. 找 Python（3.11-3.13），没有就提示去装
rem    2. 在 .venv 里建独立环境（不污染系统 Python）
rem    3. 安装全部依赖（约 500MB，一次性）
rem    4. 从本仓库 Release 下载全部模型（980MB）并校验
rem    5. 完成后双击 ZhaJinWenQianCake.bat 启动
rem ============================================================

echo.
echo  ============================================
echo   ZhaJinWenQianCake  环境安装
echo  ============================================
echo.

rem ---- 1. 找 Python ----
set "PY="
for /f "tokens=*" %%i in ('where python 2^>nul') do (
  if not defined PY set "PY=%%i"
)
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY (
  echo  [1/4] 没有找到 Python。
  echo.
  echo  请先安装 Python 3.11 - 3.13：
  echo    https://www.python.org/downloads/
  echo  安装时勾选 "Add python.exe to PATH"，然后重新双击本文件。
  echo.
  pause
  exit /b 1
)
echo  [1/4] Python: %PY%
"%PY%" -c "import sys; sys.exit(0 if (3,11)<=sys.version_info[:2]<=(3,13) else 1)" 2>nul
if errorlevel 1 (
  echo  Python 版本需要 3.11 - 3.13，当前：
  "%PY%" --version
  pause
  exit /b 1
)

rem ---- 2. 建虚拟环境 ----
if not exist ".venv\Scripts\python.exe" (
  echo  [2/4] 创建独立环境 .venv ...
  "%PY%" -m venv .venv
  if errorlevel 1 (
    echo  创建虚拟环境失败。
    pause
    exit /b 1
  )
) else (
  echo  [2/4] 已有 .venv，跳过
)
set "VPY=%~dp0.venv\Scripts\python.exe"

rem ---- 3. 装依赖 ----
echo  [3/4] 安装依赖（约 500MB，几分钟，只做一次）...
"%VPY%" -m pip install -q --upgrade pip
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
  echo  依赖安装失败。检查网络后重新双击本文件（会跳过已完成的步骤）。
  pause
  exit /b 1
)

rem ---- 4. 模型 ----
echo  [4/4] 检查/下载模型（980MB，断点续传）...
"%VPY%" -X utf8 tools\fetch_models.py
if errorlevel 1 (
  echo  模型下载失败。检查网络后重新双击本文件（已下好的不会重下）。
  pause
  exit /b 1
)

echo.
echo  ============================================
echo   全部就绪！
echo   双击 ZhaJinWenQianCake.bat 启动应用
echo  ============================================
echo.
pause
