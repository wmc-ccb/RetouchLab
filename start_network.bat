@echo off
chcp 65001 >nul
rem ============================================================
rem  ZhaJinWenQianCake 启动（局域网 + 公网）
rem
rem  局域网: 手机/平板同 Wi-Fi 打开  http://192.168.5.6:8765
rem  公网:   http://frp2.ccszxc.xin:64649  （隧道跑在 NAS 上，
rem          已设 cron @reboot 自恢复，本机无需任何隧道工具）
rem
rem  访问保护: 已关闭（任何人拿到地址就能用，不需要口令）
rem  想重新打开口令，把下面 RETOUCHLAB_PASSWD 改成想要的字符串，
rem  例如 set RETOUCHLAB_PASSWD=retouch2026
rem ============================================================

set PYTHON=C:\Users\Rana\AppData\Local\Programs\Python\Python313\python.exe
set ROOT=C:\Users\Rana\ZhaJinWenQianCake

echo.
echo  ============================================================
echo   ZhaJinWenQianCake
echo   ------------------------------------------------------------
echo   本机:     http://127.0.0.1:8765
echo   局域网:   http://192.168.5.6:8765
echo   公网:     http://frp2.ccszxc.xin:64649
echo   访问保护: 关闭（无需口令）
echo  ============================================================
echo.

cd /d "%ROOT%"
"%PYTHON%" -m uvicorn server.app:app --host 0.0.0.0 --port 8765
