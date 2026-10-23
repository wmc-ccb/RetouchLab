#!/usr/bin/env bash
# 在 NAS 上以当前用户起独立 frpc，把本机(Windows 192.168.5.6:8765)暴露到公网
set -u
DIR="$HOME/retouchlab"
mkdir -p "$DIR"
CFG="$DIR/frpc.toml"

cat > "$CFG" <<'EOF'
# RetouchLab 公网隧道（独立实例，不影响系统 frpc）
serverAddr = "frp2.ccszxc.xin"
serverPort = 64640

auth.method = "token"
auth.token = "945052"

log.to = "/home/wmc258/retouchlab/frpc.log"
log.level = "info"
log.maxDays = 1

[[proxies]]
name = "retouchlab"
type = "tcp"
localIP = "192.168.5.6"
localPort = 8765
remotePort = 64649
EOF

echo "=== 配置 ==="
cat "$CFG"

FRPC=/vol1/@appcenter/frpc/app/frpc
echo "=== 前台试跑 8 秒看握手 ==="
timeout 8 "$FRPC" -c "$CFG" 2>&1 | tail -12
echo "=== 试跑结束 ==="
