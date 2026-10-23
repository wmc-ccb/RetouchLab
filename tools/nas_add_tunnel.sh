#!/usr/bin/env bash
# 给 NAS 的 frpc 加 RetouchLab 隧道（幂等：已存在则只报告）
set -u
CFG=/vol1/@appdata/frpc/config/frpc.toml
BAK="$CFG.bak-retouchlab-$(date +%Y%m%d-%H%M%S)"

echo "=== 当前配置 ==="
tail -5 "$CFG"

if grep -q '^\[retouchlab\]' "$CFG"; then
  echo "RESULT: 已存在 [retouchlab]，不重复添加"
  exit 0
fi

cp "$CFG" "$BAK" && echo "备份: $BAK"

# remote_port 64649（现有 64642-64648，接续）
cat >> "$CFG" <<'EOF'

[retouchlab]
type = tcp
local_ip = 192.168.5.6
local_port = 8765
remote_port = 64649
EOF

echo "=== 追加后 ==="
tail -8 "$CFG"

# 热重载（frpc 的 webServer 在 127.0.0.1:7400）
if curl -s -o /dev/null -w '%{http_code}' -X GET http://127.0.0.1:7400/api/reload 2>/dev/null | grep -q 200; then
  echo "热重载: OK"
elif curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:7400/api/reload 2>/dev/null | grep -q 200; then
  echo "热重载(POST): OK"
else
  echo "热重载接口未响应，需重启 frpc"
fi

sleep 2
echo "=== 隧道状态 ==="
curl -s http://127.0.0.1:7400/api/status 2>/dev/null | head -c 600
echo
