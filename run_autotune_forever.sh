#!/usr/bin/env bash
# 无 root 的“自恢复”运行方式：auto_tune 异常退出就自动重启。
# 注意：如果机器/容器被杀（SIGKILL/关机），本脚本也无法存活；这种场景需要平台级守护或 systemd。

set -euo pipefail
cd "$(dirname "$0")"

: "${RESTART_DELAY:=10}"

while true; do
  echo "[watchdog] starting auto_tune at $(date)"
  ./auto_tune.sh
  code=$?
  echo "[watchdog] auto_tune exited code=$code at $(date)"
  sleep "$RESTART_DELAY"
done
