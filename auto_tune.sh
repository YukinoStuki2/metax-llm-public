#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# 自动加载本地私密配置（若存在）。
# 建议：cp tune_secrets.example.sh tune_secrets.sh 并填入密钥。
if [[ -f "./tune_secrets.sh" ]]; then
  # shellcheck disable=SC1091
  source "./tune_secrets.sh"
fi

# 推荐用法：
# 1) 先清干净默认参数（可选）
#    source ./env_force.sh
# 2) 再启动自动调参
#    ./auto_tune.sh
#
# 你也可以临时覆盖：
#   ACC=0.8810 EVAL_RUNS=5 ./auto_tune.sh

: "${REPO:=.}"
: "${EVAL_RUNS:=5}"
: "${ACC:=0.8800}"
: "${STARTUP_TIMEOUT:=240}"
: "${EVAL_TIMEOUT:=420}"
: "${HEARTBEAT_TRIALS:=10}"
: "${HEARTBEAT_INTERVAL_S:=${TUNE_HEARTBEAT_INTERVAL_S:-0}}"
: "${SEARCH_SPACE_FILE:=${TUNE_SEARCH_SPACE_FILE:-}}"
: "${PORT_BUSY_RETRIES:=${TUNE_PORT_BUSY_RETRIES:-3}}"
: "${PORT_BUSY_WAIT_S:=${TUNE_PORT_BUSY_WAIT_S:-10}}"
: "${PORT_BUSY_KILL:=${TUNE_PORT_BUSY_KILL:-0}}"

PORT_BUSY_KILL_FLAG=()
if [[ "$PORT_BUSY_KILL" == "1" ]]; then
  PORT_BUSY_KILL_FLAG=(--port_busy_kill)
fi

exec python3 auto_tune.py \
  --repo "$REPO" \
  --eval_runs "$EVAL_RUNS" \
  --accuracy_threshold "$ACC" \
  --startup_timeout "$STARTUP_TIMEOUT" \
  --eval_timeout "$EVAL_TIMEOUT" \
  --heartbeat_trials "$HEARTBEAT_TRIALS" \
  --heartbeat_interval_s "$HEARTBEAT_INTERVAL_S" \
  ${SEARCH_SPACE_FILE:+--search_space_file "$SEARCH_SPACE_FILE"} \
  --port_busy_retries "$PORT_BUSY_RETRIES" \
  --port_busy_wait_s "$PORT_BUSY_WAIT_S" \
  "${PORT_BUSY_KILL_FLAG[@]}" \
  --skip_existing
