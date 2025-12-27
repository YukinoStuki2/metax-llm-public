#!/usr/bin/env bash
# 一个最小启动脚本，用于在 Docker 外复现 Dockerfile 的步骤：
# 1) 设置环境变量默认值
# 2) 安装 requirements.txt
# 3) 通过 download_model.py 下载模型
# 4) 启动 uvicorn serve:app

set -euo pipefail

cd "$(dirname "$0")"

say() { echo "[run_model] $*"; }

# ======================
# 1) 环境变量默认值（与 Dockerfile 保持一致）
# ======================
export OMP_NUM_THREADS="4"

# 注意：这里用显式赋值（不继承既有环境变量），避免共享机器上预设的环境变量
# 意外覆盖本脚本的默认行为。
export MODEL_ID="YukinoStuki/Qwen3-4B-Plus-LLM"
export MODEL_REVISION="master"

# 模型下载 token（可选）：保留可继承性，避免清空已有凭证。
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

# Dockerfile 下载到 ./model，并设置运行时 MODEL_DIR=./model/$MODEL_ID
export MODEL_DIR="./model/$MODEL_ID"

# 强烈建议优先使用 vLLM（与 Dockerfile 一致）
export USE_VLLM="true"
export MAX_NEW_TOKENS="32"
export MAX_NEW_TOKENS_CODE="192"

# serve.py 运行时参数（与 Dockerfile 保持一致）
export BATCH_MODE="1"
export BATCH_CONCURRENCY="320"
export TEMPERATURE="0.0"
export TOP_P="1.0"
export TOP_K="1"
export GPU_MEMORY_UTILIZATION="0.90"
export DTYPE="float16"
export TRANSFORMERS_DTYPE="float16"
export DEBUG_NET="0"

# ======================
# 2) Python 环境 + 安装依赖
# ======================
# 使用系统 python，便于在云主机复用预装的 vLLM/torch。
# 这与 Dockerfile 的行为一致（直接往镜像环境里 pip install）。
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

say "Using python: $PYTHON_BIN"

say "Installing requirements.txt"
"$PYTHON_BIN" -m pip install --no-cache-dir -r requirements.txt

# ======================
# 3) 下载模型（与 Dockerfile 的 RUN python download_model.py ... 保持一致）
# ======================
say "Downloading model: $MODEL_ID (revision=$MODEL_REVISION)"
mkdir -p ./model
"$PYTHON_BIN" download_model.py \
  --model_name "$MODEL_ID" \
  --cache_dir ./model \
  --revision "$MODEL_REVISION"

# ======================
# 4) 启动服务（与 Dockerfile 的 CMD 保持一致）
# ======================
say "Starting server on 0.0.0.0:8000"
exec "$PYTHON_BIN" -m uvicorn serve:app --host 0.0.0.0 --port 8000
