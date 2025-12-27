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
: "${OMP_NUM_THREADS:=4}"
export OMP_NUM_THREADS

: "${MODEL_ID:=YukinoStuki/Qwen3-4B-Plus-LLM}"
export MODEL_ID
: "${MODEL_REVISION:=master}"
export MODEL_REVISION

# 模型下载 token（可选）：不在脚本内写死默认值；若用户已设置则直接使用。
: "${MODELSCOPE_API_TOKEN:=}"
export MODELSCOPE_API_TOKEN

# Dockerfile 下载到 ./model，并设置运行时 MODEL_DIR=./model/$MODEL_ID
if [[ -z "${MODEL_DIR:-}" ]]; then
  MODEL_DIR="./model/$MODEL_ID"
fi
export MODEL_DIR

# 强烈建议优先使用 vLLM（与 Dockerfile 一致）
: "${USE_VLLM:=true}"
export USE_VLLM
: "${MAX_NEW_TOKENS:=32}"
export MAX_NEW_TOKENS
: "${MAX_NEW_TOKENS_CODE:=192}"
export MAX_NEW_TOKENS_CODE

# serve.py 运行时参数（与 Dockerfile 保持一致）
: "${BATCH_MODE:=1}"
export BATCH_MODE
: "${BATCH_CONCURRENCY:=358}"
export BATCH_CONCURRENCY
: "${TEMPERATURE:=0.0}"
export TEMPERATURE
: "${TOP_P:=1.0}"
export TOP_P
: "${TOP_K:=1}"
export TOP_K
: "${GPU_MEMORY_UTILIZATION:=0.90}"
export GPU_MEMORY_UTILIZATION
: "${DTYPE:=float16}"
export DTYPE
: "${TRANSFORMERS_DTYPE:=float16}"
export TRANSFORMERS_DTYPE

# vLLM 吞吐/量化（可选）：默认不启用量化；用户可在运行前 export 覆盖。
: "${ENABLE_PREFIX_CACHING:=1}"
export ENABLE_PREFIX_CACHING
: "${VLLM_QUANTIZATION:=}"
export VLLM_QUANTIZATION
: "${VLLM_LOAD_FORMAT:=}"
export VLLM_LOAD_FORMAT
: "${VLLM_MAX_NUM_SEQS:=}"
export VLLM_MAX_NUM_SEQS
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=}"
export VLLM_MAX_NUM_BATCHED_TOKENS
: "${VLLM_COMPILATION_CONFIG:=}"
export VLLM_COMPILATION_CONFIG
: "${MAX_MODEL_LEN:=}"
export MAX_MODEL_LEN

: "${DEBUG_NET:=0}"
export DEBUG_NET

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
