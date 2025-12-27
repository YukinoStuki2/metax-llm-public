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

export MODEL_ID="YukinoStuki/Qwen3-4B-Plus-LLM-AWQ"
export MODEL_REVISION="master"

# 模型下载 token（可选）：此脚本不读取本机环境变量，避免不同机器环境不一致。
export MODELSCOPE_API_TOKEN=""

# Dockerfile 下载到 ./model，并设置运行时 MODEL_DIR=./model/$MODEL_ID
export MODEL_DIR="./model/$MODEL_ID"

# 强烈建议优先使用 vLLM（与 Dockerfile 一致）
export USE_VLLM="true"
export MAX_NEW_TOKENS="32"
export MAX_NEW_TOKENS_CODE="192"

# serve.py 运行时参数（与 Dockerfile 保持一致）
export BATCH_MODE="1"
export BATCH_CONCURRENCY="358"
export TEMPERATURE="0.0"
export TOP_P="1.0"
export TOP_K="1"
export GPU_MEMORY_UTILIZATION="0.90"
export DTYPE="float16"
export TRANSFORMERS_DTYPE="float16"

# vLLM 吞吐/量化（AWQ）
export ENABLE_PREFIX_CACHING="1"
export VLLM_QUANTIZATION="awq"
export VLLM_LOAD_FORMAT="auto"

# MetaX 上如果强制 enforce_eager 会禁用 cudagraph，吞吐可能下降。
# 默认不强制 eager；若遇到平台兼容问题再设为 1。
export VLLM_ENFORCE_EAGER="0"

# 可选：不设置表示交给 vLLM 自行决定
export VLLM_MAX_NUM_SEQS=""
export VLLM_MAX_NUM_BATCHED_TOKENS=""
export VLLM_COMPILATION_CONFIG=""

# Qwen3 系列模型 config 里可能带超长上下文（如 262144），会导致 KV cache 按超长分配，并发很低。
# 评测题通常不需要这么长，上限过大会拖慢吞吐；这里默认限制到一个更实际的值。
export MAX_MODEL_LEN="38400"

export DEBUG_NET="0"

# ======================
# 2) Python 环境 + 安装依赖
# ======================
# 使用系统 python，便于在云主机复用预装的 vLLM/torch。
# 这与 Dockerfile 的行为一致（直接往镜像环境里 pip install）。
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
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
