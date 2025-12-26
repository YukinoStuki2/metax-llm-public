#!/usr/bin/env bash
# Minimal runner that mirrors Dockerfile steps (outside Docker):
# 1) export ENV defaults
# 2) install requirements.txt
# 3) download model via download_model.py
# 4) run uvicorn serve:app

set -euo pipefail

cd "$(dirname "$0")"

say() { echo "[run_model] $*"; }

# ======================
# 1) ENV defaults (match Dockerfile)
# ======================
export OMP_NUM_THREADS="4"

# NOTE: Use explicit assignment (do not inherit existing env) so that
# pre-set env vars on a shared machine won't override this runner.
export MODEL_ID="YukinoStuki/Qwen3-4B-Plus-LLM"
export MODEL_REVISION="master"

# Model download token (optional): keep inheritable to avoid wiping credentials.
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

# Dockerfile downloads to ./model and sets runtime MODEL_DIR=./model/$MODEL_ID
export MODEL_DIR="./model/$MODEL_ID"

# Strongly prefer vLLM (as Dockerfile)
export USE_VLLM="true"
export MAX_NEW_TOKENS="32"

# serve.py runtime knobs (match Dockerfile)
export BATCH_MODE="1"
export BATCH_CONCURRENCY="96"
export TEMPERATURE="0.0"
export TOP_P="1.0"
export TOP_K="1"
export GPU_MEMORY_UTILIZATION="0.90"
export DTYPE="float16"
export TRANSFORMERS_DTYPE="float16"
export DEBUG_NET="0"

# ======================
# 2) Python venv + install deps
# ======================
# Use global/system python to allow reusing preinstalled vLLM/torch on cloud hosts.
# This mirrors Dockerfile behavior (pip install into the image environment).
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

say "Using python: $PYTHON_BIN"

say "Installing requirements.txt"
"$PYTHON_BIN" -m pip install --no-cache-dir -r requirements.txt

# ======================
# 3) Download model (match Dockerfile RUN python download_model.py ...)
# ======================
say "Downloading model: $MODEL_ID (revision=$MODEL_REVISION)"
mkdir -p ./model
"$PYTHON_BIN" download_model.py \
  --model_name "$MODEL_ID" \
  --cache_dir ./model \
  --revision "$MODEL_REVISION"

# ======================
# 4) Run server (match Dockerfile CMD)
# ======================
say "Starting server on 0.0.0.0:8000"
exec "$PYTHON_BIN" -m uvicorn serve:app --host 0.0.0.0 --port 8000
