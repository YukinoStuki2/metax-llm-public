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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

export MODEL_ID="${MODEL_ID:-YukinoStuki/Qwen3-4B-Plus-Merged}"
export MODEL_REVISION="${MODEL_REVISION:-master}"

# Model download token (optional)
export MODELSCOPE_API_TOKEN="${MODELSCOPE_API_TOKEN:-}"

# Dockerfile downloads to ./model and sets runtime MODEL_DIR=./model/$MODEL_ID
export MODEL_DIR="${MODEL_DIR:-./model/$MODEL_ID}"

# Strongly prefer vLLM (as Dockerfile)
export USE_VLLM="${USE_VLLM:-true}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"

# serve.py runtime knobs (match Dockerfile)
export BATCH_MODE="${BATCH_MODE:-0}"
export BATCH_CONCURRENCY="${BATCH_CONCURRENCY:-16}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:-1}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
export DTYPE="${DTYPE:-float16}"
export TRANSFORMERS_DTYPE="${TRANSFORMERS_DTYPE:-float16}"
export DEBUG_NET="${DEBUG_NET:-0}"

# ======================
# 2) Python venv + install deps
# ======================
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x ./.venv/bin/python ]]; then
  say "Creating venv: ./.venv"
  "$PYTHON_BIN" -m venv .venv
fi

PYTHON="$(pwd)/.venv/bin/python"

say "Installing requirements.txt"
"$PYTHON" -m pip install --no-cache-dir -r requirements.txt

# ======================
# 3) Download model (match Dockerfile RUN python download_model.py ...)
# ======================
say "Downloading model: $MODEL_ID (revision=$MODEL_REVISION)"
mkdir -p ./model
"$PYTHON" download_model.py \
  --model_name "$MODEL_ID" \
  --cache_dir ./model \
  --revision "$MODEL_REVISION"

# ======================
# 4) Run server (match Dockerfile CMD)
# ======================
say "Starting server on 0.0.0.0:8000"
exec "$PYTHON" -m uvicorn serve:app --host 0.0.0.0 --port 8000
