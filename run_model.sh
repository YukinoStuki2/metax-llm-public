#!/usr/bin/env bash
# Convenience script to mimic the Dockerfile steps outside of Docker.
# It installs deps, downloads the merged model from ModelScope, and starts the FastAPI server.

set -euo pipefail

# ----- Env defaults (can be overridden before calling this script) -----
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MODEL_ID=${MODEL_ID:-yukinostuki/qwen3-4b-ft-v1}
export MODEL_REVISION=${MODEL_REVISION:-latest}
export MODELSCOPE_API_TOKEN=${MODELSCOPE_API_TOKEN:-}
export MODEL_DIR=${MODEL_DIR:-"$(pwd)/model/$MODEL_ID"}
export USE_VLLM=${USE_VLLM:-true}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-48}

# ----- Python / pip selection (prefer repo venv) -----
PYTHON=${PYTHON:-"$(pwd)/.venv/bin/python"}
PIP=${PIP:-"$(pwd)/.venv/bin/pip"}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  PYTHON=${PYTHON_FALLBACK:-python3}
fi
if ! command -v "$PIP" >/dev/null 2>&1; then
  PIP=${PIP_FALLBACK:-pip3}
fi

echo "[run_model] Using python: $PYTHON"
echo "[run_model] Using pip:    $PIP"

# ----- Install Python dependencies -----
$PIP install --no-cache-dir -r requirements.txt

# ----- Download merged model from ModelScope -----
$PYTHON download_model.py \
  --model_name "$MODEL_ID" \
  --cache_dir "$(pwd)/model" \
  --revision "$MODEL_REVISION"

# ----- Launch server -----
exec $PYTHON -m uvicorn serve:app --host 0.0.0.0 --port 8000
