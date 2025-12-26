#!/usr/bin/env bash
# Convenience script to mimic the Dockerfile steps outside of Docker.
# It installs deps, downloads the merged model from ModelScope, and starts the FastAPI server.

set -euo pipefail

# ----- Env defaults (can be overridden before calling this script) -----
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MODEL_ID=${MODEL_ID:-YukinoStuki/Qwen3-4B-Plus-Merged}
export MODEL_REVISION=${MODEL_REVISION:-master}
export MODELSCOPE_API_TOKEN=${MODELSCOPE_API_TOKEN:-}
export MODEL_DIR=${MODEL_DIR:-"$(pwd)/model/$MODEL_ID"}
export USE_VLLM=${USE_VLLM:-auto}
export MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-48}

# Keep serve.py knobs aligned with Dockerfile defaults
export BATCH_MODE=${BATCH_MODE:-0}
export BATCH_CONCURRENCY=${BATCH_CONCURRENCY:-16}
export TEMPERATURE=${TEMPERATURE:-0.0}
export TOP_P=${TOP_P:-1.0}
export TOP_K=${TOP_K:-1}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
export DTYPE=${DTYPE:-float16}
export TRANSFORMERS_DTYPE=${TRANSFORMERS_DTYPE:-float16}
export DEBUG_NET=${DEBUG_NET:-0}

# ----- Python / pip selection (prefer repo venv) -----
PYTHON=${PYTHON:-"$(pwd)/.venv/bin/python"}
PIP=${PIP:-"$(pwd)/.venv/bin/pip"}

if [[ ! -x "$PYTHON" ]]; then
  echo "[run_model] .venv not found; creating venv in $(pwd)/.venv"
  PYTHON_BOOTSTRAP=${PYTHON_FALLBACK:-python3}
  if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
    echo "[run_model] ERROR: python3 not found. Please install Python 3.10+ first."
    exit 1
  fi
  "$PYTHON_BOOTSTRAP" -m venv .venv
  PYTHON="$(pwd)/.venv/bin/python"
  PIP="$(pwd)/.venv/bin/pip"
fi

if [[ ! -x "$PIP" ]]; then
  echo "[run_model] ERROR: pip not found in venv. Try reinstalling python3-venv / ensurepip."
  exit 1
fi

echo "[run_model] Using python: $PYTHON"
echo "[run_model] Using pip:    $PIP"

have_py_module() {
  local mod="$1"
  "$PYTHON" -c "import $mod" >/dev/null 2>&1
}

pip_install() {
  # shellcheck disable=SC2068
  "$PIP" install --no-cache-dir $@
}

# ----- Install Python dependencies -----
# Always install declared deps first.
pip_install -r requirements.txt

# Extra runtime deps for bare-metal servers (Docker base image already ships these).
# You can override installation behavior via env:
#   TORCH_INDEX_URL=...   (e.g. https://download.pytorch.org/whl/cpu)
#   TORCH_SPEC=...        (e.g. "torch==2.6.0" or "torch")
#   VLLM_SPEC=...         (e.g. "vllm==0.6.6" or "vllm")

TORCH_SPEC=${TORCH_SPEC:-torch}
VLLM_SPEC=${VLLM_SPEC:-vllm}

if ! have_py_module torch; then
  echo "[run_model] torch missing; installing..."
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    pip_install --index-url "$TORCH_INDEX_URL" "$TORCH_SPEC"
  else
    # Conservative default: CPU wheels (works on most servers without CUDA toolchain).
    pip_install --index-url https://download.pytorch.org/whl/cpu "$TORCH_SPEC"
  fi
fi

if ! have_py_module torch; then
  echo "[run_model] ERROR: torch installation failed or torch still not importable."
  echo "[run_model] Hint: set TORCH_INDEX_URL to a CUDA-compatible wheel index for GPU servers."
  exit 1
fi

# vLLM is optional unless USE_VLLM=true. For USE_VLLM=auto we try to install, but will fall back gracefully.
if [[ "${USE_VLLM}" != "false" ]]; then
  if ! have_py_module vllm; then
    echo "[run_model] vllm missing; attempting install (optional unless USE_VLLM=true)..."
    set +e
    pip_install "$VLLM_SPEC"
    vllm_rc=$?
    set -e

    if [[ $vllm_rc -ne 0 ]] || ! have_py_module vllm; then
      if [[ "${USE_VLLM}" == "true" ]]; then
        echo "[run_model] ERROR: USE_VLLM=true but vllm install/import failed."
        echo "[run_model] Hint: on many machines vllm requires a matching GPU/CUDA stack; try USE_VLLM=false."
        exit 1
      fi
      echo "[run_model] vllm not available; continuing with transformers backend (USE_VLLM=false)."
      export USE_VLLM=false
    fi
  fi
fi

# ----- Download merged model from ModelScope -----
$PYTHON download_model.py \
  --model_name "$MODEL_ID" \
  --cache_dir "$(pwd)/model" \
  --revision "$MODEL_REVISION"

# ----- Launch server -----
exec $PYTHON -m uvicorn serve:app --host 0.0.0.0 --port 8000
