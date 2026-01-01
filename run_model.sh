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

: "${MODEL_ID:=YukinoStuki/Qwen3-0.6B-Plus-LLM}"
: "${MODEL_REVISION:=master}"
export MODEL_ID MODEL_REVISION

# ======================
# Speculative Decoding（可选）
# ======================
# 默认关闭；需要时可在执行脚本前覆盖这些环境变量。
# ngram 方法：不需要 draft 模型，基于 prompt 内 n-gram 模式预测，零额外成本
: "${ENABLE_SPECULATIVE_DECODING:=0}"
: "${SPEC_METHOD:=ngram}"
: "${SPEC_NUM_SPECULATIVE_TOKENS:=6}"
: "${SPEC_NGRAM_LOOKUP_MAX:=8}"
: "${SPEC_NGRAM_LOOKUP_MIN:=1}"
: "${SPEC_DRAFT_MODEL_ID:=}"
: "${SPEC_DRAFT_MODEL_REVISION:=master}"
: "${SPEC_DRAFT_OPTIONAL:=1}"
export ENABLE_SPECULATIVE_DECODING SPEC_DRAFT_MODEL_ID SPEC_DRAFT_MODEL_REVISION \
  SPEC_NUM_SPECULATIVE_TOKENS SPEC_METHOD SPEC_NGRAM_LOOKUP_MAX SPEC_NGRAM_LOOKUP_MIN SPEC_DRAFT_OPTIONAL

# 模型下载 token（可选）：允许在执行脚本前注入，默认空=匿名下载。
: "${MODELSCOPE_API_TOKEN:=}"
export MODELSCOPE_API_TOKEN

# Dockerfile 下载到 ./model，并设置运行时 MODEL_DIR=./model/$MODEL_ID
: "${MODEL_DIR:=./model/$MODEL_ID}"
export MODEL_DIR

# 强烈建议优先使用 vLLM（与 Dockerfile 一致）
: "${USE_VLLM:=true}"
: "${DISABLE_TOKEN_ROUTING:=0}"
export USE_VLLM DISABLE_TOKEN_ROUTING
: "${MAX_NEW_TOKENS:=32}"
: "${MAX_NEW_TOKENS_CODE:=96}"
: "${MAX_NEW_TOKENS_CODE_HARD:=192}"
: "${MAX_NEW_TOKENS_CODE_SOFT:=48}"
: "${HARD_CODE_MIN_HITS:=1}"
export MAX_NEW_TOKENS MAX_NEW_TOKENS_CODE MAX_NEW_TOKENS_CODE_HARD MAX_NEW_TOKENS_CODE_SOFT
export HARD_CODE_MIN_HITS
: "${LONG_ANSWER_ENABLE_DEFAULT:=1}"
: "${LONG_ANSWER_MIN_HITS:=1}"
export LONG_ANSWER_ENABLE_DEFAULT LONG_ANSWER_MIN_HITS

# 解码稳定性：抑制复读（允许外部覆盖）
: "${REPETITION_PENALTY:=1.05}"
: "${FREQUENCY_PENALTY:=0.1}"
export REPETITION_PENALTY FREQUENCY_PENALTY

# 输出后处理：非代码题裁剪示例段、限制句子数（允许外部覆盖）
: "${OUTPUT_TRIM_EXAMPLES:=1}"
: "${OUTPUT_MAX_SENTENCES:=6}"
export OUTPUT_TRIM_EXAMPLES OUTPUT_MAX_SENTENCES

# vLLM 停止条件（与 Dockerfile 对齐；允许外部覆盖）
: "${STOP_STRINGS:=<|im_end|>,<|endoftext|>}"
: "${STOP_ON_DOUBLE_NEWLINE:=0}"
export STOP_STRINGS STOP_ON_DOUBLE_NEWLINE

# serve.py 运行时参数（与 Dockerfile 保持一致）
: "${BATCH_MODE:=1}"
: "${BATCH_CONCURRENCY:=512}"
: "${TEMPERATURE:=0.0}"
: "${TOP_P:=1.0}"
: "${TOP_K:=1}"
: "${GPU_MEMORY_UTILIZATION:=0.97}"
: "${DTYPE:=float16}"
: "${TRANSFORMERS_DTYPE:=float16}"
export BATCH_MODE BATCH_CONCURRENCY TEMPERATURE TOP_P TOP_K GPU_MEMORY_UTILIZATION DTYPE TRANSFORMERS_DTYPE

# vLLM 吞吐/量化（AWQ）
: "${ENABLE_PREFIX_CACHING:=1}"
: "${VLLM_QUANTIZATION:=}"
: "${VLLM_LOAD_FORMAT:=auto}"
export ENABLE_PREFIX_CACHING VLLM_QUANTIZATION VLLM_LOAD_FORMAT

# MetaX 上如果强制 enforce_eager 会禁用 cudagraph，吞吐可能下降。
# 默认不强制 eager；若遇到平台兼容问题再设为 1。
: "${VLLM_ENFORCE_EAGER:=0}"
export VLLM_ENFORCE_EAGER

# 可选：不设置表示交给 vLLM 自行决定
: "${VLLM_MAX_NUM_SEQS:=1024}"
: "${VLLM_MAX_NUM_BATCHED_TOKENS:=}"
: "${VLLM_COMPILATION_CONFIG:=}"
export VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS VLLM_COMPILATION_CONFIG

# Qwen3 系列模型 config 里可能带超长上下文（如 262144），会导致 KV cache 按超长分配，并发很低。
# 评测题通常不需要这么长，上限过大会拖慢吞吐；这里默认限制到一个更实际的值。
: "${MAX_MODEL_LEN:=1024}"
export MAX_MODEL_LEN

# 可选：用本地数据集抽样预热（默认关闭）
: "${WARMUP_DATA_PATH:=./data.jsonl}"
: "${WARMUP_NUM_SAMPLES:=64}"
: "${WARMUP_REPEAT:=1}"
export WARMUP_DATA_PATH WARMUP_NUM_SAMPLES WARMUP_REPEAT

: "${DEBUG_NET:=0}"
export DEBUG_NET

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
  --revision "$MODEL_REVISION" \
  --token "$MODELSCOPE_API_TOKEN" \
  --draft_model_name "$SPEC_DRAFT_MODEL_ID" \
  --draft_revision "$SPEC_DRAFT_MODEL_REVISION" \
  --draft_optional

# ======================
# 4) 启动服务（与 Dockerfile 的 CMD 保持一致）
# ======================
say "Starting server on 0.0.0.0:8000"
exec "$PYTHON_BIN" -m uvicorn serve:app --host 0.0.0.0 --port 8000
