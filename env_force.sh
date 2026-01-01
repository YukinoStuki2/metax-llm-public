#!/usr/bin/env bash
# 强制导入一套“干净”的运行参数。
# 用法（必须 source，否则不会影响当前 shell）：
#   source ./env_force.sh
# 然后：
#   ./run_model.sh
#
# 说明：该脚本会覆盖你当前 shell 里的同名环境变量。

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "[env_force] 请用 source 执行：source ./env_force.sh" >&2
  echo "[env_force] 直接运行该脚本不会把变量带到当前 shell。" >&2
  exit 2
fi

say() { echo "[env_force] $*"; }

# ======================
# 与 Dockerfile / run_model.sh 对齐的默认参数（强制覆盖）
# ======================

export OMP_NUM_THREADS="4"

# 模型下载/加载
export MODEL_ID="YukinoStuki/Qwen3-0.6B-Plus-LLM"
export MODEL_REVISION="master"
export MODEL_DIR="./model/$MODEL_ID"

# Speculative Decoding（可选）
export ENABLE_SPECULATIVE_DECODING="0"
export SPEC_METHOD="ngram"
export SPEC_NUM_SPECULATIVE_TOKENS="6"
export SPEC_NGRAM_LOOKUP_MAX="8"
export SPEC_NGRAM_LOOKUP_MIN="1"
export SPEC_DRAFT_MODEL_ID=""
export SPEC_DRAFT_MODEL_REVISION="master"
export SPEC_DRAFT_OPTIONAL="1"

# 模型下载 token（可选；为空表示匿名下载）
: "${MODELSCOPE_API_TOKEN:=}"
export MODELSCOPE_API_TOKEN

# serve.py 运行时参数
export USE_VLLM="true"
export DISABLE_TOKEN_ROUTING="0"
export MAX_NEW_TOKENS="32"
export MAX_NEW_TOKENS_CODE="96"
export MAX_NEW_TOKENS_CODE_HARD="192"
export MAX_NEW_TOKENS_CODE_SOFT="48"
export HARD_CODE_MIN_HITS="1"
export LONG_ANSWER_ENABLE_DEFAULT="1"
export LONG_ANSWER_MIN_HITS="1"

# 解码稳定性：抑制复读
export REPETITION_PENALTY="1.05"
export FREQUENCY_PENALTY="0.1"

# 输出后处理：非代码题裁剪示例段、限制句子数（提高 Rouge 稳定性）
export OUTPUT_TRIM_EXAMPLES="1"
export OUTPUT_MAX_SENTENCES="6"

# vLLM 停止条件（与 Dockerfile 对齐）
export STOP_STRINGS="<|im_end|>,<|endoftext|>"
export STOP_ON_DOUBLE_NEWLINE="0"

export BATCH_MODE="1"
export BATCH_CONCURRENCY="512"
export TEMPERATURE="0.0"
export TOP_P="1.0"
export TOP_K="1"
export GPU_MEMORY_UTILIZATION="0.97"
export DTYPE="float16"
export TRANSFORMERS_DTYPE="float16"

# vLLM 吞吐/量化
export ENABLE_PREFIX_CACHING="1"
export VLLM_QUANTIZATION=""
export VLLM_LOAD_FORMAT="auto"
export VLLM_ENFORCE_EAGER="0"
export VLLM_MAX_NUM_SEQS="1024"
export VLLM_MAX_NUM_BATCHED_TOKENS=""
export VLLM_COMPILATION_CONFIG=""

# 限制最大上下文，避免按超长 config 分配 KV cache
export MAX_MODEL_LEN="1024"

# 可选：用本地数据集抽样预热（默认关闭）
export WARMUP_DATA_PATH="./data.jsonl"
export WARMUP_NUM_SAMPLES="64"
export WARMUP_REPEAT="1"

export DEBUG_NET="0"

say "Environment loaded (forced). MODEL_ID=$MODEL_ID MODEL_DIR=$MODEL_DIR"
