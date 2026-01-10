#!/usr/bin/env bash
set -euo pipefail

# 快速本地评测封装：优先贴近线上 batch 行为。
MODEL_DIR=${MODEL_DIR:-./model/merged}
WHICH=${WHICH:-bonus}

python3 eval_local.py \
	--which "$WHICH" \
	--batch \
	--overwrite_jsonl \
	--model_dir_for_tokenizer "$MODEL_DIR"
