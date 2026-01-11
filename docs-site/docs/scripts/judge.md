---
title: 评测脚本（judge.sh）
sidebar_position: 110
---

`judge.sh` 是一个很薄的封装：调用 `eval_local.py`，用于快速跑一次本地评测。

当前内容：

```bash
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
```

你可以用它做“最短路径”回归：

1. 启动服务：`./run_model.sh`

2. 运行评测：`./judge.sh`

如果你更希望贴近线上行为，建议直接用：

```bash
python3 eval_local.py --which bonus --batch --overwrite_jsonl
```