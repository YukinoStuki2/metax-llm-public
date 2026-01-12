---
title: 评测脚本（judge.sh）
sidebar_position: 110
---

`judge.sh` 是一个薄封装：调用 `eval_local.py`，用于快速跑一次本地评测。

---

## 1) 脚本功能概述

### 作用与定位

- 用途：在本地快速跑一轮评测，验证服务端可用性与相对分数变化。
- 评测口径：本质等价于执行一次带固定参数的 `eval_local.py`。

---

## 2) 参数与环境变量详解

脚本内部使用两个环境变量作为默认值：

- `MODEL_DIR`：传给 `eval_local.py --model_dir_for_tokenizer`，用于加载 tokenizer 做 token 统计；默认 `./model/merged`。
- `WHICH`：传给 `eval_local.py --which`，用于选择评测子集；默认 `bonus`。

覆盖方式示例：

```bash
MODEL_DIR=./model/YukinoStuki/Qwen3-4B-Plus-LLM WHICH=basic ./judge.sh
```

---

## 3) 代码实现详解

脚本内容（节选）：

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

执行流程：

1. 读取 `MODEL_DIR` 与 `WHICH`（未设置则使用默认值）。
2. 调用 `eval_local.py`：固定开启 `--batch` 与 `--overwrite_jsonl`。

最短路径回归（两步）：

1. 启动服务：`./run_model.sh`
2. 运行评测：`./judge.sh`

---

## 4) 常见问题

### Q1：如何更贴近线上行为？

可直接调用评测脚本（跳过封装层）：

```bash
python3 eval_local.py --which bonus --batch --overwrite_jsonl
```