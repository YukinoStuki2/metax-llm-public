---
title: AWQ 量化（quantize_awq_llmcompressor.py）
sidebar_position: 90
---

仓库提供 `quantize_awq_llmcompressor.py`：一次性脚本，用 AutoAWQ 做 4bit 量化并导出。

> 量化通常用于吞吐/显存优化；是否适合评测要以实际分数与稳定性为准。

## 依赖

建议在单独环境安装：`requirements-quantize-awq.txt`。

## 准备校准集

脚本默认使用仓库内 `calib_8192.jsonl`。

你也可以从 `data.jsonl` 抽样生成：

```bash
N=8192 MAX_LEN=2048 OUT_JSONL=calib_8192.jsonl OUT_TXT=calib_8192.txt python3 sample_calib_from_data.py
```

## 运行示例

```bash
python3 quantize_awq_llmcompressor.py \
  --model_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM \
  --output_dir model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ
```

常用覆盖（用环境变量）：

- `AWQ_CALIB_JSONL`：指定校准 jsonl 路径
- `AWQ_MODEL_DIR` / `AWQ_OUTPUT_DIR`：覆盖输入/输出目录

## 常见问题

- **OOM**：优先降低校准阶段的 `AWQ_MAX_SEQ_LEN` 或 `AWQ_NUM_CALIB`（脚本内常量）。
- **量化后掉分**：提高校准样本数量，或让校准文本分布更贴近线上题目。
