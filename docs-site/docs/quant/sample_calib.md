---
title: 抽样校准集（sample_calib_from_data.py）
sidebar_position: 100
---

`sample_calib_from_data.py` 从 `data.jsonl` 抽样生成量化校准用的文本集合：

- 输出 jsonl：每行 `{"text":"..."}`
- 输出 txt：便于人工抽查

## 用法

```bash
DATA_JSONL=data.jsonl \
N=512 \
MAX_LEN=512 \
OUT_JSONL=calib_512.jsonl \
OUT_TXT=calib_512.txt \
python3 sample_calib_from_data.py
```

## 采样策略

- 优先抽取 `messages[].role==user` 的内容，兼容 `prompt/instruction/question/input/text` 字段。
- 做归一化与去重，避免相同问题被提示词变体占满样本。
- 最终随机采样（固定 `SEED` 时可复现）。
