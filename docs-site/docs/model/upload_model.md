---
title: 上传模型到 ModelScope（upload_model.py）
sidebar_position: 80
---

`upload_model.py` 用于把一个本地“模型目录（HF 格式）”上传到 ModelScope。

## 用法

```bash
export MODELSCOPE_API_TOKEN=...  # 或 MODELSCOPE_TOKEN
python3 upload_model.py \
  --repo-id "YukinoStuki/Qwen3-4B-Plus-LLM-AWQ" \
  --model-dir "model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ"
```

参数也可用环境变量覆盖：

- `REPO_ID`
- `MODEL_DIR`

## 目录校验

脚本会做轻量校验（例如 `config.json/tokenizer_config.json` 是否存在），避免把空目录或不完整目录误上传。
