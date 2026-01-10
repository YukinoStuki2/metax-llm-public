---
title: 模型下载（download_model.py）
sidebar_position: 60
---

`download_model.py` 用于在 build 阶段从 ModelScope 下载模型到本地目录（评测机 run 阶段断网，因此必须在 build 完成）。

## 用法

```bash
python3 download_model.py \
  --model_name "YukinoStuki/Qwen2.5-0.5B-Plus-LLM" \
  --revision master \
  --cache_dir ./model
```

可选：使用 Token（私有模型/更稳的下载）：

- 环境变量：`MODELSCOPE_API_TOKEN`
- 或参数：`--token ...`

## Speculative Draft 模型（可选）

脚本支持额外下载一个 draft 模型（用于 speculative decoding）：

- `--draft_model_name` / `--draft_revision`
- `--draft_optional`：若开启，draft 下载失败不会让整体失败（适合“可选加速”场景）。

> 评测机不要求 speculative decoding；如果启用，建议先确认平台的 vLLM/插件兼容性。
