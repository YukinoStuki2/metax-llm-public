---
title: 融合 LoRA Adapter（merge_adapter.py）
sidebar_position: 70
---

`merge_adapter.py` 用于：

1) 从 ModelScope 下载 base model
2) `git clone` 一个 adapter 仓库（可选 checkout 指定 ref）
3) 读取 adapter 权重（支持 Git LFS）
4) 用 `peft` 将 LoRA merge 到 base model，导出一个“完整模型目录”

## 典型用法

```bash
python3 merge_adapter.py \
  --base_model Qwen/Qwen3-4B \
  --cache_dir ./model \
  --adapter_repo_url git@...:your-adapter.git \
  --output_dir ./merged
```

## adapter_config.json 处理

部分 adapter 仓库可能只有 `adapter_model.safetensors`，缺少 `adapter_config.json`。

此时需要你提供：

- `ADAPTER_CONFIG_JSON`：直接给 JSON 字符串
- 或 `ADAPTER_CONFIG_PATH`：给一个本地 json 文件路径

否则脚本会报错并退出（因为 `peft` merge 需要 config）。

## Git LFS

若权重文件是 Git LFS pointer，脚本会尝试执行：

- `git lfs install --local`
- `git lfs pull`

如果环境缺少 git-lfs 或 LFS 拉取失败，会明确报错提示。

## 注意事项

- 融合过程依赖 `torch/transformers/peft`，会吃较多内存与时间，建议在非评测机环境单独执行。
- 评测交付通常不需要在 run 阶段融合；更推荐在 build 阶段直接下载“已融合”的模型目录。
