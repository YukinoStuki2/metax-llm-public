---
title: 本地评测（eval_local.py）
sidebar_position: 40
---

`eval_local.py` 用于本地复现评测打分方式（RougeL-F1 + jieba 分词），并输出吞吐/耗时统计。

## 基本用法

默认调用本机：

```bash
python3 eval_local.py --endpoint http://127.0.0.1:8000/predict --health http://127.0.0.1:8000/
```

指定评测集（需要 `basic.docx/plus.docx` 在脚本旁或传绝对路径）：

```bash
python3 eval_local.py --which basic --basic_docx /path/to/basic.docx
python3 eval_local.py --which bonus --bonus_docx /path/to/plus.docx
```

## Batch 评测（推荐与线上一致）

`--batch` 会一次性把所有问题作为 `prompt=list[str]` 发到 `/predict`，更贴近评测机的 batch 行为：

```bash
python3 eval_local.py --which bonus --batch --overwrite_jsonl
```

## 关键参数

- `--timeout`：单次请求超时（默认 300s）
- `--sleep`：请求间隔（默认 0）
- `--strip_q_suffix`：额外计算一个“清理题目重复后缀”的分数（仅用于诊断/对比）
- `--model_dir_for_tokenizer`：加载 tokenizer 用于 token 计数（缺省时 token 统计为 0）
- `--save_jsonl` / `--overwrite_jsonl`：保存每题细节到 jsonl（便于查错）

## 输出指标（auto_tune 会解析）

`auto_tune.py` 主要解析如下行：

- `Accuracy (RougeL-F1 mean, RAW): ...`
- `Throughput RAW: answer_tokens/s=..., (prompt+answer)_tokens/s=...`
- `Total time: ...s`

因此如果你改了 eval_local 的输出格式，要同步改 auto_tune 的正则解析。
