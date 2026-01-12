---
title: 抽样校准集（sample_calib_from_data.py）
sidebar_position: 100
---

# 校准集抽样脚本详解

## 1) 脚本功能概述

### 1.1 作用与定位

`sample_calib_from_data.py` 用于从项目的训练/题库数据 `data.jsonl` 中抽样出一批“校准文本”，生成给量化（例如 AWQ）使用的校准集文件。

在本项目里，它通常配合 [docs-site/docs/quant/awq.md](../quant/awq.md) 中的量化脚本使用：

- 量化脚本需要一个 jsonl，格式为每行 `{"text": "..."}`。
- 本脚本负责从 `data.jsonl` 提取用户侧文本、去重、再抽样，生成这个校准 jsonl。

### 1.2 输入与输出

**输入：**

- `DATA_JSONL`：源数据 jsonl（默认 `data.jsonl`）
- 若干环境变量控制抽样数量与长度（见第二部分）

**输出：**

- `OUT_JSONL`：校准集 jsonl（默认 `calib_512.jsonl`），每行一个对象：
  ```json
  {"text": "..."}
  ```
- `OUT_TXT`：纯文本（默认 `calib_512.txt`），一行一条，便于人工快速抽查。
- 控制台输出：打印候选总量（去重后）与最终抽样数量。

**会改变什么：**

- 会写入/覆盖 `OUT_JSONL` 与 `OUT_TXT` 两个文件。
- 不会修改 `DATA_JSONL`。

**不改变：**

- 不会修改源数据内容（只读抽样与写出）。

---

## 2) 参数与环境变量详解

这个脚本不使用命令行参数，全部通过环境变量配置。

### 2.1 环境变量

#### `DATA_JSONL`

- **作用**：源数据 jsonl 路径。
- **默认值**：`data.jsonl`

#### `OUT_JSONL`

- **作用**：输出校准集 jsonl 路径。
- **默认值**：`calib_512.jsonl`

#### `OUT_TXT`

- **作用**：输出校准集 txt 路径。
- **默认值**：`calib_512.txt`

#### `N`

- **作用**：抽样条数。
- **默认值**：`512`
- **行为**：若去重后的候选条数小于等于 `N`，则会输出全部候选，不会报错。

#### `SEED`

- **作用**：随机种子。
- **默认值**：`42`
- **影响**：固定 seed 时抽样结果可复现；改变 seed 会得到不同子集。

#### `MIN_LEN`

- **作用**：最短文本长度过滤阈值。
- **默认值**：`6`
- **单位**：字符（不是 token）。

#### `MAX_LEN`

- **作用**：最长文本长度截断阈值。
- **默认值**：`512`
- **单位**：字符（不是 token）。
- **说明**：这里截断只是“预处理”；量化时仍会被 tokenizer 与量化脚本的 `max_seq_len` 做二次截断。

### 2.2 典型用法

生成 512 条、最大 512 字符的校准集：

```bash
DATA_JSONL=data.jsonl \
N=512 \
MAX_LEN=512 \
OUT_JSONL=calib_512.jsonl \
OUT_TXT=calib_512.txt \
python3 sample_calib_from_data.py
```

生成更大规模校准集（例如 8192 条、最大 2048 字符）：

```bash
DATA_JSONL=data.jsonl \
N=8192 \
MAX_LEN=2048 \
OUT_JSONL=calib_8192.jsonl \
OUT_TXT=calib_8192.txt \
python3 sample_calib_from_data.py
```

---

## 3) 代码实现详解

### 3.1 数据读取与容错：`_iter_jsonl(path)`

脚本按行读取 jsonl，并做了较强的容错：

- 会先找到该行第一个 `{`，把它当作 JSON 起点。
- 使用 `json.JSONDecoder().raw_decode()` 解析该行的第一个 JSON 对象。
- 解析失败、无 `{`、或解析结果不是 dict 时都会跳过。

这样做是为了应对某些“非标准 jsonl”（例如行首夹杂杂字符、尾部夹杂额外文本）的情况。

### 3.2 提取用户文本：`_extract_user_text(obj)`

脚本优先提取“用户侧文本”，以便校准集更贴近真实推理输入。

兼容两类常见数据结构：

1. Chat messages 格式：
	- 若存在 `messages: list`，会遍历其中 `role == user` 的第一条有效 `content`。
2. 扁平字段格式：
	- 依次尝试 `prompt / instruction / question / input / text` 字段。

### 3.3 归一化与去重：`_normalize_question(q)` + `seen`

为了避免校准集被大量“提示词变体”占满，脚本做了两层处理：

- 空白归一化：把连续空白压缩为单个空格。
- 前缀清理：移除一些数据中常见的固定前缀（例如 `简答：`、`只给结论：` 等）。

然后对归一化后的文本做集合去重（`seen`）：

- 同一个问题即使在源数据里出现多次（或带不同前缀），最终只保留一条候选。

### 3.4 抽样：`sample_calib(...)`

流程概括：

1. 遍历 `DATA_JSONL`，提取用户文本。
2. 过滤过短文本（`len < MIN_LEN`）。
3. 对过长文本按字符截断到 `MAX_LEN`。
4. 去重后得到候选列表 `candidates`。
5. 若候选数大于 `N`：用 `random.Random(SEED).sample(candidates, N)` 无放回抽样。

返回值包含：

- `texts`：最终抽样结果（或全部候选）
- `total_candidates`：去重后的候选总数

### 3.5 输出格式：`main()`

`main()` 负责读取环境变量、调用 `sample_calib`、并写出两份文件：

- `OUT_JSONL`：每行 `{"text": <string>}`
- `OUT_TXT`：每行纯文本，并把换行替换为空格（便于肉眼扫一遍）

控制台会打印：

- `[sample_calib] data=..., unique_candidates=..., sampled=...`
- `[sample_calib] wrote: ...`

---

## 4) 常见问题

### Q1：为什么长度单位是“字符”而不是 token？

这是一个轻量的预处理筛选：字符长度便于快速过滤太短/太长文本。真正的 token 截断会在量化脚本里由 tokenizer 与 `max_seq_len` 再做一次。

### Q2：为什么要去掉“简答：/只给结论：”这类前缀？

这些前缀在数据里往往重复出现，会导致校准集出现大量“同题不同提示词”的近重复样本，反而降低校准多样性。

### Q3：输出条数少于 N 是 bug 吗？

不是。脚本会先去重和过滤长度；如果去重后的候选数本来就小于 `N`，脚本会输出全部候选，并在日志里显示 `unique_candidates`。

### Q4：我希望抽样更贴近线上题目分布，应该怎么做？

思路通常有两种：

- 改 `DATA_JSONL`：提供一个更贴近线上题目来源/领域的数据。
- 调整过滤策略：例如提高 `MIN_LEN`、降低/提高 `MAX_LEN`，或修改 `_normalize_question` 的前缀列表。

### Q5：jsonl 里是 messages 格式，但为什么抽不到？

脚本只会取 `messages` 中 `role==user` 的 `content`。若数据使用了不同 role 命名（例如 `human`），需要改 `_extract_user_text` 里的匹配规则。
