---
title: AWQ 量化（quantize_awq.py）
sidebar_position: 90
---

# AWQ 量化脚本详解

## 1) 脚本功能概述

### 1.1 作用与定位

`quantize_awq.py` 是一个“一次性量化导出”脚本：对本地 HF 模型目录执行 AutoAWQ 4bit 权重量化（AWQ），并导出一个新的量化模型目录。

它通常用于：

- 在显存/吞吐受限时，用 4bit 权重量化降低显存占用、提升推理吞吐。

- 为后续部署（例如 vLLM/transformers 插件）准备一个“可直接加载”的 AWQ 模型目录。

> 量化是否适合评测以实际分数、稳定性为准；AWQ 往往能显著提速，但可能带来一定精度损失。

### 1.2 输入与输出

**输入：**

- `--model_dir`：本地 HF 模型目录（必须包含 `config.json`，通常还包含 tokenizer 文件）。

- `AWQ_CALIB_JSONL`（环境变量）或默认 `calib_8192.jsonl`：用于校准的 jsonl 文件。

- `--output_dir`：量化后导出目录。

**输出：**

- `output_dir`：量化后的模型目录（包含量化权重与 tokenizer）。

- 额外文件：脚本会尝试写出 `quant_config.json`，并在有需要时写出 `modules_to_not_convert.txt`。

- 控制台日志：打印量化配置、校准规模、保存路径等。

**会改变什么：**

- 会在 `output_dir` 写入/覆盖量化模型文件（目录存在时会复用，但文件会被新的导出结果覆盖）。

**不改变：**

- 不会修改 `model_dir` 与校准集文件。

---

## 2) 参数与环境变量详解

### 2.1 命令行参数

#### `--model_dir`

- **作用**：待量化的本地 HF 模型目录。

- **默认值**：环境变量 `AWQ_MODEL_DIR`，否则为脚本内置 `model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM`。

- **校验**：

  - 必须是目录；否则直接退出。

  - 必须包含 `config.json`；缺失会直接退出。

#### `--output_dir`

- **作用**：量化导出目录。

- **默认值**：环境变量 `AWQ_OUTPUT_DIR`，否则为脚本内置 `model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ`。

- **说明**：脚本会 `mkdir -p` 创建目录；不会自动清空目录。

#### `--trust_remote_code` / `--no_trust_remote_code`

- **作用**：控制是否启用 `trust_remote_code`（Qwen 系列通常需要）。

- **默认值**：默认启用（`--trust_remote_code` 的 default=True）。

- **建议**：除非明确该模型无需 remote code，否则不要关闭。

### 2.2 环境变量

#### `AWQ_CALIB_JSONL`

- **作用**：指定校准集 jsonl 的路径。

- **默认值**：脚本内置 `calib_8192.jsonl`。

- **格式要求**：每行一个 JSON 对象，且至少包含：`{"text": "..."}`。

#### `AWQ_MODEL_DIR` / `AWQ_OUTPUT_DIR`

- **作用**：覆盖 `--model_dir` / `--output_dir` 的默认值。

#### `TRANSFORMERS_NO_TORCHVISION`

- **作用**：避免 `transformers` 因 torchvision/torch 版本不匹配导致导入失败。

- **脚本行为**：脚本会在运行时 `os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")`，即默认设为 `1`（如已设置其它值则保留）。

### 2.3 量化配置（脚本内置常量）

这部分不提供命令行参数，目的是“减少可变项，优先保准确率”。如需调整请直接修改脚本常量。

#### 权重量化配置

- `AWQ_W_BIT = 4`：权重位宽（4bit）。

- `AWQ_Q_GROUP_SIZE = 128`：group size（兼容性优先，很多插件对 128 支持更稳）。

- `AWQ_ZERO_POINT = True`：是否启用 zero-point。

- `AWQ_BACKEND = "GEMM"`：AutoAWQ quant backend。

- `AWQ_MODULES_TO_NOT_CONVERT = ["lm_head"]`：默认不量化 `lm_head`，更稳。

#### 校准规模与序列长度

- `AWQ_NUM_CALIB = 8192`：校准样本数（从 jsonl 随机抽样）。

- `AWQ_MAX_SEQ_LEN = 2048`：校准阶段最大序列长度。

#### 校准 prompt 分布

- `AWQ_CALIB_APPLY_CHAT_TEMPLATE = True`：默认把校准文本包装成 chat prompt。

- `AWQ_CALIB_SYSTEM_PROMPT`：内置的系统提示词，用于让校准分布更接近线上推理（更像 system+user+assistant 的格式）。

---

## 3) 代码实现详解

### 3.1 依赖库与职责

- `argparse` / `os` / `pathlib.Path`：参数与路径处理。

- `random`：用于对校准集做均匀随机抽样。

- `torch`：用于检测 CUDA 并加载模型。

- `transformers.AutoTokenizer`：加载 tokenizer，用于 chat template 与量化 API。

- `awq.AutoAWQForCausalLM`：AutoAWQ 的模型加载与量化导出。

脚本采用“延迟 import”：只有真正开始量化时才导入 `torch/transformers/awq`，避免 `--help` 时强依赖环境。

### 3.2 校准集读取：`_iter_calib_texts()`

- 输入：`calib_jsonl` 路径与 `limit`（即 `AWQ_NUM_CALIB`）。

- 读取方式：单次扫描 jsonl，使用 reservoir sampling 做**均匀随机抽样**，避免总是拿文件开头导致分布偏差。

- 过滤逻辑：

  - 解析失败/空行/非 dict/缺少 `text` 都会跳过。

- 输出：`List[str]` 校准文本。

### 3.3 Chat 模板包装：`_format_calib_as_chat()`

当 `AWQ_CALIB_APPLY_CHAT_TEMPLATE=1` 时：

- 优先调用 `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`。

- 若 tokenizer 不支持或抛异常，则回退到 Qwen 常见 `<|im_start|>...` 格式。

这样做的目的：让量化校准阶段看到的 prompt 更贴近实际推理时的格式，从而减少量化带来的分布偏移。

### 3.4 量化与导出：`main()`

主流程关键点：

1. 路径校验：

- `model_dir` 必须存在且包含 `config.json`。

- 校准集 `calib_jsonl` 必须存在；缺失会提示如何用 `sample_calib_from_data.py` 生成。

1. 环境兼容性：默认设置 `TRANSFORMERS_NO_TORCHVISION=1`。

2. 加载 tokenizer 与校准数据：

- 抽样得到 `AWQ_NUM_CALIB` 条文本。

- 可选包装成 chat prompt。

1. 构造 `quant_config` 并打印：

- `zero_point/q_group_size/w_bit/version`。

- 并打印 `modules_to_not_convert`。

1. 加载模型：

- `AutoAWQForCausalLM.from_pretrained(..., device_map="auto" if cuda else None)`。

1. 执行量化：

- 调用 `model.quantize(...)`。

- 为兼容不同版本 AutoAWQ 的函数签名，脚本做了 `TypeError` 回退调用。

1. 保存导出：

- `model.save_quantized(output_dir, safetensors=True)`

- `tokenizer.save_pretrained(output_dir)`

- 尝试写 `quant_config.json`（以及 `modules_to_not_convert.txt`）提高下游加载兼容性。

---

## 4) 常见问题

### Q1：提示缺少 AutoAWQ 依赖怎么办？

脚本会在导入失败时直接退出，并提示安装量化依赖。按仓库建议在单独环境安装 `requirements-quantize-awq.txt` 后再运行。

### Q2：校准集文件不存在或格式错误怎么办？

- 文件不存在：设置 `AWQ_CALIB_JSONL` 指向正确路径，或用脚本提示的命令生成。

- 格式错误：确保每行是 JSON，且包含 `text` 字段，例如：`{"text": "..."}`。

### Q3：OOM（显存不足）怎么处理？

OOM 通常发生在“校准量化阶段”，优先调整脚本常量：

- 降低 `AWQ_MAX_SEQ_LEN`（最有效）

- 降低 `AWQ_NUM_CALIB`

其次再考虑：减少并发、关闭其它占显存进程、换更大显存 GPU。

### Q4：量化后掉分明显怎么办？

常见思路：

- 让校准集分布更贴近线上题目：校准集来自同类领域文本比“泛文本”更稳。

- 增大 `AWQ_NUM_CALIB`（在显存允许的前提下）。

- 保持 `AWQ_CALIB_APPLY_CHAT_TEMPLATE=1`，不要用完全原始文本做校准。

### Q5：为什么脚本默认不量化 `lm_head`？

这是偏保守的策略：不量化 `lm_head` 往往更稳，能减少输出分布被破坏的风险；代价是占用略多一点显存与体积。

### Q6：量化导出后下游加载报错（缺少 quant_config.json）怎么办？

脚本已经尝试显式写出 `quant_config.json`；如果日志里出现写文件失败的 warning，请检查：

- `output_dir` 是否有写权限

- 文件系统是否只读

必要时手动创建 `quant_config.json` 并与日志输出的 `quant_config` 保持一致。