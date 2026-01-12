---
title: 启动脚本（run_model.sh / env_force.sh）
sidebar_position: 20
---

本页面对应两个脚本：

- `run_model.sh`：在 Docker 外“一键复现” Dockerfile 的运行链路（装依赖 → 下载模型 → 启动 `serve.py`）
- `env_force.sh`：强制导入一套“干净参数”，清除你当前 shell 里遗留的环境变量污染

文档结构：

1. 用法 / 作用 / 输入输出 / 副作用
2. 参数与环境变量详解
3. 代码详解
4. 常见问题（FAQ）

---

## 1) 用法 / 作用 / 输入输出 / 副作用

### 1.1 run_model.sh：最小可复现启动

作用：

- 尽量复现 Dockerfile 的链路：
  - 安装 `requirements.txt`
  - 执行 `download_model.py` 下载模型到 `./model/`
  - 启动 `uvicorn serve:app --host 0.0.0.0 --port 8000`
- 只在“未设置该变量”时写入默认值（`:` + `${VAR:=default}`），不会覆盖你已 `export` 的变量。

用法：

```bash
./run_model.sh
```

常见加速（本地反复试验时很有用）：

```bash
SKIP_PIP_INSTALL=1 SKIP_MODEL_DOWNLOAD=1 ./run_model.sh
```

输入：

- 主要通过环境变量输入（见第 2 节）。

输出：

- 标准输出打印运行阶段与 Python 解释器选择（前缀为 `[run_model]`）。
- 启动服务后，进程会被 `exec` 替换成 `uvicorn`；脚本本身不再继续执行。

副作用：

- 可能会对当前 Python 环境执行 `pip install -r requirements.txt`（默认开启）。
- 可能会下载模型权重到 `./model/`（默认开启）。
- 会监听 `0.0.0.0:8000` 启动推理服务。

### 1.2 env_force.sh：强制导入“干净参数”

作用：

- 用一组固定值覆盖当前 shell 的同名环境变量，用于清理上一次实验留下的参数污染。
- 该脚本本身不启动服务，只负责 `export`。

用法（必须 source）：

```bash
source ./env_force.sh
./run_model.sh
```

临时覆盖某个参数（推荐写法）：

```bash
source ./env_force.sh
export MAX_MODEL_LEN=2048
./run_model.sh
```

输入：无（只读取你已有的 `MODELSCOPE_API_TOKEN` 的值作为兜底）。

输出：

- 打印一行提示当前已加载的关键变量（前缀为 `[env_force]`）。

副作用：

- 覆盖当前 shell 的同名环境变量。
- 如果你直接运行 `./env_force.sh` 而不是 `source`，会退出并返回退出码 `2`。

---

## 2) 参数与环境变量详解

这一节不解释 `serve.py` 的所有变量语义（那部分在服务文档里）；这里只说明：

- 这些变量是否被脚本设置
- 默认值是多少
- 你如何覆盖
- 脚本与 Dockerfile 是否一致

### 2.1 变量优先级（非常重要）

`run_model.sh` 的优先级：

1. 你在运行脚本前已经 `export` 的环境变量
2. `run_model.sh` 内用 `: "${VAR:=default}"` 设定的默认值

`env_force.sh` 的优先级：

- `env_force.sh` 会强制 `export VAR=...` 覆盖同名变量（少数例外：`MODELSCOPE_API_TOKEN` 只在未设置时保留你的值）。

### 2.2 与 Dockerfile 的一致性说明

理想状态下：Dockerfile / `run_model.sh` / `env_force.sh` 应保持同一套默认参数，避免出现“本地好、线上坏”。

当前仓库里存在以下事实（以代码为准）：

- Dockerfile 与 `run_model.sh` 的 `MODEL_ID` 默认值一致（均为 `YukinoStuki/Qwen2.5-0.5B-Plus-LLM`）。
- `env_force.sh` 会把 `MODEL_ID` 强制设为 `YukinoStuki/Qwen3-4B-Plus-LLM`，这会导致“source env_force.sh 后再跑 run_model.sh”加载不同模型。

如果你的目标是严格复现评测镜像的默认行为，不要 source `env_force.sh`，直接运行 `run_model.sh`（或仅手动覆写你确实要改的少数变量）。

### 2.3 run_model.sh 设置/读取的关键变量

模型与下载：

- `MODEL_ID`（默认 `YukinoStuki/Qwen2.5-0.5B-Plus-LLM`）
- `MODEL_REVISION`（默认 `master`）
- `MODEL_DIR`（默认 `./model/$MODEL_ID`）
- `MODELSCOPE_API_TOKEN`（默认空字符串；用于 `download_model.py --token ...`）
- 跳过下载：`SKIP_MODEL_DOWNLOAD=1`

依赖安装：

- 跳过安装：`SKIP_PIP_INSTALL=1`

Speculative Decoding（可选，默认关闭）：

- `ENABLE_SPECULATIVE_DECODING`（默认 `0`）
- `SPEC_METHOD`（默认 `ngram`）
- `SPEC_NUM_SPECULATIVE_TOKENS`（默认 `6`）
- `SPEC_NGRAM_LOOKUP_MAX`（默认 `8`）
- `SPEC_NGRAM_LOOKUP_MIN`（默认 `1`）
- `SPEC_DRAFT_MODEL_ID`（默认空）
- `SPEC_DRAFT_MODEL_REVISION`（默认 `master`）

推理服务相关（传给 `serve.py` 作为环境变量）：

- `USE_VLLM`（默认 `true`）
- `DISABLE_TOKEN_ROUTING`（默认 `0`）
- 解码：`TEMPERATURE`（默认 `0.0`）、`TOP_P`（默认 `1.0`）、`TOP_K`（默认 `1`）
- 长度：`MAX_NEW_TOKENS`（默认 `64`）、`MAX_MODEL_LEN`（默认 `1024`）
- vLLM 资源：`GPU_MEMORY_UTILIZATION`（默认 `0.97`）、`DTYPE`/`TRANSFORMERS_DTYPE`（默认 `float16`）
- Batch：`BATCH_MODE`（默认 `1`）、`BATCH_CONCURRENCY`（默认 `512`）、`VLLM_BATCH_USE_LLM`（默认 `1`）
- vLLM 其他：`ENABLE_PREFIX_CACHING`（默认 `1`）、`VLLM_QUANTIZATION`/`VLLM_KV_CACHE_DTYPE`（默认空）、`VLLM_LOAD_FORMAT`（默认 `auto`）
- vLLM 上限：`VLLM_MAX_NUM_SEQS`（默认 `1024`）、`VLLM_MAX_NUM_BATCHED_TOKENS`（默认空）
- 停止条件：`STOP_STRINGS`（默认 `<|im_end|>,<|endoftext|>`）、`STOP_ON_DOUBLE_NEWLINE`（默认 `0`）
- 输出后处理：`OUTPUT_TRIM_EXAMPLES`（默认 `1`）、`OUTPUT_MAX_SENTENCES`（默认 `6`）
- 预热：`WARMUP_DATA_PATH`（默认 `./data.jsonl`）、`WARMUP_NUM_SAMPLES`（默认 `7000`）、`WARMUP_REPEAT`（默认 `2`）

### 2.4 env_force.sh 强制导入的关键变量

`env_force.sh` 会强制设置一整套变量（覆盖你当前 shell 中的同名项）。其中几个与 `run_model.sh`/Dockerfile 不同的点（以当前脚本内容为准）：

- `MODEL_ID`：强制为 `YukinoStuki/Qwen3-4B-Plus-LLM`
- `DISABLE_TOKEN_ROUTING`：强制为 `1`
- `MAX_NEW_TOKENS`：强制为 `512`
- `MAX_MODEL_LEN`：强制为 `8192`
- `WARMUP_REPEAT`：强制为 `1`
- `VLLM_MAX_NUM_SEQS`：强制为空字符串（让后端/引擎自行决定）

---

## 3) 代码详解

### 3.1 run_model.sh 的执行流程

1. 进入脚本所在目录：`cd "$(dirname "$0")"`
	- 作用：保证相对路径（`requirements.txt`、`download_model.py`、`./model/`）在任意工作目录下都可用。
2. 设置默认环境变量
	- 大量使用 `: "${VAR:=default}"`：只有当变量未设置时才赋默认值。
	- `OMP_NUM_THREADS` 例外：直接 `export OMP_NUM_THREADS="4"`（强制覆盖）。
3. 选择 Python 解释器
	- 优先使用 `python3`，否则退回 `python`。
4. 安装依赖（可跳过）
	- 若 `SKIP_PIP_INSTALL!=1`：执行 `python -m pip install --no-cache-dir -r requirements.txt`。
5. 下载模型（可跳过）
	- 若 `SKIP_MODEL_DOWNLOAD!=1`：调用 `download_model.py` 下载模型到 `./model`。
	- 会把 `MODEL_ID`、`MODEL_REVISION`、`MODELSCOPE_API_TOKEN` 以及 speculative draft 相关参数一并传入。
6. 启动服务
	- `exec python -m uvicorn serve:app --host 0.0.0.0 --port 8000`
	- 使用 `exec` 的含义：当前 shell 进程被 uvicorn 替换，信号处理更直接，也不会多一层父进程。

### 3.2 env_force.sh 的关键实现点

- 首先检测是否被 source：
  - 如果 `BASH_SOURCE[0] == $0`，说明你是直接执行，会打印提示并 `exit 2`。
- 然后用一串 `export VAR="..."` 强制覆盖环境变量。
- 对 `MODELSCOPE_API_TOKEN` 的处理是“有就用，没有才用默认空值”：
  - `: "${MODELSCOPE_API_TOKEN:=}"`

---

## 4) 常见问题（FAQ）

### Q1：为什么一定要 `source ./env_force.sh`？

因为环境变量属于当前 shell 进程。你直接运行 `./env_force.sh` 时，变量只在子进程里生效，子进程退出后不会影响你当前终端，所以脚本会直接退出并返回码 `2`。

### Q2：我在 venv 里装了依赖，为什么 run_model.sh 又装了一遍？

`run_model.sh` 会使用 `python3`/`python`（取决于 PATH），它不会自动激活你的 venv。

可选解决：

- 先手动激活 venv 再运行 `./run_model.sh`
- 或使用 `SKIP_PIP_INSTALL=1`，自己管理依赖版本

### Q3：如何只改一个参数，又不被 env_force 覆盖？

推荐流程：

```bash
source ./env_force.sh
export MAX_MODEL_LEN=2048
./run_model.sh
```

如果你先 `export` 再 `source env_force.sh`，会被强制覆盖回 env_force 的值。

### Q4：为什么 source env_force 后启动的模型和评测镜像不一致？

因为当前 `env_force.sh` 会强制设置 `MODEL_ID` 等变量，与 Dockerfile/`run_model.sh` 的默认值不同。要严格复现评测镜像默认行为，请不要 source `env_force.sh`。

### Q5：端口被占用怎么办？

`run_model.sh` 固定启动在 `0.0.0.0:8000`。如果 8000 端口已占用，你需要先停止占用进程，或自行修改启动命令（例如改 `--port`）并同步你的调用方（评测固定端口不应修改）。

这两个脚本用于在 Docker 外复现 Dockerfile 的“安装依赖→下载模型→启动服务”流程，并帮助你管理一套可控的环境变量。

## run_model.sh：最小可复现启动

`run_model.sh` 的设计原则：

- **有就用，没有才用默认**：用 `: "${VAR:=default}"` 的方式设置默认值，不强制覆盖你提前 `export` 的变量。
- 复现 Dockerfile：安装 `requirements.txt` → 运行 `download_model.py` → `uvicorn serve:app`。

常用：

```bash
./run_model.sh
```

跳过安装/下载（本地反复试验加速）：

```bash
SKIP_PIP_INSTALL=1 SKIP_MODEL_DOWNLOAD=1 ./run_model.sh
```

## env_force.sh：强制导入一套“干净参数”

`env_force.sh` 会 **覆盖当前 shell 中的同名环境变量**，用于清除上一次实验留下的变量污染。

用法（必须 `source`）：

```bash
source ./env_force.sh
./run_model.sh
```

临时覆盖某个参数（推荐方式）：

```bash
source ./env_force.sh
export MAX_MODEL_LEN=2048
./run_model.sh
```

> 说明：Dockerfile / run_model.sh / env_force.sh 的默认参数应保持一致；若你修改了其中之一，请同步检查另外两处，避免出现“本地好/线上坏”。

## 关键变量速查（与 Dockerfile 同名）

- 模型：`MODEL_ID` / `MODEL_REVISION` / `MODEL_DIR`
- 服务：`USE_VLLM`、`BATCH_MODE`、`BATCH_CONCURRENCY`
- 解码：`TEMPERATURE`、`TOP_P`、`TOP_K`
- 长度：`MAX_NEW_TOKENS`、`MAX_MODEL_LEN`
- vLLM：`GPU_MEMORY_UTILIZATION`、`VLLM_MAX_NUM_SEQS`、`VLLM_MAX_NUM_BATCHED_TOKENS`

## 常见坑

- **直接运行 env_force.sh 不生效**：必须 `source ./env_force.sh`。
- **脚本用系统 python**：`run_model.sh` 默认使用 `python3`/`python`，与你本地 venv 可能不同；如需固定解释器，建议在外层手动激活 venv 后再跑。
