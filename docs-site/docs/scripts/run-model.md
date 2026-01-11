---
title: 启动脚本（run_model.sh / env_force.sh）
sidebar_position: 20
---

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
