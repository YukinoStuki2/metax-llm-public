---
title: 自动调参（auto_tune.py / auto_tune.sh）
sidebar_position: 50
---

本仓库提供守护式调参脚本：对一组参数组合重复执行“启动服务→跑评测→统计均值→记录→关闭服务”。

- 入口脚本：`auto_tune.sh`（会自动 source `tune_secrets.sh`，若存在）
- 核心实现：`auto_tune.py`
- 守护重启：`run_autotune_forever.sh`

## 快速开始

建议先把环境变量清干净：

```bash
source ./env_force.sh
./auto_tune.sh
```

常用覆盖：

```bash
ACC=0.8800 EVAL_RUNS=5 ./auto_tune.sh
```

## 结果文件

默认输出（位于 repo 根目录）：

- `tune_results.jsonl`：每个 trial 一条记录（包含均值准确率/吞吐/耗时/参数/状态）
- `best_params.json`：当前最优参数（只保存 params，不保存分数）
- `tune_status.json`：实时状态（便于外部监控）
- `tune_server_logs/`：每轮启动的服务日志

仓库里也有 `*.selftest*` 文件，用于脚本自检/示例。

## 搜索空间

`auto_tune.py` 内置默认搜索空间（示例）：

- `GPU_MEMORY_UTILIZATION`
- `VLLM_MAX_NUM_BATCHED_TOKENS`
- `VLLM_MAX_NUM_SEQS`
- `MAX_MODEL_LEN`
- 以及若干 stop/长度/惩罚项相关参数

你可以用 JSON 文件扩展搜索空间：

```bash
export TUNE_SEARCH_SPACE_FILE=./tune_search_space.json
./auto_tune.sh
```

文件格式：

```json
{
  "GPU_MEMORY_UTILIZATION": ["0.965", "0.975"],
  "NEW_PARAM": ["a", "b"]
}
```

## 通知（可选）

复制示例并填写：

```bash
cp tune_secrets.example.sh tune_secrets.sh
chmod 600 tune_secrets.sh
```

- 飞书：`TUNE_WEBHOOK_KIND=feishu` + `TUNE_WEBHOOK_URL`
- 邮件：`TUNE_SMTP_*` 一组变量

## 端口占用处理

默认只等待重试；如确认占用者是残留的 `uvicorn serve:app`，可谨慎开启清理：

```bash
export TUNE_PORT_BUSY_KILL=1
./auto_tune.sh
```
