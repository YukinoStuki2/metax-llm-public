---
title: 推理服务（serve.py）
sidebar_position: 10
---

本项目评测交付的核心是 `serve.py`：用 FastAPI 提供稳定的评测接口。

## 评测接口契约（不要破坏）

- `GET /`：健康检查，必须快速返回（评测机会先调用它）。
- `POST /predict`：请求 JSON `{"prompt":"..."}`，返回 JSON `{"response":"..."}`。
- 端口固定 `8000`（Dockerfile 已 `EXPOSE 8000`）。

### Batch 模式
当 `BATCH_MODE=1` 时，服务端会把健康检查返回为 `{"status":"batch"}`，并支持：

- 单条：`{"prompt": "..."}` → `{"response":"..."}`
- 批量：`{"prompt": ["q1", "q2", ...]}` → `{"response":["a1", "a2", ...]}`

（服务端也会对“服务端仍返回单条字符串”等异常情况做一定兜底，见 `eval_local.py` 的调用逻辑。）

## 启动方式

### Docker（评测机推荐）
参考「快速开始」页面；Dockerfile 在 build 阶段会下载模型到 `./model/$MODEL_ID`，run 阶段不联网。

### 裸机/容器内（本地复现）
使用 `run_model.sh` 复现 Dockerfile：安装依赖 → 下载模型 → 启动 `uvicorn serve:app`。

## 关键环境变量

`serve.py` 的行为主要由环境变量驱动（Dockerfile 和脚本也会设置同名变量）。

### 模型路径
- `MODEL_DIR`：默认 `./model/$MODEL_ID`（与 Dockerfile/run_model.sh 对齐）。

### 生成长度（控吞吐/控截断）
- `MAX_NEW_TOKENS`：默认 32（serve.py 内默认）；Dockerfile 默认 64。
- `MAX_NEW_TOKENS_CODE`：代码题更长输出（默认 192）。
- `MAX_NEW_TOKENS_CODE_SOFT` / `MAX_NEW_TOKENS_CODE_HARD`：更细的代码题上限控制。
- `LONG_ANSWER_ENABLE_DEFAULT` / `LONG_ANSWER_MIN_HITS`：启发式允许更长答案。

### 解码与稳定性
- `TEMPERATURE`（默认 0.0）、`TOP_P`（默认 1.0）、`TOP_K`（默认 1）
- `REPETITION_PENALTY`、`FREQUENCY_PENALTY`：抑制复读（小模型尤其重要）。

### 提前停止（减少尾巴 token）
- `STOP_STRINGS`：逗号分隔的 stop 字符串列表（默认含 `<|im_end|>,<|endoftext|>`）。
- `STOP_ON_DOUBLE_NEWLINE`：若为 1，会把 `"\n\n"` 也作为 stop（可能提升吞吐，但也可能过早截断）。

### vLLM / 并发相关
- `USE_VLLM=true|false`：优先使用 vLLM（推荐）。
- `BATCH_CONCURRENCY`：批量请求时服务端并发提交给引擎的并发度。
- `GPU_MEMORY_UTILIZATION`、`MAX_MODEL_LEN`、`VLLM_MAX_NUM_SEQS`、`VLLM_MAX_NUM_BATCHED_TOKENS`：吞吐与 OOM 的主要权衡点。

## 输出纪律

评测按 RougeL（jieba 分词）计分。

- 尽量短答、贴近教材表述。
- 不输出“思考过程”。
- 若模型输出包含 `<think>...</think>`，服务端会做剥离/裁剪（具体规则以 `serve.py` 实现为准）。

## 常见问题

- **健康检查慢/超时**：优先关闭预热相关配置（`WARMUP_*`），避免评测 health 阶段被拖慢。
- **OOM 或并发很低**：先降低 `MAX_MODEL_LEN`，再调整 `GPU_MEMORY_UTILIZATION/VLLM_MAX_NUM_*`。
