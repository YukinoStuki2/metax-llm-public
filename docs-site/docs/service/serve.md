---
title: 推理服务（用法）
sidebar_position: 10
---

本项目评测交付的核心是 `serve.py`：用 FastAPI 提供稳定的评测接口，并尽量兼顾 **准确率（RougeL）** 与 **吞吐**。

本服务文档拆为 4 篇（本页是第 1/4）：

- 第 2/4：参数与环境变量（默认值、覆盖关系、调参含义）见 [推理服务（参数与环境变量）](./serve_env)

- 第 3/4：实现细节（vLLM/transformers、batch、预热、回退策略）见 [推理服务（代码详解）](./serve_code)

- 第 4/4：常见问题与排查建议见 [推理服务（FAQ）](./serve_faq)

## 评测接口

- `GET /`：健康检查，必须快速返回（评测机会先调用它）。

- `POST /predict`：请求 JSON `{"prompt":"..."}`，返回 JSON `{"response":"..."}`。

- 端口固定 `8000`（Dockerfile 已 `EXPOSE 8000`）。

除以上外，`serve.py` 还提供了一些辅助接口（不影响评测）：

- `GET /info`：返回后端信息与环境变量白名单视图（供 WebUI 展示）

- `GET /system_prompt`：读取当前 system prompt

- `POST /system_prompt`：运行时更新 system prompt（无需重启，立即影响后续 /predict）

### Batch 模式

当 `BATCH_MODE=1` 时，服务端会把健康检查返回为 `{"status":"batch"}`，并支持：

- 单条：`{"prompt": "..."}` → `{"response":"..."}`

- 批量：`{"prompt": ["q1", "q2", ...]}` → `{"response":["a1", "a2", ...]}`

（服务端也会对“服务端仍返回单条字符串”等异常情况做一定兜底，见 `eval_local.py` 的调用逻辑。）

## 性能优化

为了贴合评测的计分与时限，`serve.py` 额外做了几件“评测导向”的工作：

- **system prompt 强约束**：强调短答、贴近教材措辞、不要思考过程。

- **输出纪律**：若模型输出包含 `<think>...</think>`，服务会做剥离/回退处理，避免返回很长的思考内容。

- **按题型分流生成长度**：短答题默认很短；代码题/长答案题会自动放宽 `max_new_tokens`，减少截断。

- **后处理（短答模式）**：对“举例扩展”进行裁剪、限制句子数，减少发散并提升 Rouge-L 词序重合。

这些行为的开关与默认值详见 [推理服务（参数与环境变量）](./serve_env)。

## 启动方式

### Docker（评测机）

参考「快速启动」页面；Dockerfile 在 build 阶段会下载模型到 `./model/$MODEL_ID`，run 阶段不联网。

### 裸机/容器内（本地复现）

使用 `run_model.sh` 复现 Dockerfile：安装依赖 → 下载模型 → 启动 `uvicorn serve:app`。

## 下一部分

- 调参/对齐 Dockerfile 与脚本： [推理服务（参数与环境变量）](./serve_env)

-  batch 模式提速、Transformers回退： [推理服务（代码详解）](./serve_code)

- 遇到 health 超时、OOM、tokens/s 异常： [推理服务（FAQ）](./serve_faq)