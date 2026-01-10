---
sidebar_position: 4
---

# 关键文件说明

面向评测交付（Gitee 精简仓库）时，通常只需要少量文件即可完成构建与运行。

## `Dockerfile`

- 定义镜像构建流程：安装依赖、下载模型、设置默认环境变量
- 固定对外端口（通常为 8000）

## `serve.py`

- 推理服务核心：FastAPI 应用
- 提供：
  - `GET /` 健康检查
  - `POST /predict` 推理接口（支持单条与 batch）
- 运行阶段断网：不要在请求路径进行联网调用

## `download_model.py`

- 构建阶段从 ModelScope 下载模型权重到本地目录
- 可选下载 speculative decoding 的 draft 模型（如启用）

## `requirements.txt`

- 服务端与构建下载所需的最小 Python 依赖

## `data.jsonl`

- 可选：用于预热/抽样测试的数据文件
- 不应成为服务启动的强依赖

## 进一步阅读（GitHub 文档站）

为了避免单页过长，仓库的 `.py/.sh` 入口脚本已拆分到多个文档页：

- 推理服务：见「推理服务（serve.py）」
- 启动脚本：见「启动脚本（run_model.sh / env_force.sh）」
- WebUI：见「WebUI（Gradio）」
- 本地评测：见「本地评测（eval_local.py）」
- 自动调参：见「自动调参（auto_tune.py / auto_tune.sh）」
- 模型工程：见「模型下载 / 融合 LoRA / 上传」
- 量化：见「AWQ 量化 / 抽样校准集」
