---
sidebar_position: 3
---

# 快速开始

本项目提供两种启动方式：

- 本地/云主机：`env_force.sh + run_model.sh` 启动后端，可选再启动 WebUI。
- 评测形态：Docker 构建并运行（评测系统通常不包含 WebUI）。

## 使用 WebUI（推荐本地调试）

1）导入默认环境变量（清理上次遗留参数）：

```bash
source ./env_force.sh
```

2）启动推理后端（端口 8000）：

```bash
bash ./run_model.sh
```

3）启动 Web 界面（端口 7860）：

```bash
./start_webui.sh
```

浏览器访问：[http://localhost:7860](http://localhost:7860)

> WebUI 侧还提供生成参数面板、SYSTEM_PROMPT 编辑、RAG/Batch 测试等，详见「[WebUI（Gradio）](webui/overview)」。

（可选）如果在远程服务器上启动，可通过 SSH 隧道访问 WebUI：

```bash
ssh -CNg -L 7860:127.0.0.1:7860 root+<username>@<IP> -p <PORT>
```

## Docker 启动（更贴近评测机）

在评测/部署场景中，通常通过 Docker 构建并运行服务。

- 构建（build 阶段会下载模型权重）：

```bash
docker build -t metax-demo:latest .
```

- 运行（默认暴露 8000）：

```bash
docker run --rm -p 8000:8000 metax-demo:latest
```

> 注意：沐曦容器上无法直接用 Docker 启动（需要按平台提供的评测方式运行）。不同平台 GPU 运行参数不同（NVIDIA/MetaX 等），以各平台模板为准。

时间限制（参考）：

```text
docker build stage: 900s
docker run - health check stage: 180s
docker run - predict stage: 360s
```

## 接口自测

- 健康检查：

```bash
curl -s http://127.0.0.1:8000/
```

- 推理请求：

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请简要回答：什么是xxx？"}'
```

## Batch（可选提速）

当开启 batch 模式时，评测系统可能会一次性把所有问题推到 `POST /predict`。

- batch 请求：

```json
{"prompt": ["Q1", "Q2", "Q3"]}
```

- batch 响应：

```json
{"response": ["A1", "A2", "A3"]}
```

要求：返回数组长度必须与问题数量一致。

提示：若模型输出包含 `<think>...</think>`，建议在返回前剥离，避免影响评测。
