---
sidebar_position: 2
---

# 快速启动（Docker / API）

## Docker 运行

在评测/部署场景中，通常通过 Docker 构建并运行服务。

- 构建（build 阶段会下载模型权重）：

```bash
docker build -t metax-llm:docs .
```

- 运行（默认暴露 8000）：

```bash
docker run --rm -p 8000:8000 metax-llm:docs
```

> 不同平台 GPU 运行参数不同（NVIDIA/MetaX 等），以各平台模板为准。

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
