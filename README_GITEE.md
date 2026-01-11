# 微调Qwen大模型的推理部署与工具

说明：本 README 面向“同步到 Gitee 的精简文件集”。Gitee 仓库只保留评测运行必须文件（见下文“文件说明”）；完整研发/调参/工具脚本以 GitHub 仓库为准。

---

**2025年秋季中国科学院大学(国科大)《GPU架构与编程》课程项目二**

📖 详细文档请查看 <https://docs-gpu.yukino.uk>

## 摩尔线程一等奖——沐曦赛道

---

本项目提供了 vLLM 加速的推理代码和 Web 界面以及一些测试模型的工具脚本。

Gitee仓库（用于实际提交）：<https://gitee.com/yukinostuki/metax-demo>

Github仓库（用于具体开发）：<https://github.com/YukinoStuki2/metax-llm-public>

两个仓库仅有文件数量区别，其版本和代码内容都是同步的。

## 1) 项目介绍

这是一个对微调开源大模型进行推理，用于问答评测的项目：

- 微调数据集来源于《Programming Massively Parallel Processors.2017》。

- 评测目标：在准确率（RougeL-F1，jieba 分词）达到阈值（常见参考 ≥ 0.35）的前提下，尽量提升吞吐（tokens/s）。

评测系统关键约束：

- Build 阶段允许联网（用于下载依赖/权重）；Run 阶段断网（请求路径内不要做任何联网操作）。

- 必须提供：

  - `GET /`：健康检查（需快速返回，评测机先调它）

  - `POST /predict`：推理接口（见下文 API 契约）

- 端口：保持 `8000`。

---

## 2) 快速启动（Docker）

**请保证你已经有如下前置环境：**

主要软件包(vllm:maca.ai3.1.0.7-torch2.6-py310-ubuntu22.04-amd64)版本如下：

|软件|版本|
|-|-|
|python|3.10|
|ubuntu|22.04|
|pytorch|2.6|
|vLLM|0.10.0|

**测试时使用Gitee提供的沐曦算力服务器和系统镜像进行部署，实测没有问题，其他平台上请注意前置依赖是否满足。**

### 2.1 构建镜像（会在 build 阶段下载模型）

```bash
docker build -t metax-llm-public:latest .
```

默认会从 ModelScope 下载 Dockerfile 中配置的模型：

- `MODEL_ID`：模型仓库 ID（例如 `YukinoStuki/...`）

- `MODEL_REVISION`：分支/Tag/commit（默认 `master`）

如需切换模型/版本：修改 Dockerfile 里的 `ENV MODEL_ID=...` / `ENV MODEL_REVISION=...` 后重新 build。

### 2.2 运行容器

本地自测（NVIDIA GPU 环境示例）：

```bash
docker run --rm --gpus all -p 8000:8000 metax-llm-public:latest
```

启动后自测健康检查：

```bash
curl -s http://127.0.0.1:8000/
```

单条推理请求：

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"请简要回答：什么是xxx？"}'
```

### 2.3 API 契约（必须保持不变）

- 评测机将按照如下方式调用服务（**不要破坏**）：

  - 端口：`8000`

  - `GET /`：健康检查

  - `POST /predict`：

    - 请求 JSON：`{"prompt":"..."}`

    - 响应 JSON：`{"response":"..."}`

- 本项目也支持 batch：

  - 当 `BATCH_MODE=1` 时，`GET /` 会返回 `{"status":"batch"}`，评测系统会将所有问题一次性推到 `POST /predict`。

  - batch 请求格式：`{"prompt":["...","...",...]}`

  - batch 响应格式：`{"response":["...","...",...]}`（答案数量需与问题数量一致）

### 2.4 评测机环境（参考配置）

评测机常见配置（沐曦赛道示例）：

- OS：Ubuntu 24.04

- CPU：24 核

- 内存：200GB

- 磁盘：1TB

- GPU：MXC500（64GB 显存）

- 网络：100Mbps

时间限制（模板常用默认）：

- docker build：900s

- health（`GET /`）：180s

- predict（推理阶段总计）：360s

---

## 3) 文件说明（Gitee 同步文件集）

Gitee 侧仅保留以下文件：

- `Dockerfile`

  - 评测容器构建入口：安装依赖、在 build 阶段执行 `download_model.py` 下载权重、设置运行默认环境变量、最终用 `uvicorn` 启动服务。

  - 关键点：Run 阶段断网，因此权重必须在 build 阶段准备好。

- `serve.py`

  - 推理服务核心：FastAPI 应用，提供 `GET /` 与 `POST /predict` 等端点。

  - 主要职责：加载本地模型目录（默认 `MODEL_DIR=./model/$MODEL_ID`）、用 vLLM 优先推理（失败可回退 transformers）、控制输出长度与后处理（避免无效长输出）。

- `download_model.py`

  - 模型下载脚本：在 build 阶段从 ModelScope 下载模型到 `./model/`。

  - 支持（可选）下载 speculative decoding 的 draft 模型；若设置为 optional，draft 下载失败不阻断构建。

- `requirements.txt`

  - 运行所需最小依赖：FastAPI/uvicorn、ModelScope（仅用于 build 下载）、transformers（作为回退）、以及少量兼容性依赖。

- `data.jsonl`

  - 可选的本地数据样例：用于服务启动/预热抽样（是否使用由 `serve.py` 中预热相关环境变量控制）。

  - 注意：评测机同步/裁剪可能影响数据文件是否存在，因此服务不应强依赖它。

- `.dockerignore`

  - Docker 构建忽略规则：减少上下文体积与无关文件拷贝。

- `.gitignore`

  - Git 忽略规则。

- `README.md`

  - 当前文档：专门给 Gitee 精简版使用。

---

## 4) 特别感谢

（留空，后续补充）