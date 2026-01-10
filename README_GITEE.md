# Gitee 同步精简版：MetaX 评测推理服务

说明：本 README 面向“同步到 Gitee 的精简文件集”。Gitee 仓库只保留评测运行必须文件（见下文“文件说明”）；完整研发/调参/工具脚本以 GitHub 仓库为准。

---

## 1) 项目介绍

这是一个“微调开源大模型用于问答评测”的项目：

- 数据：将一本书制作成问答对数据集，用于微调与验证。
- 服务：通过 HTTP 提供推理接口，满足评测系统的 API 契约。
- 目标：在准确率（RougeL-F1，jieba 分词）达到阈值（常见参考 ≥ 0.35）的前提下，尽量提升吞吐（tokens/s）。

评测系统关键约束：

- Build 阶段允许联网（用于下载依赖/权重）；Run 阶段断网（请求路径内不要做任何联网操作）。
- 必须提供：
  - `GET /`：健康检查（需快速返回，评测机先调它）
  - `POST /predict`：推理接口（见下文 API 契约）
- 端口：保持 `8000`。

---

## 2) 快速启动（Docker）

### 2.1 构建镜像（会在 build 阶段下载模型）

```bash
docker build -t metax-demo:latest .
```

默认会从 ModelScope 下载 Dockerfile 中配置的模型：

- `MODEL_ID`：模型仓库 ID（例如 `YukinoStuki/...`）
- `MODEL_REVISION`：分支/Tag/commit（默认 `master`）

如需切换模型/版本：修改 Dockerfile 里的 `ENV MODEL_ID=...` / `ENV MODEL_REVISION=...` 后重新 build。

### 2.2 运行容器

本地自测（NVIDIA GPU 环境示例）：

```bash
docker run --rm --gpus all -p 8000:8000 metax-demo:latest
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

- `GET /`：健康检查
- `POST /predict`：
  - 请求 JSON：`{"prompt":"..."}`
  - 响应 JSON：`{"response":"..."}`

本仓库也支持 batch（用于提速）：

- 当 `BATCH_MODE=1` 时，`GET /` 会返回 `{"status":"batch"}`，评测系统可能将所有问题一次性推到 `POST /predict`。
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
