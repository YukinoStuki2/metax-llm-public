---
title: WebUI（Gradio）
sidebar_position: 30
---

WebUI 由 `webui.py` 提供，默认调用本机的 `serve.py`：

- 后端：`API_BASE_URL`（默认 `http://127.0.0.1:8000`）
- WebUI：`WEBUI_HOST`（默认 `0.0.0.0`）、`WEBUI_PORT`（默认 `7860`）

## 启动方式

### 方式 A：直接运行（你自己管理依赖）

```bash
API_BASE_URL=http://127.0.0.1:8000 python3 webui.py
```

### 方式 B：使用 start_webui.sh（推荐）

`start_webui.sh` 会：

- 检查并激活 `VENV_PATH`（默认 `./.venv`）
- 若缺少 gradio，则安装 `requirements-webui.txt`
- 启动 WebUI

```bash
python3 -m venv ./.venv
./start_webui.sh
```

## 与评测的关系

- WebUI **不参与评测**；评测只关心 `serve.py` 的 `/` 与 `/predict`。
- `webui.py` 里包含可选的“RAG/网页抓取/百度搜索”逻辑（使用 `requests.get` 访问 URL）。这在评测机 run 阶段（断网）不可用，也不应在评测链路依赖它。

## 主要功能（与当前实现一致）

- **生成参数面板**：单次请求生效，会把 `max_new_tokens/temperature/top_p/top_k/...` 透传到 `/predict`。
- **系统提示词**：通过后端 `GET /system_prompt` 与 `POST /system_prompt` 读写（影响后续请求的 prompt 组装）。
- **后端信息**：读取 `GET /info` 展示后端信息与白名单环境变量。
- **Batch 测试**：一键运行 `eval_local.py`（固定参数），并在 WebUI 中流式显示输出。
- **RAG（可选）**：本地文件 + URL 抓取；可选百度搜索与 `metax_url.json` 固定 URL 库（默认关闭）。

## 参数透传

WebUI 会把额外生成参数（如果你在 UI 中填了）透传到 `/predict` 请求 JSON 中；服务端会以自身默认值为准处理缺省参数。

## 常用环境变量

- `API_BASE_URL`：后端地址（默认 `http://127.0.0.1:8000`）
- `API_TIMEOUT`：请求超时（默认 360s）
- `WEBUI_HOST` / `WEBUI_PORT`：WebUI 监听地址/端口
- RAG：`RAG_MAX_DOC_BYTES`、`RAG_MAX_URLS`、`RAG_HTTP_TIMEOUT`、`RAG_BAIDU_MAX_RESULTS`、`METAX_URL_DB_PATH`
