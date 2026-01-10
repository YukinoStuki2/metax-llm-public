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
- `webui.py` 里包含可选的“RAG/网页抓取”逻辑（使用 `requests.get` 访问 URL）。这在评测机 run 阶段（断网）不可用，也不应在评测链路依赖它。

## 参数透传

WebUI 会把额外生成参数（如果你在 UI 中填了）透传到 `/predict` 请求 JSON 中；服务端会以自身默认值为准处理缺省参数。
