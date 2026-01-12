---
title: WebUI（Gradio）
sidebar_position: 30
---

本页面向 `webui.py`：一个轻量的 Gradio WebUI，用于**本地/自测**时方便调用后端推理服务（`serve.py`）。

文档结构：

1. 用法 / 作用 / 输入输出 / 副作用
2. 参数与环境变量详解
3. 代码详解
4. 常见问题（FAQ）

---

## 1) 用法 / 作用 / 输入输出 / 副作用

### 作用与定位

- WebUI 本身不做推理：它通过 HTTP 调用后端 `serve.py` 的接口（主要是 `GET /` 与 `POST /predict`）。
- WebUI 只用于交互与自测：评测只关心后端 `serve.py`，不会启动/访问 WebUI。

### 输入与输出

- 输入：你在浏览器里输入的文本（以及可选的生成参数、可选 RAG 资料）。
- 输出：后端 `/predict` 返回的 `response` 字段；在 UI 的 Chatbot 中展示。

### 副作用（重要）

- 会对后端发起请求：`GET /`、`POST /predict`，以及可选的 `GET/POST /system_prompt`、`GET /info`。
- 会读本地文件（当启用 RAG 且上传文件时）。
- 可能会访问公网（当启用 RAG 且允许联网时），包括抓取 URL、访问 `www.baidu.com` 搜索页。
- 会启动子进程运行评测脚本（当点击“Batch 测试”时）：调用 `eval_local.py` 并将输出流式展示。

### 启动方式

方式 A：直接运行（你自行管理依赖）

```bash
API_BASE_URL=http://127.0.0.1:8000 \
WEBUI_HOST=0.0.0.0 \
WEBUI_PORT=7860 \
python3 webui.py
```

方式 B：使用 `start_webui.sh`（推荐本地快速拉起）

`start_webui.sh` 会：

- 创建/激活虚拟环境 `VENV_PATH`（默认 `./.venv`）
- 安装依赖：`requirements-webui.txt` 与 `requirements-eval.txt`
- 检查后端是否可访问（仅提示，不会阻止 WebUI 启动）
- 启动 `webui.py`

```bash
python3 -m venv ./.venv
./start_webui.sh
```

更详细的脚本说明见：[启动脚本（start_webui.sh）](./start-webui.md)。

---

## 2) 参数与环境变量详解

### WebUI 侧环境变量（`webui.py` 读取）

- `API_BASE_URL`
	- 默认：`http://127.0.0.1:8000`
	- 作用：后端地址；WebUI 会请求 `${API_BASE_URL}/`、`${API_BASE_URL}/predict` 等。
- `API_TIMEOUT`
	- 默认：`360`（秒）
	- 作用：`/predict` 请求超时。
- `WEBUI_HOST`
	- 默认：`0.0.0.0`
	- 作用：Gradio 监听地址。
- `WEBUI_PORT`
	- 默认：`7860`
	- 作用：Gradio 监听端口。
- `WEBUI_SHARE`
	- 默认：`0`（仅当值为字符串 `"1"` 时为真）
	- 作用：传给 `gradio.Blocks.launch(share=...)`；用于生成外网分享链接（是否可用取决于运行环境）。

### RAG 相关环境变量（仅 WebUI 使用；默认不启用）

注意：RAG 的“是否启用/是否联网”由 UI 里的勾选框控制；下面这些环境变量是容量与超时限制。

- `RAG_MAX_DOC_BYTES`
	- 默认：`1000000`（约 1MB）
	- 作用：上传本地文件超过该大小将被忽略，避免 UI 卡顿。
- `RAG_MAX_URLS`
	- 默认：`8`
	- 作用：联网抓取时最多处理多少个 URL（包含手动 URL、metax_url.json 选出的 URL、或百度搜索结果）。
- `RAG_HTTP_TIMEOUT`
	- 默认：`10`（秒）
	- 作用：抓取网页/百度搜索的 HTTP 超时。
- `RAG_BAIDU_MAX_RESULTS`
	- 默认：`5`
	- 作用：启用百度搜索时最多取多少个链接（之后仍会受 `RAG_MAX_URLS` 再次截断）。
- `METAX_URL_DB_PATH`
	- 默认：`./metax_url.json`
	- 作用：固定 URL 库路径；启用“使用 metax_url.json 固定URL库”时会读取。

### UI 里可透传到后端的生成参数

WebUI 会把下列字段（若非 `None`）透传给 `/predict` 的 JSON：

- `max_new_tokens`
- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `frequency_penalty`

说明：后端是否采纳、如何裁剪/兜底，以 `serve.py` 的实现为准。

---

## 3) 代码详解

这一节按 `webui.py` 的真实实现梳理关键路径，方便你排查“为什么 UI 行为和预期不一致”。

### 3.1 基本推理链路（不含 RAG）

1. `check_api_health()`
	 - `GET ${API_BASE_URL}/`，期望返回 JSON，读取 `status` 字段。
	 - 用途：在推理前做一次快速健康检查。
2. `predict(user_input, gen_params)`
	 - 构造 `payload={"prompt": ...}`，把 UI 生成参数字段（非 `None`）追加进去。
	 - `POST ${API_BASE_URL}/predict`，读取返回 JSON 的 `response`。
	 - 以 generator 方式 `yield`，但当前实现是“单次产出最终答案”，不做 token streaming。

### 3.2 RAG（可选增强，发生在 WebUI 侧）

入口是 `build_rag_context(...)`，它返回：

- `augmented_prompt`：最终发给后端的 prompt（在原问题后拼上“参考资料”和“回答要求”）。
- `display_text`：UI 用来展示“本次命中了哪些资料片段”。

RAG 的资料来源与开关逻辑：

- `enable_rag` 为假：直接返回原 query，不做任何事。
- 本地文件：从上传的 txt/md 里读文本（超过 `RAG_MAX_DOC_BYTES` 会跳过），按段落切块后做“轻量检索”。
- 联网抓取：只有当 UI 勾选 `allow_network` 时才会走。
	- 若 `use_baidu_search` 为真：访问 `https://www.baidu.com/s?wd=...` 抓取搜索页并提取链接。
	- 否则：使用手动输入的 URL 列表；如果同时勾选 `use_metax_url_db`，还会从 `metax_url.json` 中按重叠得分挑选一批 URL 追加进去。
	- URL 抓取内容会做简单的 HTML 清洗（去 script/style、去标签、压缩空白），并按 URL 缓存。

轻量检索方法（不依赖 jieba）：

- `_tokenize_for_retrieval()`：英文/数字按词，中文按单字。
- `_score_overlap()`：计算 query token 与 chunk token 的交集数量，并做简单长度归一。

### 3.3 “Batch 测试”按钮做了什么

`run_batch_test()` 会启动子进程执行固定命令：

```bash
python eval_local.py --which bonus \
	--model_dir_for_tokenizer ./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM \
	--batch --overwrite_jsonl --debug_first_n 5 --debug_random_n 5
```

- 它会流式读取 stdout 并在 UI 中滚动展示（有最大字符数裁剪）。
- 该测试会真实调用后端 `/predict`（依赖后端已启动且 batch 模式可用）。

### 3.4 “系统提示词 / 后端信息”

- 系统提示词：调用后端 `GET /system_prompt` 读取，`POST /system_prompt` 写入。
- 后端信息：调用 `GET /info` 获取 JSON，并把其中的 `env` 映射成表格展示。

---

## 4) 常见问题（FAQ）

### Q1：评测会用到 WebUI 吗？

不会。评测只关心后端 `serve.py` 的 `GET /` 与 `POST /predict`，WebUI 不参与。

### Q2：评测机 run 阶段断网，WebUI 的 RAG 会不会影响成绩？

不会直接影响成绩（因为 WebUI 不参与评测），但如果你在本地复现评测时通过 WebUI 触发了联网抓取/百度搜索，在断网环境会失败。

建议：断网环境下不要勾选“允许联网抓取 URL / 使用百度搜索”。

### Q3：我点“发送”提示无法连接后端怎么办？

确认后端服务已启动且可访问：

- 后端启动：`./run_model.sh`
- WebUI 侧 `API_BASE_URL` 是否指向正确地址（例如远端机器、容器端口映射等）。

### Q4：RAG 开了但命不中任何资料？

常见原因：

- 上传文件不是纯文本（或文件太大超过 `RAG_MAX_DOC_BYTES` 被忽略）。
- query 太短或检索词与资料完全不重叠（当前检索是非常轻量的 overlap 打分）。
- 联网未开启，且未提供 URL，也未启用 metax_url.json。

### Q5：为什么 UI 里改了生成参数，但回答看不出变化？

WebUI 只负责把参数透传给 `/predict`；最终是否生效由后端实现决定。

如果你怀疑后端忽略了参数，可以在后端日志里打印收到的请求字段（或临时在 WebUI 侧打印 payload）来确认。
