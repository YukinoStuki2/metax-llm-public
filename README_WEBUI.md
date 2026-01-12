# Qwen2.5-0.5B Plus WebUI 使用指南

## 简介

这是一个轻量级的 Gradio WebUI,用于方便地与 Qwen2.5-0.5B Plus 模型进行交互。

**特点**:

- 🪶 **轻量级**: 只需 2 个依赖 (gradio + requests),无需复杂配置

- 🚀 **即插即用**: 直接调用现有的 serve.py FastAPI 接口（`GET /`、`POST /predict`）

- 💬 **聊天界面**: Chatbot 对话 + 历史记录

- 🎛️ **可调生成参数**: 在 UI 中按请求透传 `max_new_tokens/temperature/top_p/top_k/...`

- 🧩 **后端管理**: 支持查看 `/info`、读写 `/system_prompt`

- 📚 **RAG（可选）**: 本地文件 + URL 抓取（默认关闭，不影响评测/后端）

- ⚡ **性能优化**: 仍由后端 `serve.py` 负责（vLLM/batch/预热等）

> 说明：WebUI 本身不会改变评测接口契约；评测环境 Run 阶段断网，WebUI 的“联网抓取/百度搜索”仅用于本地/云主机调试。

## 架构

```
┌─────────────────┐
│  浏览器访问      │
│  localhost:7860  │
└────────┬────────┘
         │ HTTP/SSH
         ▼
┌─────────────────┐
│   webui.py      │  ← Gradio 界面
│   (端口 7860)    │
└────────┬────────┘
         │ GET /, GET /info, GET/POST /system_prompt, POST /predict
         ▼
┌─────────────────┐
│   serve.py      │  ← FastAPI 后端
│   (端口 8000)    │  ← vLLM 推理引擎
└─────────────────┘

（可选）RAG 资料来源：本地上传文件 / URL 内容抓取 / 百度搜索 / `metax_url.json` 固定 URL 库。
```

## 快速开始

### 方法一: 使用启动脚本 (推荐)

**1.启动后端** (终端 1):

```bash
# （推荐）导入默认环境变量，避免上一次运行遗留配置影响本次结果
source ./env_force.sh

# 使用默认配置
./run_model.sh

# 或使用自定义参数
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ MAX_NEW_TOKENS=64 ./run_model.sh
```

**2.启动 WebUI** (终端 2):

```bash
./start_webui.sh
```

3.打开浏览器访问: http://localhost:7860

### 方法二: 手动启动

```bash
# 1. 安装依赖
pip install -r requirements-webui.txt

# 2. 启动 WebUI
python webui.py
```

> 提示：`start_webui.sh` 会自动创建虚拟环境并安装 `requirements-webui.txt` + `requirements-eval.txt`（用于 模型测试）。

## 配置

通过环境变量配置 WebUI:

```bash
# 后端 API 地址 (默认: http://127.0.0.1:8000)
export API_BASE_URL="http://127.0.0.1:8000"

# WebUI 监听端口 (默认: 7860)
export WEBUI_PORT=7860

# WebUI 监听地址 (默认: 0.0.0.0, 监听所有网卡)
export WEBUI_HOST="0.0.0.0"

# 是否创建公开分享链接 (默认: 0)
export WEBUI_SHARE=0

# API 请求超时 (默认: 360 秒)
export API_TIMEOUT=360

# RAG（默认关闭；仅 WebUI 侧生效，不影响后端）
# 单个本地文件最大读取字节数（默认: 1000000）
export RAG_MAX_DOC_BYTES=1000000

# URL 抓取的最大总数（默认: 8）
export RAG_MAX_URLS=8

# URL/百度搜索 HTTP 超时（默认: 10）
export RAG_HTTP_TIMEOUT=10

# 百度搜索最大结果数（默认: 5）
export RAG_BAIDU_MAX_RESULTS=5

# 固定 URL 库路径（默认: ./metax_url.json）
export METAX_URL_DB_PATH=./metax_url.json

# 启动
./start_webui.sh
```

### UI 功能说明

- **生成参数**：对每次请求生效，WebUI 会把这些参数以 JSON 透传到 `/predict`。

  - 常用：`max_new_tokens/temperature/top_p/top_k/repetition_penalty/frequency_penalty`。

  - 若后端未实现某些参数，会被忽略（以 `serve.py` 实现为准）。

- **系统提示词**：通过后端接口 `GET /system_prompt` + `POST /system_prompt` 读写（修改后影响后续请求的 prompt 组装）。

- **后端信息**：读取 `GET /info`，展示后端信息与白名单环境变量。

- **Batch 测试**：点击按钮会运行一条固定命令（见下方说明），并在 UI 中流式显示输出。

- 该命令目前固定为：

  - `python eval_local.py --which bonus --model_dir_for_tokenizer ./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM --batch --overwrite_jsonl --debug_first_n 5 --debug_random_n 5`

  - 注意：`--model_dir_for_tokenizer` 路径在 `webui.py` 里是硬编码的；如果你本地模型目录不同，请修改 `webui.py` 的 `run_batch_test()`。

- **RAG**：启用后，会把命中的资料片段拼到最终 prompt 中，并在界面展示“本次命中片段”。

### RAG 使用规则（和代码一致）

- 默认关闭（`启用 RAG` 不勾选时，行为等同普通对话）。

- **本地文件**：上传 `txt/md` 等纯文本文件，WebUI 会按段落切分并做简单关键词重叠召回。

- **联网抓取**：需勾选 `允许联网抓取 URL` 才会访问网络。

  - 若勾选 `使用 www.baidu.com 搜索结果`：将优先使用百度搜索出的 URL（忽略你手填的 URL 列表）。

  - 若不勾选百度搜索：使用你在“URL 列表”里填写的链接；可选叠加 `metax_url.json` 固定 URL 库（默认勾选）。

- `metax_url.json` 期望格式为 JSON 对象，包含 `seed_pages`（list\[dict\]），每项至少含 `url` 字段。

- 限制：最多抓取 `RAG_MAX_URLS` 个 URL；网页内容会被简化为纯文本并做长度截断缓存。

## 沐曦云平台部署

### 1\. 准备工作

```bash
# SSH 连接到沐曦云主机
ssh user@your-metax-host

# 克隆仓库
cd ~
git clone <你的仓库地址>
cd <仓库目录>

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-webui.txt
```

### 2\. 启动服务

**启动后端**:

```bash
# 使用默认配置启动 serve.py
./run_model.sh

# 或使用 AWQ 量化模型 (更快)
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ ./run_model.sh
```

**启动 WebUI** (新终端或使用 tmux):

```bash
cd ~/metax-llm-public
source .venv/bin/activate
./start_webui.sh
```

### 3\. 访问 WebUI

如果你的沐曦云主机有公网 IP:

```
http://your-public-ip:7860
```

如果只有内网 IP,使用 SSH 端口转发:

```bash
# 本地机器执行
ssh -L 7860:localhost:7860 -L 8000:localhost:8000 user@metax-host

# 然后访问本地
http://localhost:7860
```

### 4\. 使用 tmux 保持服务运行

```bash
# 安装 tmux (如果没有)
sudo apt install tmux

# 创建后端会话
tmux new -s backend
source .venv/bin/activate
./run_model.sh
# 按 Ctrl+B 再按 D 脱离会话

# 创建 WebUI 会话
tmux new -s webui
source .venv/bin/activate
./start_webui.sh
# 按 Ctrl+B 再按 D 脱离会话

# 查看所有会话
tmux ls

# 重新连接
tmux attach -t backend
tmux attach -t webui
```

## 高级配置

### 性能优化

WebUI 会调用后端 `serve.py`，因此**性能优化仍建议在后端侧完成**（例如启用 vLLM、batch、量化、预热等）。

同时，WebUI 也提供“生成参数”面板（单次请求生效），适合快速调参观察效果。

```bash
# 启用 AWQ 量化
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ ./run_model.sh

# 调整生成参数
MAX_NEW_TOKENS=128 TEMPERATURE=0.0 ./run_model.sh

# 启用 batch 模式 (适合大量请求)
BATCH_MODE=1 ./run_model.sh
```

### 远程访问

如果需要从其他机器访问 WebUI:

```bash
# 监听所有网卡 (默认已是)
WEBUI_HOST=0.0.0.0 ./start_webui.sh

# 创建 Gradio 公开分享链接 (有 72 小时限制)
WEBUI_SHARE=1 ./start_webui.sh
```

⚠️ **安全提示**: 公开分享时请注意数据安全,建议配合防火墙/反向代理使用。

### 自定义端口

如果端口冲突:

```bash
# 修改 WebUI 端口
WEBUI_PORT=8860 ./start_webui.sh

# 修改后端端口 (需同时修改 serve.py)
# ⚠️ 评测机要求后端固定 8000 端口，请勿为评测目的修改。
# 1. 修改 Dockerfile 中的 EXPOSE
# 2. 启动时指定:
uvicorn serve:app --host 0.0.0.0 --port 8001

# 3. WebUI 连接到新端口
API_BASE_URL=http://127.0.0.1:8001 ./start_webui.sh
```

## 故障排查

### 1\. WebUI 无法连接后端

**问题**: WebUI 显示 "❌ 无法连接到后端"

**解决**:

```bash
# 检查后端是否启动
curl http://127.0.0.1:8000/

# 检查端口占用
sudo netstat -tlnp | grep 8000

# 查看后端日志
# (如果使用 tmux)
tmux attach -t backend
```

### 2\. 推理速度慢

**问题**: 每次推理需要很长时间

**解决**:

```bash
# 1. 使用 AWQ 量化模型
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ ./run_model.sh

# 2. 减少 max_new_tokens
MAX_NEW_TOKENS=32 ./run_model.sh

# 3. 检查 GPU 利用率
nvidia-smi

# 4. 启用 vLLM (默认已启用)
USE_VLLM=true ./run_model.sh
```

### 3\. 内存不足

**问题**: OOM (Out of Memory)

**解决**:

```bash
# 1. 降低 GPU 内存占用
GPU_MEMORY_UTILIZATION=0.70 ./run_model.sh

# 2. 减少最大序列长度
MAX_MODEL_LEN=4096 ./run_model.sh

# 3. 使用量化模型
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ ./run_model.sh
```

### 4\. 权限错误

**问题**: `Permission denied` 或无法启动

**解决**:

```bash
# 给脚本添加执行权限
chmod +x start_webui.sh run_model.sh

# 检查虚拟环境
ls -la .venv/bin/python

# 重新创建虚拟环境
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-webui.txt
```

## 开发说明

### 修改 WebUI

编辑 `webui.py` 可以自定义界面:

```python
# 修改标题
gr.Markdown("# 🤖 我的自定义 WebUI")

# 修改默认端口
WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "8860"))

# 添加更多参数控制
temperature = gr.Slider(0, 1, value=0, label="Temperature")
```

## 许可证

本 WebUI 遵循项目主许可证 (Apache-2.0)。

## 问题反馈

如有问题请提交 Issue 或联系维护者。