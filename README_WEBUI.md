# Qwen2.5-0.5B Plus WebUI 使用指南

## 简介

这是一个轻量级的 Gradio WebUI,用于方便地与 Qwen2.5-0.5B Plus 模型进行交互。

**特点**:
- 🪶 **轻量级**: 只需 2 个依赖 (gradio + requests),无需复杂配置
- 🚀 **即插即用**: 直接调用现有的 serve.py API
- 💬 **聊天界面**: 类似 ChatGPT 的对话体验
- ⚡ **性能优化**: 保留 serve.py 的所有优化 (vLLM/batch/预热等)

## 架构

```
┌─────────────────┐
│  浏览器访问      │
│  localhost:7860  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   webui.py      │  ← Gradio 界面
│   (端口 7860)    │
└────────┬────────┘
         │ POST /predict
         ▼
┌─────────────────┐
│   serve.py      │  ← FastAPI 后端
│   (端口 8000)    │  ← vLLM 推理引擎
└─────────────────┘
```

## 快速开始

### 方法一: 使用启动脚本 (推荐)

1. **启动后端** (终端 1):
```bash
# 使用默认配置
./run_model.sh

# 或使用自定义参数
MODEL_ID=YukinoStuki/Qwen2.5-0.5B-Plus-LLM-AWQ MAX_NEW_TOKENS=64 ./run_model.sh
```

2. **启动 WebUI** (终端 2):
```bash
./start_webui.sh
```

3. 打开浏览器访问: http://localhost:7860

### 方法二: 手动启动

```bash
# 1. 安装依赖
pip install -r requirements-webui.txt

# 2. 启动 WebUI
python webui.py
```

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

# 启动
./start_webui.sh
```

## 沐曦云平台部署

### 1. 准备工作

```bash
# SSH 连接到沐曦云主机
ssh user@your-metax-host

# 克隆仓库
cd ~
git clone https://github.com/YukinoStuki2/metax-llm-public.git
cd metax-llm-public

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
pip install -r requirements-webui.txt
```

### 2. 启动服务

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

### 3. 访问 WebUI

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

### 4. 使用 tmux 保持服务运行

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

WebUI 会自动调用 serve.py 的所有优化特性:

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
# 1. 修改 Dockerfile 中的 EXPOSE
# 2. 启动时指定:
uvicorn serve:app --host 0.0.0.0 --port 8001

# 3. WebUI 连接到新端口
API_BASE_URL=http://127.0.0.1:8001 ./start_webui.sh
```

## 故障排查

### 1. WebUI 无法连接后端

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

### 2. 推理速度慢

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

### 3. 内存不足

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

### 4. 权限错误

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

### 扩展功能

可以基于当前架构添加:
- ✅ 流式输出 (需修改 serve.py 支持 SSE)
- ✅ 多轮对话历史
- ✅ 参数调节 (temperature, top_p 等)
- ✅ 文件上传 (文档问答)
- ✅ 图片输入 (需多模态模型)

## 与 Text Generation WebUI 对比

| 特性 | 本 WebUI | Text Gen WebUI |
|------|---------|----------------|
| 安装大小 | < 50 MB | ~10 GB |
| 依赖数量 | 2 个 | 100+ 个 |
| 配置复杂度 | ⭐ 简单 | ⭐⭐⭐⭐ 复杂 |
| 启动速度 | < 5 秒 | 30-60 秒 |
| 与现有代码集成 | ✅ 完美 | ⚠️ 需重构 |
| 性能优化 | ✅ 保留全部 | ⚠️ 部分冲突 |
| 功能丰富度 | ⭐⭐ 基础 | ⭐⭐⭐⭐⭐ 强大 |

**推荐**: 如果你只需要一个简单的聊天界面,使用本 WebUI;如果需要高级功能 (训练/多模型切换/扩展系统等),再考虑 Text Generation WebUI。

## 许可证

本 WebUI 遵循项目主许可证 (AGPL-3.0)。

## 问题反馈

如有问题请提交 Issue 或联系维护者。
