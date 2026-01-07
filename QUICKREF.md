# 🚀 快速参考卡

## 一键部署 (沐曦云平台)

```bash
# 1. 克隆项目
git clone https://github.com/YukinoStuki2/metax-demo-mirror.git
cd metax-demo-mirror

# 2. 创建环境
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-webui.txt

# 3. 启动后端 (终端1 或 tmux)
./run_model.sh

# 4. 启动 WebUI (终端2 或 tmux)
./start_webui.sh

# 5. 访问 http://localhost:7860
```

## 常用命令

### tmux 会话管理
```bash
tmux new -s backend      # 创建后端会话
tmux new -s webui        # 创建 WebUI 会话
tmux ls                  # 列出所有会话
tmux attach -t backend   # 连接到后端
Ctrl+B, D                # 脱离会话
tmux kill-session -t backend  # 关闭会话
```

### 性能调优
```bash
# AWQ 量化 (更快)
MODEL_ID=YukinoStuki/Qwen3-4B-Plus-LLM-AWQ ./run_model.sh

# 降低内存占用
GPU_MEMORY_UTILIZATION=0.70 ./run_model.sh

# 调整生成长度
MAX_NEW_TOKENS=64 ./run_model.sh
```

### 端口转发 (本地访问云主机)
```bash
ssh -L 7860:localhost:7860 -L 8000:localhost:8000 user@host
```

### 健康检查
```bash
curl http://localhost:8000/  # 后端状态
nvidia-smi                    # GPU 状态
tmux ls                       # 查看会话
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `serve.py` | FastAPI 推理后端 (vLLM) |
| `webui.py` | Gradio Web 界面 |
| `run_model.sh` | 启动推理后端 |
| `start_webui.sh` | 启动 WebUI |
| `README_WEBUI.md` | WebUI 详细文档 |
| `DEPLOY.md` | 部署指南 |

## 故障排查速查

| 问题 | 解决方案 |
|------|---------|
| OOM 错误 | `GPU_MEMORY_UTILIZATION=0.60 ./run_model.sh` |
| 推理太慢 | 使用 AWQ 模型 |
| 端口冲突 | `WEBUI_PORT=8860 ./start_webui.sh` |
| 无法连接后端 | 检查 `curl http://localhost:8000/` |
| 权限错误 | `chmod +x *.sh` |

## 环境变量速查

### 后端 (serve.py)
- `MODEL_ID`: 模型路径
- `MAX_NEW_TOKENS`: 生成长度
- `GPU_MEMORY_UTILIZATION`: GPU 内存占用
- `BATCH_MODE`: 批处理模式

### WebUI
- `API_BASE_URL`: 后端地址 (默认: http://127.0.0.1:8000)
- `WEBUI_PORT`: 监听端口 (默认: 7860)
- `WEBUI_HOST`: 监听地址 (默认: 0.0.0.0)
- `API_TIMEOUT`: 请求超时 (默认: 360s)

## 更多信息

📖 完整文档: [README_WEBUI.md](README_WEBUI.md)  
🚀 部署指南: [DEPLOY.md](DEPLOY.md)  
🐛 问题反馈: GitHub Issues
