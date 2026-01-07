# 沐曦云平台部署指南

本指南帮助你在沐曦 C500 (64GB) 云主机上部署 Qwen3-4B Plus WebUI。

## 前置要求

- ✅ 沐曦云主机 (C500, 64GB 显存)
- ✅ Ubuntu 22.04 (无桌面)
- ✅ SSH 访问权限
- ✅ Git 已安装

## 部署步骤

### 1. SSH 连接到云主机

```bash
ssh your-username@your-metax-host-ip
```

### 2. 克隆项目

```bash
cd ~
git clone https://github.com/YukinoStuki2/metax-demo-mirror.git
cd metax-demo-mirror
```

### 3. 创建 Python 虚拟环境

```bash
# 安装 venv (如果需要)
sudo apt update
sudo apt install -y python3.12-venv python3-pip

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 升级 pip
python -m pip install -U pip setuptools wheel
```

### 4. 安装依赖

```bash
# 安装推理后端依赖
pip install -r requirements.txt

# 安装 WebUI 依赖
pip install -r requirements-webui.txt
```

### 5. 下载模型 (如果需要)

```bash
# 下载基础模型
python download_model.py

# 或者使用已有的模型
# 确保 model/ 目录下有模型文件
ls -la model/YukinoStuki/
```

### 6. 使用 tmux 启动服务

tmux 可以让服务在后台持续运行,即使断开 SSH 也不会停止。

#### 安装 tmux

```bash
sudo apt install tmux
```

#### 启动推理后端

```bash
# 创建后端会话
tmux new -s backend

# 在 tmux 会话中执行:
cd ~/metax-demo-mirror
source .venv/bin/activate
./run_model.sh

# 按 Ctrl+B, 再按 D 可以脱离会话(服务继续运行)
```

#### 启动 WebUI

```bash
# 创建 WebUI 会话
tmux new -s webui

# 在 tmux 会话中执行:
cd ~/metax-demo-mirror
source .venv/bin/activate
./start_webui.sh

# 按 Ctrl+B, 再按 D 可以脱离会话
```

### 7. 访问 WebUI

#### 方法 A: 云主机有公网 IP

直接访问:
```
http://your-public-ip:7860
```

#### 方法 B: 使用 SSH 端口转发

在**本地机器**执行:
```bash
ssh -L 7860:localhost:7860 -L 8000:localhost:8000 your-username@your-metax-host-ip
```

然后访问本地:
```
http://localhost:7860
```

### 8. 验证服务状态

```bash
# 查看 tmux 会话列表
tmux ls

# 重新连接到后端会话
tmux attach -t backend

# 重新连接到 WebUI 会话
tmux attach -t webui

# 检查后端健康状态
curl http://localhost:8000/

# 检查进程
ps aux | grep python
```

## 常用 tmux 命令

```bash
# 列出所有会话
tmux ls

# 创建新会话
tmux new -s session-name

# 连接到会话
tmux attach -t session-name

# 脱离会话 (服务继续运行)
# 按 Ctrl+B, 再按 D

# 关闭会话
tmux kill-session -t session-name

# 在会话间切换
# 按 Ctrl+B, 再按 S
```

## 性能优化

### 使用 AWQ 量化模型 (推荐)

AWQ 量化可以显著提升推理速度:

```bash
# 修改启动配置
MODEL_ID=YukinoStuki/Qwen3-4B-Plus-LLM-AWQ ./run_model.sh
```

### 调整 GPU 内存占用

如果遇到 OOM:

```bash
GPU_MEMORY_UTILIZATION=0.70 ./run_model.sh
```

### 减少生成长度

对于短回答场景:

```bash
MAX_NEW_TOKENS=32 ./run_model.sh
```

## 故障排查

### 1. 后端无法启动

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 环境
python -c "import torch; print(torch.cuda.is_available())"

# 查看详细日志
tmux attach -t backend
```

### 2. WebUI 无法连接后端

```bash
# 检查后端是否运行
curl http://localhost:8000/

# 检查端口占用
sudo netstat -tlnp | grep 8000

# 检查防火墙
sudo ufw status
```

### 3. 模型文件缺失

```bash
# 检查模型目录
ls -la model/YukinoStuki/

# 重新下载
python download_model.py
```

### 4. 权限错误

```bash
# 给脚本添加执行权限
chmod +x run_model.sh start_webui.sh

# 检查虚拟环境
source .venv/bin/activate
which python
```

## 更新项目

```bash
cd ~/metax-demo-mirror

# 停止服务
tmux kill-session -t backend
tmux kill-session -t webui

# 拉取最新代码
git pull origin master

# 更新依赖
source .venv/bin/activate
pip install -r requirements.txt --upgrade
pip install -r requirements-webui.txt --upgrade

# 重新启动服务
tmux new -s backend
# ... (重复启动步骤)
```

## 自动启动 (可选)

如果需要开机自动启动,可以创建 systemd 服务:

```bash
# 创建后端服务
sudo nano /etc/systemd/system/qwen-backend.service
```

内容:
```ini
[Unit]
Description=Qwen3-4B Plus Backend
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/metax-demo-mirror
Environment="PATH=/home/your-username/metax-demo-mirror/.venv/bin"
ExecStart=/home/your-username/metax-demo-mirror/.venv/bin/python serve.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# 创建 WebUI 服务
sudo nano /etc/systemd/system/qwen-webui.service
```

内容:
```ini
[Unit]
Description=Qwen3-4B Plus WebUI
After=qwen-backend.service
Requires=qwen-backend.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/home/your-username/metax-demo-mirror
Environment="PATH=/home/your-username/metax-demo-mirror/.venv/bin"
Environment="API_BASE_URL=http://127.0.0.1:8000"
ExecStart=/home/your-username/metax-demo-mirror/.venv/bin/python webui.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启用服务:
```bash
sudo systemctl daemon-reload
sudo systemctl enable qwen-backend qwen-webui
sudo systemctl start qwen-backend qwen-webui

# 查看状态
sudo systemctl status qwen-backend
sudo systemctl status qwen-webui
```

## 安全建议

1. **不要暴露到公网**:除非配置了认证/防火墙
2. **使用 SSH 密钥**:而不是密码登录
3. **定期更新**:保持系统和依赖最新
4. **监控资源**:使用 `nvidia-smi` 监控 GPU 使用

## 需要帮助?

- 📖 查看 [README_WEBUI.md](README_WEBUI.md) 获取更多配置选项
- 🐛 提交 Issue 到 GitHub
- 📧 联系维护者

---

**提示**: 首次启动推理后端可能需要 1-2 分钟下载/加载模型,请耐心等待!
