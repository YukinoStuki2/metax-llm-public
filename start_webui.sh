#!/bin/bash
# WebUI 启动脚本 (适用于沐曦云平台)

set -euo pipefail

# 配置区
VENV_PATH="${VENV_PATH:-./.venv}"
API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
WEBUI_PORT="${WEBUI_PORT:-7860}"
WEBUI_HOST="${WEBUI_HOST:-0.0.0.0}"
WEBUI_SHARE="${WEBUI_SHARE:-0}"

echo "========================================"
echo "🌐 启动 Qwen3-4B Plus WebUI"
echo "========================================"
echo "虚拟环境: ${VENV_PATH}"
echo "后端 API: ${API_BASE_URL}"
echo "监听地址: ${WEBUI_HOST}:${WEBUI_PORT}"
echo "========================================"

# 检查虚拟环境
if [[ ! -d "${VENV_PATH}" ]]; then
    echo "❌ 错误: 虚拟环境不存在: ${VENV_PATH}"
    echo "请先运行: python3 -m venv ${VENV_PATH}"
    exit 1
fi

# 激活虚拟环境
source "${VENV_PATH}/bin/activate"

# 检查并安装 WebUI 依赖
if ! python -c "import gradio" &>/dev/null; then
    echo "📦 安装 WebUI 依赖..."
    pip install -r requirements-webui.txt --upgrade
fi

# 检查后端是否启动
echo ""
echo "🔍 检查后端状态..."
if curl -s --max-time 5 "${API_BASE_URL}/" >/dev/null 2>&1; then
    echo "✅ 后端已就绪"
else
    echo "⚠️  后端未启动或无法访问"
    echo "请在另一个终端启动 serve.py:"
    echo "  ./run_model.sh"
    echo ""
    echo "继续启动 WebUI (可以稍后启动后端)..."
fi

echo ""
echo "🚀 启动 WebUI..."
export API_BASE_URL WEBUI_PORT WEBUI_HOST WEBUI_SHARE

python webui.py

echo ""
echo "✅ WebUI 已停止"
