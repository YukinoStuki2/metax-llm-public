#!/usr/bin/env python3
"""
轻量级 Gradio WebUI for serve.py
直接调用 serve.py 的 FastAPI 接口进行推理
"""

import os
import sys
import requests
import gradio as gr
from typing import Optional, Iterator

# 配置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "360"))
WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "7860"))
WEBUI_HOST = os.environ.get("WEBUI_HOST", "0.0.0.0")
WEBUI_SHARE = os.environ.get("WEBUI_SHARE", "0") == "1"


def check_api_health() -> tuple[bool, str]:
    """检查后端 API 健康状态"""
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=10)
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "ok")
            return True, f"✅ 后端状态: {status}"
        else:
            return False, f"❌ 后端返回错误: HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, f"❌ 无法连接到后端 {API_BASE_URL}"
    except Exception as e:
        return False, f"❌ 健康检查失败: {str(e)}"


def predict(user_input: str, history: Optional[list] = None) -> Iterator[str]:
    """调用后端 API 进行推理"""
    if not user_input or not user_input.strip():
        yield "⚠️ 请输入问题"
        return

    # 检查 API 可用性
    is_healthy, health_msg = check_api_health()
    if not is_healthy:
        yield health_msg
        return

    try:
        # 调用 /predict 接口
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"prompt": user_input.strip()},
            timeout=API_TIMEOUT,
        )
        response.raise_for_status()

        result = response.json()
        answer = result.get("response", "")

        if not answer:
            yield "⚠️ 模型返回了空答案"
        else:
            yield answer

    except requests.exceptions.Timeout:
        yield f"❌ 请求超时 (>{API_TIMEOUT}s)"
    except requests.exceptions.RequestException as e:
        yield f"❌ 请求失败: {str(e)}"
    except Exception as e:
        yield f"❌ 推理出错: {str(e)}"


def create_ui():
    """创建 Gradio 界面"""
    # 检查后端状态
    is_healthy, health_status = check_api_health()

    with gr.Blocks(
        title="Qwen3-4B Plus WebUI",
    ) as demo:
        gr.Markdown(
            f"""
# 🤖 Qwen3-4B Plus WebUI

**后端地址**: `{API_BASE_URL}`  
**状态**: {health_status}

---
"""
        )

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="对话历史",
                    height=500,
                )
                user_input = gr.Textbox(
                    label="输入问题",
                    placeholder="请输入你的问题...",
                    lines=3,
                    max_lines=10,
                )

                with gr.Row():
                    submit_btn = gr.Button("🚀 发送", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ 清空", scale=1)

            with gr.Column(scale=3):
                gr.Markdown("### ℹ️ 使用说明")
                gr.Markdown(
                    """
1. 在输入框输入问题
2. 点击「发送」或按 Enter
3. 等待模型推理完成
4. 查看回答

**注意事项**:
- 当前为非流式模式
- 超时时间: {timeout}s
- 模型: Qwen3-4B-Plus-LLM
""".format(
                        timeout=API_TIMEOUT
                    )
                )

                # 添加后端信息
                gr.Markdown("### 🔧 后端配置")
                backend_info = gr.Textbox(
                    label="API 地址",
                    value=API_BASE_URL,
                    interactive=False,
                )
                health_btn = gr.Button("🔄 检查健康状态")
                health_output = gr.Textbox(
                    label="健康状态",
                    value=health_status,
                    interactive=False,
                )

        # 事件处理
        def user_submit(user_msg, history):
            """处理用户提交"""
            if not history:
                history = []
            history.append([user_msg, None])
            return "", history

        def bot_respond(history):
            """处理机器人回复"""
            if not history or history[-1][1] is not None:
                return history

            user_msg = history[-1][0]
            bot_msg = ""

            # 调用 predict 并逐步更新
            for response in predict(user_msg):
                bot_msg = response
                history[-1][1] = bot_msg
                yield history

        def clear_history():
            """清空对话历史"""
            return [], ""

        def refresh_health():
            """刷新健康状态"""
            _, status = check_api_health()
            return status

        # 绑定事件
        submit_btn.click(
            user_submit, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(bot_respond, chatbot, chatbot)

        user_input.submit(
            user_submit, [user_input, chatbot], [user_input, chatbot], queue=False
        ).then(bot_respond, chatbot, chatbot)

        clear_btn.click(clear_history, None, [chatbot, user_input], queue=False)

        health_btn.click(refresh_health, None, health_output)

    return demo


def main():
    """启动 WebUI"""
    print("=" * 60)
    print("🚀 启动 Qwen3-4B Plus WebUI")
    print(f"后端 API: {API_BASE_URL}")
    print(f"监听地址: {WEBUI_HOST}:{WEBUI_PORT}")
    print(f"公开分享: {'是' if WEBUI_SHARE else '否'}")
    print("=" * 60)

    # 检查后端可用性
    is_healthy, health_msg = check_api_health()
    if not is_healthy:
        print(f"\n⚠️  警告: {health_msg}")
        print("请确保 serve.py 已启动并监听在", API_BASE_URL)
        print("\n继续启动 WebUI (后端可以稍后启动)...\n")

    demo = create_ui()
    demo.launch(
        server_name=WEBUI_HOST,
        server_port=WEBUI_PORT,
        share=WEBUI_SHARE,
    )


if __name__ == "__main__":
    main()
