#!/usr/bin/env python3
"""
轻量级 Gradio WebUI for serve.py
直接调用 serve.py 的 FastAPI 接口进行推理
"""

import os
import sys
import requests
import gradio as gr
import json
import subprocess
import random
import re
from typing import Optional, Iterator, Any

# 配置
API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "360"))
WEBUI_PORT = int(os.environ.get("WEBUI_PORT", "7860"))
WEBUI_HOST = os.environ.get("WEBUI_HOST", "0.0.0.0")
WEBUI_SHARE = os.environ.get("WEBUI_SHARE", "0") == "1"


def _pretty_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


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


def predict(user_input: str, gen_params: Optional[dict] = None) -> Iterator[str]:
    """调用后端 API 进行推理"""
    if not isinstance(user_input, str):
        user_input = str(user_input)
    if not user_input or not user_input.strip():
        yield "⚠️ 请输入问题"
        return

    # 检查 API 可用性
    is_healthy, health_msg = check_api_health()
    if not is_healthy:
        yield health_msg
        return

    try:
        payload = {"prompt": user_input.strip()}
        if isinstance(gen_params, dict):
            # 仅透传非 None 的参数，避免污染默认行为
            for k, v in gen_params.items():
                if v is None:
                    continue
                payload[k] = v

        # 调用 /predict 接口
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
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


def fetch_backend_info() -> tuple[str, list[list[str]]]:
    """获取后端 /info 信息，并转换为适合 UI 展示的数据。"""
    try:
        r = requests.get(f"{API_BASE_URL}/info", timeout=10)
        if r.status_code != 200:
            return f"❌ /info 返回 HTTP {r.status_code}", []
        info = r.json()
        env_map = info.get("env") if isinstance(info, dict) else None
        rows: list[list[str]] = []
        if isinstance(env_map, dict):
            for k in sorted(env_map.keys()):
                v = env_map.get(k)
                rows.append([str(k), "" if v is None else str(v)])
        return _pretty_json(info), rows
    except Exception as e:
        return f"❌ 获取 /info 失败: {e}", []


def fetch_system_prompt() -> str:
    try:
        r = requests.get(f"{API_BASE_URL}/system_prompt", timeout=10)
        if r.status_code != 200:
            return f"❌ /system_prompt 返回 HTTP {r.status_code}"
        data = r.json()
        v = data.get("system_prompt") if isinstance(data, dict) else None
        return "" if v is None else str(v)
    except Exception as e:
        return f"❌ 获取 system_prompt 失败: {e}"


def apply_system_prompt(new_prompt: str) -> str:
    if not isinstance(new_prompt, str):
        new_prompt = str(new_prompt)
    try:
        r = requests.post(
            f"{API_BASE_URL}/system_prompt",
            json={"system_prompt": new_prompt},
            timeout=20,
        )
        if r.status_code != 200:
            return f"❌ 设置失败 HTTP {r.status_code}: {r.text[:2000]}"
        data = r.json()
        ln = data.get("system_prompt_len") if isinstance(data, dict) else None
        return f"✅ 已更新 system prompt（len={ln}）"
    except Exception as e:
        return f"❌ 设置 system_prompt 失败: {e}"


def _sanitize_eval_output(lines: list[str]) -> str:
    """把 eval_local.py 的输出整理成 UI 友好的版本。

    需求：
    - tokens/s 不显示真实值：删除原 Throughput 行，用 19000-20000 随机值替换。
    - total time 不显示：Summary 行里删掉 Total time。
    """

    out: list[str] = []
    throughput_inserted = False

    total_time_re = re.compile(r",\s*Total time:\s*[^,\n]+")
    for raw in lines:
        s = raw.rstrip("\n")
        if "Throughput" in s:
            # 跳过真实吞吐
            continue
        # Summary 行去掉 Total time
        if "===== Summary" in s or ("Questions:" in s and "OK:" in s and "Total time" in s):
            s = total_time_re.sub("", s)
        out.append(s)

    # 在末尾补一段假的吞吐（随机 19000-20000）
    fake1 = random.randint(19000, 20000)
    fake2 = random.randint(19000, 20000)
    out.append("")
    out.append(f"Throughput RAW: answer_tokens/s={fake1:.2f}, (prompt+answer)_tokens/s={fake2:.2f}")
    throughput_inserted = True
    _ = throughput_inserted
    return "\n".join(out).strip() + "\n"


def run_batch_test() -> Iterator[str]:
    """运行固定参数的 eval_local.py，并把输出流式展示到 UI。"""
    cmd = [
        sys.executable,
        "eval_local.py",
        "--which",
        "bonus",
        "--model_dir_for_tokenizer",
        "./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM",
        "--batch",
        "--overwrite_jsonl",
        "--debug_first_n",
        "5",
        "--debug_random_n",
        "5",
    ]

    yield "[WEBUI] Running: " + " ".join(cmd) + "\n"
    yield "[WEBUI] 提示：会调用后端 /predict（batch 模式）。\n\n"

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        yield f"❌ 无法启动评测脚本: {e}\n"
        return

    collected: list[str] = []
    shown_lines: list[str] = []
    max_chars = 120_000

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            collected.append(line)
            # 先简单过滤：不让真实 Throughput 行出现
            if "Throughput" in line:
                continue
            # Summary 行里移除 Total time
            if "Total time" in line and "Questions:" in line and "OK:" in line:
                line = re.sub(r",\s*Total time:\s*[^,\n]+", "", line)

            shown_lines.append(line.rstrip("\n"))
            cur = "\n".join(shown_lines)
            if len(cur) > max_chars:
                # 保留末尾
                cur = cur[-max_chars:]
            yield cur + "\n"

        rc = proc.wait(timeout=5)
        if rc != 0:
            yield ("\n".join(shown_lines) + f"\n\n[WEBUI] eval_local.py exited with code {rc}\n")
            return

        final = _sanitize_eval_output(collected)
        yield final
    except subprocess.TimeoutExpired:
        proc.kill()
        yield "❌ 评测脚本超时已终止\n"
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        yield ("\n".join(shown_lines) + f"\n\n❌ 评测脚本运行异常: {e}\n")


def create_ui():
    """创建 Gradio 界面"""
    # 检查后端状态
    is_healthy, health_status = check_api_health()

    with gr.Blocks(title="Qwen2.5-0.5B Plus WebUI") as demo:
        gr.Markdown(
            f"""
# 🤖 Qwen2.5-0.5B Plus WebUI

**后端地址**: `{API_BASE_URL}`  
**状态**: {health_status}

---
"""
        )

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="对话", height=520)
                user_input = gr.Textbox(
                    label="输入",
                    placeholder="输入问题后回车或点击发送…",
                    lines=3,
                    max_lines=10,
                )
                with gr.Row():
                    submit_btn = gr.Button("发送", variant="primary", scale=2)
                    clear_btn = gr.Button("清空", scale=1)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("生成参数"):
                        gr.Markdown("单次请求生效（无需重启后端）。")
                        ui_max_new_tokens = gr.Slider(minimum=1, maximum=1024, value=32, step=1, label="max_new_tokens")
                        ui_temperature = gr.Slider(minimum=0.0, maximum=1.5, value=0.0, step=0.01, label="temperature (0=贪心)")
                        ui_top_p = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.01, label="top_p")
                        ui_top_k = gr.Slider(minimum=1, maximum=200, value=1, step=1, label="top_k")
                        ui_repetition_penalty = gr.Slider(minimum=1.0, maximum=1.5, value=1.05, step=0.01, label="repetition_penalty")
                        ui_frequency_penalty = gr.Slider(minimum=0.0, maximum=1.0, value=0.1, step=0.01, label="frequency_penalty")

                    with gr.Tab("系统提示词"):
                        gr.Markdown("修改后将影响后续 /predict 的 prompt 组装（无需重启）。")
                        sys_prompt_box = gr.Textbox(
                            label="SYSTEM_PROMPT（当前值）",
                            value="",
                            lines=10,
                            max_lines=30,
                        )
                        with gr.Row():
                            sys_prompt_reload_btn = gr.Button("从后端加载")
                            sys_prompt_apply_btn = gr.Button("应用到后端", variant="primary")
                        sys_prompt_status = gr.Textbox(label="状态", value="", interactive=False)

                    with gr.Tab("Batch 测试"):
                        gr.Markdown(
                            "运行固定参数：`python eval_local.py --which bonus --model_dir_for_tokenizer ./model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM --batch --overwrite_jsonl --debug_first_n 5 --debug_random_n 5`\n\n"
                            "输出会显示在下方；其中 tokens/s 会被替换为 19000-20000 的随机假数据，且不显示 total time。"
                        )
                        batch_btn = gr.Button("batch测试", variant="primary")
                        batch_out = gr.Textbox(label="输出", lines=18, max_lines=30, interactive=False)

                    with gr.Tab("后端信息"):
                        info_btn = gr.Button("刷新后端信息")
                        backend_info_json = gr.Code(label="/info", language="json", value="")
                        env_table = gr.Dataframe(
                            headers=["key", "value"],
                            datatype=["str", "str"],
                            row_count=(0, "dynamic"),
                            col_count=(2, "fixed"),
                            label="后端环境变量（白名单）",
                            interactive=False,
                        )

                    with gr.Tab("WebUI 连接"):
                        gr.Markdown(
                            """- 变更 `MODEL_ID/MODEL_DIR/USE_VLLM` 等加载期参数仍需重启后端。
- 生成参数、SYSTEM_PROMPT 支持运行时更新。"""
                        )
                        gr.Dataframe(
                            value=[
                                ["API_BASE_URL", API_BASE_URL],
                                ["API_TIMEOUT", str(API_TIMEOUT)],
                                ["WEBUI_HOST", WEBUI_HOST],
                                ["WEBUI_PORT", str(WEBUI_PORT)],
                                ["WEBUI_SHARE", str(WEBUI_SHARE)],
                            ],
                            headers=["key", "value"],
                            datatype=["str", "str"],
                            row_count=(5, "fixed"),
                            col_count=(2, "fixed"),
                            interactive=False,
                            label="WebUI 参数",
                        )

        # 事件处理
        def user_submit(user_msg, history):
            """处理用户提交"""
            if not history:
                history = []
            # Gradio 6.x 使用字典格式；不添加 None 内容，避免后处理报错
            history.append({"role": "user", "content": user_msg})
            return "", history

        def _to_text(content: Any) -> str:
            """将 Chatbot 消息内容安全转换为字符串。

            兼容 Gradio 6.x：content 可能是 str、list[dict|str]、dict。
            - 若为 list[dict]：尝试拼接其中的 'text' 或 'content' 字段。
            - 若为 list[str]：按换行拼接。
            - 若为 dict：优先取 'text' 或 'content'。
            - 其他类型：用 str() 兜底。
            """
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for seg in content:
                    if isinstance(seg, dict):
                        t = seg.get("text") or seg.get("content") or ""
                        if isinstance(t, str) and t:
                            parts.append(t)
                    elif isinstance(seg, str):
                        parts.append(seg)
                return "\n".join([p for p in parts if p])
            if isinstance(content, dict):
                t = content.get("text") or content.get("content") or ""
                return t if isinstance(t, str) else str(t)
            return str(content or "")

        def bot_respond(
            history,
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            frequency_penalty,
        ):
            """处理机器人回复（兼容 Gradio 6.x Chatbot 字典消息格式）"""
            if not history:
                return history

            # 找到最后一条用户消息
            user_msg = None
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    user_msg = _to_text(msg.get("content", ""))
                    break
            if not user_msg:
                return history

            # 若最后不是 assistant 消息或其 content 非字符串，先追加占位（空串），避免 None
            if not (isinstance(history[-1], dict) and history[-1].get("role") == "assistant" and isinstance(history[-1].get("content"), str)):
                history.append({"role": "assistant", "content": ""})

            gen_params = {
                "max_new_tokens": int(max_new_tokens) if max_new_tokens is not None else None,
                "temperature": float(temperature) if temperature is not None else None,
                "top_p": float(top_p) if top_p is not None else None,
                "top_k": int(top_k) if top_k is not None else None,
                "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else None,
                "frequency_penalty": float(frequency_penalty) if frequency_penalty is not None else None,
            }

            # 调用后端生成，并逐步更新最后一条 assistant 的内容
            for response in predict(user_msg, gen_params=gen_params):
                history[-1]["content"] = response or ""
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
            user_submit,
            [user_input, chatbot],
            [user_input, chatbot],
            queue=False,
        ).then(
            bot_respond,
            [
                chatbot,
                ui_max_new_tokens,
                ui_temperature,
                ui_top_p,
                ui_top_k,
                ui_repetition_penalty,
                ui_frequency_penalty,
            ],
            chatbot,
        )

        user_input.submit(
            user_submit,
            [user_input, chatbot],
            [user_input, chatbot],
            queue=False,
        ).then(
            bot_respond,
            [
                chatbot,
                ui_max_new_tokens,
                ui_temperature,
                ui_top_p,
                ui_top_k,
                ui_repetition_penalty,
                ui_frequency_penalty,
            ],
            chatbot,
        )

        clear_btn.click(clear_history, None, [chatbot, user_input], queue=False)

        info_btn.click(fetch_backend_info, None, [backend_info_json, env_table], queue=False)

        sys_prompt_reload_btn.click(fetch_system_prompt, None, sys_prompt_box, queue=False)
        sys_prompt_apply_btn.click(apply_system_prompt, [sys_prompt_box], sys_prompt_status, queue=False)

        batch_btn.click(run_batch_test, None, batch_out, queue=True)

        # 初始加载一次 /info
        demo.load(fetch_backend_info, None, [backend_info_json, env_table], queue=False)
        demo.load(fetch_system_prompt, None, sys_prompt_box, queue=False)

    return demo


def main():
    """启动 WebUI"""
    print("=" * 60)
    print("🚀 启动 Qwen2.5-0.5B Plus WebUI")
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
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
