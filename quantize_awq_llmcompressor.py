#!/usr/bin/env python3
"""\
一次性脚本：对 Qwen2.5-0.5B-Plus-LLM 做 AutoAWQ 4bit 量化并导出。

你只需要改路径（或传参），直接运行导出一版 AWQ；其它参数全部内置，
以“尽量保准确率”为优先。

备注：脚本里的“max_seq_len 越大越慢、越吃显存”说的是【量化校准阶段】。
你这里不在乎耗时没问题，但如果显存不够，仍可能 OOM。

依赖：建议在单独环境安装 requirements-quantize-awq.txt 后运行。
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any, List


# =====================
# 一次性配置区（只为 0.5B）
# =====================
DEFAULT_MODEL_DIR = "model/YukinoStuki/Qwen2.5-0.5B-Plus-LLM"
# 输出目录/仓库名对齐：Qwen2.5-0.5B-Plus-AWQ
DEFAULT_OUTPUT_DIR = "model/YukinoStuki/Qwen2.5-0.5B-Plus-AWQ"

# 校准集（建议：用与你线上评测相近分布的数据；这里默认用仓库已有的 calib_512.jsonl）
DEFAULT_CALIB_JSONL = "calib_8192.jsonl"

# 量化参数（偏保守以保准确率）
AWQ_W_BIT = 4
# 兼容性优先：不少 vLLM/平台插件对 AWQ group size=128 支持更稳。
AWQ_Q_GROUP_SIZE = 128
AWQ_ZERO_POINT = True
AWQ_BACKEND = "GEMM"  # AutoAWQ quant backend

# 校准规模（时间无所谓时可适当加大；显存不足就降 max_seq_len 或 num_calib）
AWQ_NUM_CALIB = 8192
AWQ_MAX_SEQ_LEN = 2048

# 一般不量化 lm_head（更稳）
AWQ_MODULES_TO_NOT_CONVERT = ["lm_head"]

# 关键：校准 prompt 分布要贴近线上推理（system+user+assistant 的 chat prompt）
AWQ_CALIB_APPLY_CHAT_TEMPLATE = True
AWQ_CALIB_SYSTEM_PROMPT = (
    "你是评测答题模型。目标：ROUGE-L高分且尽量少输出token。\n"
    "只输出答案正文，切中要点，列出个别关键术语或结论，不要任何“思考过程/推理/分析/步骤/解释/客套”，不要出现“思考完成”等字样。\n\n"
    "写法要求：\n"
    "1) 尽量复用教材/标准表述，少改写，保持常见措辞与词序。\n"
    "2) 用3-6个短句/短语覆盖关键点（定义/参数/公式/步骤/关键术语），不要长段落。\n"
    "3) 不举例、不扩展背景、不重复。\n"
    "4) 若题目要求代码：只输出最短可用的核心代码/伪代码骨架，不加Markdown围栏，不解释。"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("AWQ_MODEL_DIR") or DEFAULT_MODEL_DIR,
        help="本地 HF 模型目录（含 config.json / tokenizer 等）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("AWQ_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR,
        help="输出目录（标准 HF 目录，用于 upload_model.py 覆盖上传）",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="是否启用 trust_remote_code（Qwen 系列通常需要）",
    )
    parser.add_argument(
        "--no_trust_remote_code",
        action="store_false",
        dest="trust_remote_code",
        help="关闭 trust_remote_code",
    )
    return parser.parse_args()


def _iter_calib_texts(calib_jsonl: Path, limit: int) -> List[str]:
    import json

    # 为了更贴近真实分布，避免“只取文件开头”的偏差：做一次均匀随机抽样。
    # reservoir sampling：单次扫描、O(limit) 内存，适合大文件。
    limit = int(limit)
    if limit <= 0:
        return []

    rng = random.Random(42)
    texts: List[str] = []
    seen = 0

    with calib_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            t = obj.get("text")
            if not (isinstance(t, str) and t.strip()):
                continue

            seen += 1
            if len(texts) < limit:
                texts.append(t)
                continue

            # 以 limit/seen 的概率替换已有样本
            j = rng.randrange(seen)
            if j < limit:
                texts[j] = t

    return texts


def _format_calib_as_chat(tokenizer: Any, user_text: str, system_prompt: str) -> str:
    """把校准样本文本包装成 chat prompt。

    优先使用 tokenizer.apply_chat_template（最兼容）；失败则回退到 Qwen 常见 im_start 格式。
    """

    u = (user_text or "").strip()
    if not u:
        return ""

    sys_p = (system_prompt or "").strip()
    if tokenizer is None:
        # 无 tokenizer 时只能用纯文本拼接。
        if sys_p:
            return sys_p + "\n" + u
        return u

    messages: list[dict[str, str]] = []
    if sys_p:
        messages.append({"role": "system", "content": sys_p})
    messages.append({"role": "user", "content": u})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Qwen2/2.5/3 系列常见格式：<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
        if sys_p:
            return (
                "<|im_start|>system\n"
                + sys_p
                + "<|im_end|>\n"
                + "<|im_start|>user\n"
                + u
                + "<|im_end|>\n"
                + "<|im_start|>assistant\n"
            )
        return (
            "<|im_start|>user\n" + u + "<|im_end|>\n" + "<|im_start|>assistant\n"
        )


def main() -> None:
    args = _parse_args()

    model_dir = Path(args.model_dir).expanduser()
    calib_jsonl = Path(os.environ.get("AWQ_CALIB_JSONL") or DEFAULT_CALIB_JSONL).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not model_dir.is_dir():
        raise SystemExit(f"model_dir 不存在或不是目录：{model_dir}")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"model_dir 缺少 config.json：{model_dir}")
    if not calib_jsonl.is_file():
        raise SystemExit(
            f"calib_jsonl 不存在：{calib_jsonl}\n"
            "请先生成：N=8192 MAX_LEN=2048 OUT_JSONL=calib_8192.jsonl OUT_TXT=calib_8192.txt python3 sample_calib_from_data.py"
        )

    # 延迟 import：用户只看 --help 时不强制安装重依赖。
    # 避免 transformers 因 torchvision/torch 版本不匹配导致导入崩溃（量化不需要 vision）。
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    import torch
    from transformers import AutoTokenizer

    try:
        from awq import AutoAWQForCausalLM  # type: ignore
    except Exception as e:
        raise SystemExit(
            "缺少 AutoAWQ 依赖：请先安装 requirements-quantize-awq.txt 后再运行。\n"
            f"ImportError: {e}"
        )

    print("[AutoAWQ] model_dir =", str(model_dir))
    print("[AutoAWQ] output_dir =", str(output_dir))
    print("[AutoAWQ] calib_jsonl =", str(calib_jsonl))
    print("[AutoAWQ] torch.cuda.is_available() =", torch.cuda.is_available())

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    # 读取校准文本（AutoAWQ 支持传 list[str]）
    calib_texts = _iter_calib_texts(calib_jsonl, limit=int(AWQ_NUM_CALIB))
    if not calib_texts:
        raise SystemExit("校准集为空或解析失败：请检查 calib_jsonl 格式是否为每行 {\"text\": \"...\"}")

    if AWQ_CALIB_APPLY_CHAT_TEMPLATE:
        sys_p = str(AWQ_CALIB_SYSTEM_PROMPT or "").strip()
        formatted: List[str] = []
        for t in calib_texts:
            ft = _format_calib_as_chat(tokenizer, t, sys_p)
            if ft:
                formatted.append(ft)
        if formatted:
            calib_texts = formatted
        print(
            "[AutoAWQ] calib_apply_chat_template=1",
            f"system_prompt_len={len(sys_p)}",
            f"num_calib={len(calib_texts)}",
        )
    else:
        print("[AutoAWQ] calib_apply_chat_template=0 (using raw texts)")

    quant_config: dict[str, Any] = {
        "zero_point": bool(AWQ_ZERO_POINT),
        "q_group_size": int(AWQ_Q_GROUP_SIZE),
        "w_bit": int(AWQ_W_BIT),
        "version": str(AWQ_BACKEND),
    }
    mods = list(AWQ_MODULES_TO_NOT_CONVERT)
    if mods:
        print("[AutoAWQ] modules_to_not_convert =", mods)
    print("[AutoAWQ] quant_config =", quant_config)
    print(
        f"[AutoAWQ] Quantizing with num_calib={len(calib_texts)}, max_seq_len={AWQ_MAX_SEQ_LEN}"
    )

    model = AutoAWQForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=args.trust_remote_code,
        safetensors=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    # AutoAWQ 典型 API：model.quantize(tokenizer, quant_config, calib_data, max_calib_seq_len)
    # 不同版本的 AutoAWQ API 可能略有差异：这里做一次兼容调用。
    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,
            max_calib_seq_len=int(AWQ_MAX_SEQ_LEN),
            modules_to_not_convert=mods if mods else None,
        )
    except TypeError:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,
            max_calib_seq_len=int(AWQ_MAX_SEQ_LEN),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir), safetensors=True)
    tokenizer.save_pretrained(str(output_dir))

    # 显式写出 quant_config.json：部分加载器/插件依赖该文件；缺失时可能出现输出异常。
    try:
        import json

        with (output_dir / "quant_config.json").open("w", encoding="utf-8") as f:
            f.write(json.dumps(quant_config, ensure_ascii=False, indent=2) + "\n")
        if mods:
            with (output_dir / "modules_to_not_convert.txt").open("w", encoding="utf-8") as f:
                for mname in mods:
                    f.write(str(mname) + "\n")
    except Exception as e:
        print("[AutoAWQ] Warning: failed to write quant_config.json:", e)

    print("[AutoAWQ] Saved quantized model to:", str(output_dir))
    print("[AutoAWQ] Done.")


if __name__ == "__main__":
    main()
