#!/usr/bin/env python3
"""\
用 llm-compressor 对本地 HuggingFace 目录模型做 AWQ，并导出 vLLM 可直接加载的 compressed-tensors 格式目录。

设计目标：
- 不触碰本仓库的 serving 依赖（requirements.txt 不改）；建议在单独环境里运行本脚本。
- 使用本仓库已生成的校准集（calib_512.jsonl，每行 {"text": "..."}）。

典型用法：
  python quantize_awq_llmcompressor.py \
    --model_dir merged \
    --calib_jsonl calib_512.jsonl \
    --output_dir merged-awq-w4a16-asym \
    --num_calib 256 \
    --max_seq_len 1024

提示：
- 首次建议 --num_calib 128/256、--max_seq_len 512/1024 先跑通，避免 OOM。
- 成功后目录里应包含 safetensors + 压缩/量化相关 config，供 vLLM 加载。
"""

from __future__ import annotations

import argparse
import os

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default="merged",
        help="本地 HF 模型目录（含 config.json / tokenizer 等）",
    )
    parser.add_argument(
        "--calib_jsonl",
        type=str,
        default="calib_512.jsonl",
        help='校准集 JSONL（每行 {"text": "..."}）',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="merged-awq-w4a16-asym",
        help="输出目录（将写入 compressed-tensors 格式）",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=256,
        help="使用多少条校准样本（<= 校准集行数）",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="校准时最大序列长度（越大越慢、越吃显存）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="shuffle 随机种子",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="加载模型 dtype",
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


def _resolve_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {dtype}")


def main() -> None:
    args = _parse_args()

    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"model_dir 不存在或不是目录：{args.model_dir}")
    if not os.path.isfile(args.calib_jsonl):
        raise SystemExit(f"calib_jsonl 不存在：{args.calib_jsonl}")

    # 延迟 import，避免用户仅查看 --help 时也强制依赖安装。
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    torch.manual_seed(args.seed)

    model_dtype = _resolve_dtype(args.dtype)

    print(f"[AWQ] Loading model from: {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
        device_map=None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    if torch.cuda.is_available():
        model = model.to("cuda")

    print(f"[AWQ] Loading calibration data: {args.calib_jsonl}")
    ds = load_dataset("json", data_files=args.calib_jsonl, split="train")
    ds = ds.shuffle(seed=args.seed)

    # 让 oneshot 拿到模型实际输入：input_ids / attention_mask。
    def tokenize(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=args.max_seq_len,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # 默认 recipe：W4A16_ASYM（4-bit weight + 16-bit activation，非对称）
    # ignore lm_head，避免输出层被量化导致损失增大。
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16_ASYM",
            targets=["Linear"],
        )
    ]

    os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"[AWQ] Running oneshot: num_calib={args.num_calib}, max_seq_len={args.max_seq_len}"
    )
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_calib,
        output_dir=args.output_dir,
        log_dir=None,
    )

    # oneshot(output_dir=...) 会负责保存压缩后的模型；这里补齐 tokenizer。
    tokenizer.save_pretrained(args.output_dir)

    print(f"[AWQ] Saved compressed model to: {args.output_dir}")

    print("[AWQ] Done.")


if __name__ == "__main__":
    main()
