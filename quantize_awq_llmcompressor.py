#!/usr/bin/env python3
"""\
使用 AutoAWQ 对本地 HuggingFace 目录模型做 AWQ（非 marlin 依赖），并导出为标准 HF 目录。

背景：
- 沐曦/MetaX 的部分环境/架构不支持 Marlin kernel。
- 之前 llm-compressor 导出的 compressed-tensors 路径在 vLLM 上会走 marlin 分支，导致无法运行。
- 这里改用 AutoAWQ 产物（HF 目录 + quant config），用于覆盖同名仓库的旧产物。

重要约束（按你的要求固定）：
- 输出目录默认固定为：model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ
- 上传脚本 upload_model.py 默认也指向同一路径，可直接覆盖之前那份。

典型用法：
  python quantize_awq_llmcompressor.py \
    --model_dir model/YukinoStuki/Qwen3-4B-Plus-LLM \
    --calib_jsonl calib_512.jsonl \
    --output_dir model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ

说明：
- 本脚本不改 requirements.txt（线上 serving 不依赖 AutoAWQ）。
- 建议在单独环境安装依赖后运行（见 requirements-quantize-awq.txt）。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, List


DEFAULT_MODEL_DIR = "model/YukinoStuki/Qwen3-4B-Plus-LLM"
DEFAULT_OUTPUT_DIR = "model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("AWQ_MODEL_DIR") or DEFAULT_MODEL_DIR,
        help="本地 HF 模型目录（含 config.json / tokenizer 等）",
    )
    parser.add_argument(
        "--calib_jsonl",
        type=str,
        default=os.environ.get("AWQ_CALIB_JSONL") or "calib_512.jsonl",
        help='校准集 JSONL（每行 {"text": "..."}）',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("AWQ_OUTPUT_DIR") or DEFAULT_OUTPUT_DIR,
        help="输出目录（标准 HF 目录，用于 upload_model.py 覆盖上传）",
    )
    parser.add_argument(
        "--num_calib",
        type=int,
        default=int(os.environ.get("AWQ_NUM_CALIB", "256")),
        help="使用多少条校准样本（<= 校准集行数）",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=int(os.environ.get("AWQ_MAX_SEQ_LEN", "1024")),
        help="校准时最大序列长度（越大越慢、越吃显存）",
    )
    parser.add_argument(
        "--w_bit",
        type=int,
        default=int(os.environ.get("AWQ_W_BIT", "4")),
        help="权重量化 bit 数（常用 4）",
    )
    parser.add_argument(
        "--q_group_size",
        type=int,
        default=int(os.environ.get("AWQ_Q_GROUP_SIZE", "128")),
        help="group size（常用 128）",
    )
    parser.add_argument(
        "--zero_point",
        type=int,
        default=int(os.environ.get("AWQ_ZERO_POINT", "1")),
        help="是否使用 zero point（1/0）",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=os.environ.get("AWQ_BACKEND") or "GEMM",
        choices=["GEMM", "GEMV"],
        help="AutoAWQ quant backend（通常 GEMM 更通用）",
    )
    parser.add_argument(
        "--modules_to_not_convert",
        type=str,
        default=os.environ.get("AWQ_MODULES_TO_NOT_CONVERT") or "lm_head",
        help=(
            "不量化的模块名（逗号分隔）。默认 lm_head，与之前脚本 ignore lm_head 对齐。"
        ),
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

    texts: List[str] = []
    with calib_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if len(texts) >= limit:
                break
            s = (line or "").strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                t = obj.get("text")
                if isinstance(t, str) and t.strip():
                    texts.append(t)
    return texts


def main() -> None:
    args = _parse_args()

    model_dir = Path(args.model_dir).expanduser()
    calib_jsonl = Path(args.calib_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not model_dir.is_dir():
        raise SystemExit(f"model_dir 不存在或不是目录：{model_dir}")
    if not (model_dir / "config.json").exists():
        raise SystemExit(f"model_dir 缺少 config.json：{model_dir}")
    if not calib_jsonl.is_file():
        raise SystemExit(f"calib_jsonl 不存在：{calib_jsonl}")

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
    calib_texts = _iter_calib_texts(calib_jsonl, limit=int(args.num_calib))
    if not calib_texts:
        raise SystemExit("校准集为空或解析失败：请检查 calib_jsonl 格式是否为每行 {\"text\": \"...\"}")

    quant_config: dict[str, Any] = {
        "zero_point": bool(int(args.zero_point)),
        "q_group_size": int(args.q_group_size),
        "w_bit": int(args.w_bit),
        "version": str(args.backend),
    }
    mods = [m.strip() for m in str(args.modules_to_not_convert).split(",") if m.strip()]
    if mods:
        print("[AutoAWQ] modules_to_not_convert =", mods)
    print("[AutoAWQ] quant_config =", quant_config)
    print(
        f"[AutoAWQ] Quantizing with num_calib={len(calib_texts)}, max_seq_len={args.max_seq_len}"
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
            max_calib_seq_len=int(args.max_seq_len),
            modules_to_not_convert=mods if mods else None,
        )
    except TypeError:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data=calib_texts,
            max_calib_seq_len=int(args.max_seq_len),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir), safetensors=True)
    tokenizer.save_pretrained(str(output_dir))

    print("[AutoAWQ] Saved quantized model to:", str(output_dir))
    print("[AutoAWQ] Done.")


if __name__ == "__main__":
    main()
