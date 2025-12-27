import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Optional


def run(cmd: list[str], cwd: Optional[str] = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_clean_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def resolve_adapter_config(adapter_dir: str) -> None:
    """确保 PEFT 所需的 adapter_config.json 存在。

    如果适配器仓库只提供 adapter_model.safetensors，则需要用户通过环境变量提供配置：
      - ADAPTER_CONFIG_JSON：原始 JSON 字符串
      - ADAPTER_CONFIG_PATH：json 文件路径
    """

    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.isfile(cfg_path):
        return

    cfg_json = os.environ.get("ADAPTER_CONFIG_JSON", "").strip()
    cfg_from_path = os.environ.get("ADAPTER_CONFIG_PATH", "").strip()

    if cfg_json:
        try:
            parsed = json.loads(cfg_json)
        except Exception as e:
            raise RuntimeError(f"ADAPTER_CONFIG_JSON is not valid JSON: {e}")
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, ensure_ascii=False, indent=2)
        return

    if cfg_from_path:
        if not os.path.isfile(cfg_from_path):
            raise RuntimeError(f"ADAPTER_CONFIG_PATH not found: {cfg_from_path}")
        shutil.copyfile(cfg_from_path, cfg_path)
        return

    raise RuntimeError(
        "adapter_config.json not found in adapter repo. "
        "Provide ADAPTER_CONFIG_JSON or ADAPTER_CONFIG_PATH so we can merge adapter_model.safetensors."
    )


def clone_adapter_repo(repo_url: str, dest_dir: str, ref: Optional[str]) -> None:
    ensure_clean_dir(dest_dir)
    # Docker build 场景优先浅克隆；ref 可以是 branch/tag/commit。
    run(["git", "clone", "--depth", "1", repo_url, dest_dir])
    if ref:
        run(["git", "fetch", "--depth", "1", "origin", ref], cwd=dest_dir)
        run(["git", "checkout", "FETCH_HEAD"], cwd=dest_dir)


def find_adapter_weights(adapter_dir: str) -> str:
    # 常见 PEFT 权重文件名
    candidates = [
        os.path.join(adapter_dir, "adapter_model.safetensors"),
        os.path.join(adapter_dir, "adapter_model.bin"),
        os.path.join(adapter_dir, "pytorch_model.bin"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError(
        "Adapter weights not found. Expected one of: adapter_model.safetensors / adapter_model.bin / pytorch_model.bin"
    )


def is_git_lfs_pointer(file_path: str) -> bool:
    try:
        with open(file_path, "rb") as f:
            head = f.read(256)
    except Exception:
        return False

    marker = b"git-lfs"
    return head.startswith(b"version https://git-lfs.github.com/spec/v1") or marker in head


def materialize_lfs_file(file_path: str, repo_dir: str) -> None:
    if not is_git_lfs_pointer(file_path):
        return

    print("[merge_adapter] Detected Git LFS pointer; trying git lfs pull to fetch real weights...")
    try:
        run(["git", "lfs", "install", "--local"], cwd=repo_dir)
        run(["git", "lfs", "pull"], cwd=repo_dir)
    except Exception as e:
        raise RuntimeError(
            "Adapter weight appears to be a Git LFS pointer; git-lfs is missing or pull failed. "
            "Install git-lfs and ensure LFS objects can be downloaded."
        ) from e

    if is_git_lfs_pointer(file_path):
        raise RuntimeError(
            "Git LFS pull completed but the adapter weight file is still a pointer. "
            "Please verify LFS availability and that the large files are accessible."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download base model from ModelScope, clone adapter repo, and merge adapter into a full merged model directory."
    )

    # 默认使用当前工作目录，便于在本地/WSL 运行时不依赖 /app 写权限。
    cwd = os.getcwd()
    default_cache_dir = os.path.join(cwd, "model")
    default_work_dir = os.path.join(cwd, "merge_work")
    default_output_dir = os.path.join(cwd, "merged")

    p.add_argument(
        "--base_model",
        default=os.environ.get("BASE_MODEL", "Qwen/Qwen3-4B"),
        help="ModelScope base model id. Env: BASE_MODEL. Default: Qwen/Qwen3-4B",
    )
    p.add_argument(
        "--base_revision",
        default=os.environ.get("BASE_REVISION", "master"),
        help="ModelScope base model revision. Env: BASE_REVISION",
    )
    p.add_argument(
        "--cache_dir",
        default=os.environ.get("MODEL_CACHE_DIR", default_cache_dir),
        help="Where to store downloaded models. Env: MODEL_CACHE_DIR",
    )
    p.add_argument(
        "--adapter_repo_url",
        default=os.environ.get("ADAPTER_REPO_URL", "git@gitee.com:yukinostuki/qwen3-4b-plus.git"),
        help="Adapter repo URL to clone. Env: ADAPTER_REPO_URL",
    )
    p.add_argument(
        "--adapter_ref",
        default=os.environ.get("ADAPTER_REPO_REF", ""),
        help="Optional git ref (branch/tag/commit). Env: ADAPTER_REPO_REF",
    )
    p.add_argument(
        "--work_dir",
        default=os.environ.get("MERGE_WORK_DIR", default_work_dir),
        help="Working directory for cloning adapter. Env: MERGE_WORK_DIR",
    )
    p.add_argument(
        "--output_dir",
        default=os.environ.get("MERGED_MODEL_DIR", default_output_dir),
        help="Output merged model directory. Env: MERGED_MODEL_DIR",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 延迟导入：让缺依赖时报错信息更干净。
    from modelscope import snapshot_download

    print("[merge_adapter] Downloading base model from ModelScope...")
    base_dir = snapshot_download(args.base_model, cache_dir=args.cache_dir, revision=args.base_revision)
    print("[merge_adapter] Base model resolved to:", base_dir)

    adapter_dir = os.path.join(args.work_dir, "adapter_repo")
    os.makedirs(args.work_dir, exist_ok=True)

    print("[merge_adapter] Cloning adapter repo...")
    clone_adapter_repo(args.adapter_repo_url, adapter_dir, args.adapter_ref or None)
    adapter_weights = find_adapter_weights(adapter_dir)
    print("[merge_adapter] Adapter weights:", adapter_weights)

    # 若适配器权重由 Git LFS 跟踪，则确保已拉取到真实大文件。
    materialize_lfs_file(adapter_weights, adapter_dir)

    print("[merge_adapter] Resolving adapter config...")
    resolve_adapter_config(adapter_dir)

    print("[merge_adapter] Loading base model + adapter and merging...")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True, use_fast=False)

    # 优先 fp16 降内存；若 CPU 上 fp16 加载失败则回退到 fp32。
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    except Exception as e:
        print("[merge_adapter] fp16 load failed, fallback to fp32:", repr(e))
        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    print("[merge_adapter] Saving merged model to:", args.output_dir)
    ensure_clean_dir(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    print("[merge_adapter] Done.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("[merge_adapter] FAILED:", repr(e))
        sys.exit(1)
