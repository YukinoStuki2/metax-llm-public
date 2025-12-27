import argparse
import os
import sys
from pathlib import Path

from modelscope.hub.api import HubApi


def _env_token() -> str:
    return (
        os.environ.get("MODELSCOPE_API_TOKEN")
        or os.environ.get("MODELSCOPE_TOKEN")
        or ""
    ).strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="上传本地模型目录到 ModelScope（支持用参数覆盖 repo_id/model_dir）。"
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("REPO_ID")
        or "YukinoStuki/Qwen3-4B-Plus-LLM-AWQ",
        help="ModelScope repo id（也可用环境变量 REPO_ID 覆盖）",
    )
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MODEL_DIR")
        or "model/YukinoStuki/Qwen3-4B-Plus-LLM-AWQ",
        help="本地模型目录（也可用环境变量 MODEL_DIR 覆盖）",
    )
    parser.add_argument(
        "--token",
        default="",
        help=(
            "ModelScope API Token。默认不内置任何值；"
            "若不传则从环境变量 MODELSCOPE_API_TOKEN/MODELSCOPE_TOKEN 读取。"
        ),
    )
    parser.add_argument(
        "--commit-message",
        default="upload model folder",
        help="上传时的 commit message",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    repo_id = (args.repo_id or "").strip()
    folder = Path(args.model_dir).expanduser()
    token = (args.token or _env_token()).strip()

    if not repo_id:
        print("ERROR: repo_id 为空；请用 --repo-id 或环境变量 REPO_ID 指定")
        sys.exit(1)

    if not folder.exists():
        print(f"ERROR: MODEL_DIR 不存在：{folder}")
        sys.exit(1)

    # 轻量校验：避免把 LoRA 目录/空目录传上去
    must_any = ["config.json", "tokenizer_config.json"]
    missing = [f for f in must_any if not (folder / f).exists()]
    if missing:
        print("WARNING: 目录缺少关键文件：", missing)
        print(
            "如果你上传的是“完整模型”，请确认 Export 输出目录是否正确；"
            "如果你只上传 LoRA adapter 可忽略。"
        )

    api = HubApi()
    if token:
        api.login(token)
    else:
        # 有些环境可能已通过 modelscope 的本地配置缓存登录。
        # 这里不强制要求 token（按用户要求：不在脚本内写默认 token），但若上传失败会提示。
        print(
            "WARNING: 未提供 token（--token 或 MODELSCOPE_API_TOKEN/MODELSCOPE_TOKEN）。"
            "若上传失败，请补充 token。"
        )

    print(f"Uploading folder: {folder}  -->  {repo_id}")
    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(folder),
            commit_message=args.commit_message,
        )
    except Exception as e:
        print("ERROR: 上传失败：", repr(e))
        if not token:
            print("提示：请设置 MODELSCOPE_API_TOKEN（或使用 --token）后重试")
        sys.exit(1)

    print("Upload finished.")


if __name__ == "__main__":
    main()