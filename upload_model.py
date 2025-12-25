import argparse
import os
import sys
from typing import Tuple
import inspect

from modelscope.hub.api import HubApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a merged model directory to ModelScope via SDK."
    )
    parser.add_argument(
        "--model_dir",
        default=os.environ.get("MODEL_DIR", os.path.join(os.getcwd(), "merged")),
        help="Path to merged model directory. Env: MODEL_DIR",
    )
    parser.add_argument(
        "--model_id",
        default=os.environ.get("MODEL_ID", ""),
        help="Target ModelScope model ID, e.g. org/name. Env: MODEL_ID (required)",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("MODEL_REVISION", "v2"),
        help="Target revision/tag to push. Env: MODEL_REVISION",
    )
    return parser.parse_args()


def summarize_dir(model_dir: str) -> Tuple[int, int]:
    total_files = 0
    total_bytes = 0
    try:
        from tqdm import tqdm  # optional for nicer progress

        for root, _dirs, files in tqdm(os.walk(model_dir), desc="[upload_model] Scanning", unit="dir"):
            for f in files:
                total_files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except Exception:
        # Fallback: plain walk without progress bar if tqdm unavailable
        for root, _dirs, files in os.walk(model_dir):
            for f in files:
                total_files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    return total_files, total_bytes


def main() -> int:
    args = parse_args()
    token = os.environ.get("MODELSCOPE_API_TOKEN", "")

    if not token:
        print("[upload_model] ERROR: MODELSCOPE_API_TOKEN is required.")
        return 1
    if not args.model_id:
        print("[upload_model] ERROR: MODEL_ID is required (org/name).")
        return 1
    if not os.path.isdir(args.model_dir):
        print(f"[upload_model] ERROR: model_dir not found: {args.model_dir}")
        return 1

    total_files, total_bytes = summarize_dir(args.model_dir)
    print(f"[upload_model] Preparing upload: {total_files} files, {total_bytes/1024/1024:.2f} MB")

    api = HubApi()
    api.login(token)

    print(f"[upload_model] Uploading {args.model_dir} -> {args.model_id} (revision={args.revision}) via upload_folder...")
    try:
        sig = inspect.signature(api.upload_folder)
        param_names = set(sig.parameters.keys())

        # Prefer modern parameter names (repo_id/folder_path). Fall back if SDK differs.
        candidate_kwargs = {
            "repo_id": args.model_id,
            "model_id": args.model_id,
            "folder_path": args.model_dir,
            "model_dir": args.model_dir,
            "revision": args.revision,
            "token": token,
        }

        call_kwargs = {k: v for k, v in candidate_kwargs.items() if k in param_names}

        # Ensure we pass a valid repo identifier field.
        if "repo_id" not in call_kwargs and "model_id" not in call_kwargs:
            raise RuntimeError(f"HubApi.upload_folder signature not recognized: {sig}")

        api.upload_folder(**call_kwargs)
    except Exception as e:
        print("[upload_model] FAILED:", repr(e))
        return 1

    print("[upload_model] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
