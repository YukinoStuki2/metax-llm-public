import argparse
import os
import shutil
import sys
import time
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
    parser.add_argument(
        "--force_git",
        action="store_true",
        help="Skip upload_folder and force git-based push (HubApi.push_model).",
    )
    return parser.parse_args()


def summarize_dir(model_dir: str) -> Tuple[int, int]:
    total_files = 0
    total_bytes = 0
    try:
        from tqdm import tqdm  # 可选：用于更友好的进度条

        for root, _dirs, files in tqdm(os.walk(model_dir), desc="[upload_model] Scanning", unit="dir"):
            for f in files:
                total_files += 1
                try:
                    total_bytes += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except Exception:
        # 兜底：若 tqdm 不可用，则不带进度条地遍历目录
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

    def _clean_modelscope_temp_dir(model_dir: str) -> None:
        # ModelScope 的 push_model 会在 model_dir 下创建临时目录 ._____temp。
        # 如果上一次上传中断/失败，残留的未跟踪文件会导致后续 checkout 分支报错。
        tmp_dir = os.path.join(model_dir, "._____temp")
        if not os.path.exists(tmp_dir):
            return

        def _on_rm_error(func, path, exc_info):
            # Windows 上常见：只读文件/权限导致删除失败。
            try:
                os.chmod(path, 0o700)
            except Exception:
                pass
            func(path)

        last_err: Exception | None = None
        for _ in range(3):
            try:
                if os.path.isdir(tmp_dir):
                    shutil.rmtree(tmp_dir, onerror=_on_rm_error)
                else:
                    os.remove(tmp_dir)
                if not os.path.exists(tmp_dir):
                    return
            except Exception as e:
                last_err = e
                time.sleep(0.2)

        # 仍然存在：不要静默继续，否则 git clone 会报“already exists and is not empty”。
        raise RuntimeError(
            f"Failed to remove temporary dir: {tmp_dir}. "
            "Please close any process using it (Explorer, AV, VSCode), then delete it manually and retry. "
            f"Last error: {last_err!r}"
        )

    def _upload_via_upload_folder() -> None:
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
        if "repo_id" not in call_kwargs and "model_id" not in call_kwargs:
            raise RuntimeError(f"HubApi.upload_folder signature not recognized: {sig}")

        api.upload_folder(**call_kwargs)

    def _upload_via_git_push() -> None:
        # push_model 走 git clone + push，会自动创建 revision 分支（若远端不存在）。
        # 同时对大权重文件启用 LFS，避免普通 git 提交失败。
        abs_model_dir = os.path.abspath(args.model_dir)
        _clean_modelscope_temp_dir(abs_model_dir)
        try:
            api.push_model(
                model_id=args.model_id,
                model_dir=abs_model_dir,
                revision=args.revision,
                lfs_suffix=["*.safetensors", "*.bin"],
            )
        except Exception as e:
            # 典型错误：残留 ._____temp 导致 checkout 分支时提示 untracked files would be overwritten。
            msg = str(e)
            if "untracked working tree files would be overwritten" in msg.lower():
                _clean_modelscope_temp_dir(abs_model_dir)
                api.push_model(
                    model_id=args.model_id,
                    model_dir=abs_model_dir,
                    revision=args.revision,
                    lfs_suffix=["*.safetensors", "*.bin"],
                )
                return
            raise

    if args.force_git:
        print(
            f"[upload_model] Uploading {args.model_dir} -> {args.model_id} (revision={args.revision}) via git push..."
        )
        try:
            _upload_via_git_push()
        except Exception as e:
            print("[upload_model] FAILED:", repr(e))
            return 1
    else:
        print(
            f"[upload_model] Uploading {args.model_dir} -> {args.model_id} (revision={args.revision}) via upload_folder..."
        )
        try:
            _upload_via_upload_folder()
        except Exception as e:
            msg = str(e)
            print("[upload_model] upload_folder failed:", repr(e))
            # ModelScope 偶发 500: get gitattributes failed / Failed to create commit...
            # 这种情况下回退到 git 推送路径通常更稳。
            should_fallback = any(
                key in msg.lower()
                for key in [
                    "gitattributes",
                    "failed to create commit",
                    "http 500",
                    "server error",
                ]
            )
            if not should_fallback:
                return 1
            print(
                f"[upload_model] Falling back to git push (HubApi.push_model) for revision={args.revision}..."
            )
            try:
                _upload_via_git_push()
            except Exception as e2:
                print("[upload_model] FAILED:", repr(e2))
                return 1

    print("[upload_model] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
