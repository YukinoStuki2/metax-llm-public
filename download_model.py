
import argparse
import os
import sys
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

def parse_args():
    parser = argparse.ArgumentParser(description='Download model parameters settings')
    parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_ID', 'YukinoStuki/Qwen3-4B-Plus-Merged'),
                        help='Model name in format: organization/model_name')
    parser.add_argument(
        '--token',
        type=str,
        default=os.environ.get('MODELSCOPE_API_TOKEN', ''),
        help='(Optional) ModelScope API token. If empty, download anonymously.',
    )
    parser.add_argument(
        '--draft_model_name',
        type=str,
        default=os.environ.get('SPEC_DRAFT_MODEL_ID', ''),
        help='(Optional) Draft model for speculative decoding. Empty means skip.',
    )
    parser.add_argument('--cache_dir', type=str, default='./',
                        help='Directory to save the model')
    parser.add_argument('--revision', type=str, default='master',
                        help='Model revision/version')
    parser.add_argument(
        '--draft_revision',
        type=str,
        default=os.environ.get('SPEC_DRAFT_MODEL_REVISION', 'master'),
        help='(Optional) Draft model revision/version',
    )
    parser.add_argument(
        '--draft_optional',
        action='store_true',
        default=(os.environ.get('SPEC_DRAFT_OPTIONAL', '1') == '1'),
        help='If set, draft model download failure will not fail the build.',
    )
    return parser.parse_args()

if __name__ == "__main__":
    sh_args = parse_args()

    token = (sh_args.token or '').strip()
    if token:
        api = HubApi()
        api.login(token)
        print("Using ModelScope token for authenticated download.")
    try:
        model_dir = snapshot_download(
            sh_args.model_name,
            cache_dir=sh_args.cache_dir,
            revision=sh_args.revision
        )
        print("Model download successful!")
        print(f"Resolved model_dir: {model_dir}")

        draft_name = (sh_args.draft_model_name or '').strip()
        if draft_name:
            try:
                print(f"[spec] Downloading draft model: {draft_name} (revision={sh_args.draft_revision})")
                draft_dir = snapshot_download(
                    draft_name,
                    cache_dir=sh_args.cache_dir,
                    revision=sh_args.draft_revision,
                )
                print("[spec] Draft model download successful!")
                print(f"[spec] Resolved draft_model_dir: {draft_dir}")
            except Exception as e:
                if sh_args.draft_optional:
                    print("[spec] Draft model download FAILED (optional, continue):", repr(e))
                else:
                    print("[spec] Draft model download FAILED:", repr(e))
                    sys.exit(2)
    except Exception as e:
        print("Model download FAILED:", repr(e))
        sys.exit(1)