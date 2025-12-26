
import argparse
import os
import sys
from modelscope import snapshot_download
from modelscope.hub.api import HubApi

def parse_args():
    parser = argparse.ArgumentParser(description='Download model parameters settings')
    parser.add_argument('--model_name', type=str, default=os.environ.get('MODEL_ID', 'YukinoStuki/Qwen3-4B-Plus-Merged'),
                        help='Model name in format: organization/model_name')
    parser.add_argument('--cache_dir', type=str, default='./',
                        help='Directory to save the model')
    parser.add_argument('--revision', type=str, default='master',
                        help='Model revision/version')
    return parser.parse_args()

if __name__ == "__main__":
    token = os.environ.get("MODELSCOPE_API_TOKEN", "")
    if token:
        api = HubApi()
        api.login(token)
        print("Using MODELSCOPE_API_TOKEN from environment for authenticated download.")
    sh_args = parse_args()
    try:
        model_dir = snapshot_download(
            sh_args.model_name,
            cache_dir=sh_args.cache_dir,
            revision=sh_args.revision
        )
        print("Model download successful!")
        print(f"Resolved model_dir: {model_dir}")
    except Exception as e:
        print("Model download FAILED:", repr(e))
        sys.exit(1)