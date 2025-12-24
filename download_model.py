
import argparse
import sys
from modelscope import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description='Download model parameters settings')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-4B',
                        help='Model name in format: organization/model_name')
    parser.add_argument('--cache_dir', type=str, default='./',
                        help='Directory to save the model')
    parser.add_argument('--revision', type=str, default='master',
                        help='Model revision/version')
    return parser.parse_args()

if __name__ == "__main__":
    # api = HubApi()
    # api.login('YOUR_MODELSCOPE_ACCESS_TOKEN')
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
