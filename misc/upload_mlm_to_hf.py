"""
Upload the fine-tuned XLM-RoBERTa MLM model to HuggingFace Hub.
Run from: C:\Y5_Docs\FYP\SinCode\SinCode_v3
Usage: python misc/upload_mlm_to_hf.py --token YOUR_HF_WRITE_TOKEN
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi

MODEL_LOCAL_PATH = Path(
    r"C:\Y5_Docs\FYP\SinCode\SinCode_v2-20260315T161648Z-1-001"
    r"\SinCode_v2\SinCode\SinCode\xlm-roberta-sinhala-v5-strict-full\final"
)
REPO_ID = "Kalana001/xlm-roberta-base-finetuned-sinhala"

FILES_TO_UPLOAD = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write-access token")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    print(f"Uploading to: {REPO_ID}")
    for filename in FILES_TO_UPLOAD:
        local_file = MODEL_LOCAL_PATH / filename
        if not local_file.exists():
            print(f"  SKIP (not found): {filename}")
            continue
        size_mb = round(local_file.stat().st_size / 1024 / 1024, 1)
        print(f"  Uploading {filename} ({size_mb} MB)...")
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="model",
        )
        print(f"  Done: {filename}")

    print(f"\nAll files uploaded to https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    main()
