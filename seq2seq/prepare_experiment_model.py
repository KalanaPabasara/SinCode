"""
Prepare local clean model snapshots and experiment copies.

Workflow:
1) Download/save a clean Hugging Face model to a stable local path once.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.constants import DEFAULT_MBART_MODEL

CLEAN_ROOT = ROOT / "seq2seq" / "clean_models"
EXPERIMENT_ROOT = ROOT / "seq2seq" / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a clean model once and create an isolated experiment copy (GPU required)."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MBART_MODEL,
        help="Hugging Face model ID to prepare.",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=None,
        help="Optional custom clean-model directory.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional experiment run folder name. Defaults to timestamp.",
    )
    parser.add_argument(
        "--force-refresh-clean",
        action="store_true",
        help="Re-download and overwrite the local clean model snapshot.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow running without CUDA. Default is GPU-only to avoid workstation slowdown.",
    )
    return parser.parse_args()


def safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU is required by default. "
            "No GPU detected. Use --allow-cpu only if you intentionally want CPU mode."
        )

    model_slug = safe_name(args.model_id)
    clean_dir = args.clean_dir or (CLEAN_ROOT / model_slug)

    if clean_dir.exists() and args.force_refresh_clean:
        print(f"Removing existing clean model at: {clean_dir}")
        shutil.rmtree(clean_dir)

    if not clean_dir.exists():
        print(f"Downloading clean model: {args.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)

        clean_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(clean_dir)
        model.save_pretrained(clean_dir)
        print(f"Saved clean model to: {clean_dir}")
    else:
        print(f"Using existing clean model: {clean_dir}")

    run_name = args.run_name or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    exp_dir = EXPERIMENT_ROOT / model_slug / run_name
    exp_model_dir = exp_dir / "model"

    if exp_model_dir.exists():
        raise FileExistsError(
            f"Experiment model directory already exists: {exp_model_dir}. "
            "Use a different --run-name."
        )

    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(clean_dir, exp_model_dir)

    print("\nExperiment ready")
    print(f"  clean_model : {clean_dir}")
    print(f"  experiment  : {exp_dir}")
    print(f"  model_copy  : {exp_model_dir}")


if __name__ == "__main__":
    main()
