"""
seq2seq/finetune_corrections.py

Targeted correction fine-tune for the already-trained ByT5 model.

Problem: ByT5 struggles with short/ambiguous tokens like "na"→නෑ, "ba"→බෑ,
         extreme abbreviations like "mn"→මං, and colloquial negations.

Solution: Inject high-confidence correction pairs (from core/mappings.py)
          heavily repeated, mixed with a random sample of the original
          training data to prevent catastrophic forgetting.

The output is saved to byt5-singlish-sinhala/final/ (overwrites in place).
Run from the project root:
    python seq2seq/finetune_corrections.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH    = ROOT / "seq2seq" / "wsd_pairs.csv"
# Clean base model downloaded from HF Hub — never fine-tuned directly.
# Experiments always read from here and write to a timestamped subfolder.
DEFAULT_MODEL_PATH = ROOT / "seq2seq" / "byt5-base-clean"
EXPERIMENTS_ROOT = ROOT / "seq2seq" / "experiments" / "byt5-corrections"

REPEAT       = 500       # how many times each correction pair is repeated
BG_SAMPLES   = 50_000    # random background pairs from wsd_pairs.csv to prevent forgetting
MAX_INPUT_LEN  = 64
MAX_TARGET_LEN = 64
BATCH_SIZE   = 32
LR           = 5e-5      # low LR — gentle correction, not retraining
EPOCHS       = 1
SEED         = 42

# ── Correction pairs (sourced from core/mappings.py) ─────────────────────────
# Only include pairs where ByT5 is known to be unreliable.
# English-safe tokens (pr, dm, ai…) are excluded — they never reach ByT5.

CORRECTIONS = [
    # negation — most critical
    ("na",          "නෑ"),
    ("naa",         "නෑ"),
    ("ba",          "බෑ"),
    ("bari",        "බැරි"),
    ("bri",         "බැරි"),
    ("nathi",       "නැති"),
    ("nati",        "නැති"),
    ("naththe",     "නැත්තෙ"),
    ("epa",         "එපා"),
    ("ep",          "එපා"),
    # pronouns / first person
    ("mn",          "මං"),
    ("mama",        "මම"),
    ("mage",        "මගේ"),
    ("mge",         "මගේ"),
    ("oya",         "ඔයා"),
    ("oyaa",        "ඔයා"),
    ("api",         "අපි"),
    ("mata",        "මට"),
    ("mta",         "මට"),
    ("oyata",       "ඔයාට"),
    ("oyta",        "ඔයාට"),
    ("oyage",       "ඔයාගේ"),
    ("oyge",        "ඔයාගෙ"),
    ("ape",         "අපේ"),
    # common particles
    ("one",         "ඕනෙ"),
    ("oney",        "ඕනේ"),
    ("on",          "ඕනෙ"),
    ("oni",         "ඕනි"),
    ("hari",        "හරි"),
    ("hri",         "හරි"),
    ("wage",        "වගේ"),
    ("nisa",        "නිසා"),
    ("dan",         "දැන්"),
    ("gena",        "ගැන"),
    # time
    ("heta",        "හෙට"),
    ("hta",         "හෙට"),
    ("ada",         "අද"),
    ("iye",         "ඊයේ"),
    ("kalin",       "කලින්"),
    ("passe",       "පස්සෙ"),
    # abbreviations
    ("mn",          "මං"),
    ("ek",          "එක"),
    ("ekta",        "එකට"),
    ("eke",         "එකේ"),
    ("me",          "මේ"),
    # common words
    ("honda",       "හොඳ"),
    ("hodai",       "හොඳයි"),
    ("gedara",      "ගෙදර"),
    ("wada",        "වැඩ"),
    ("kema",        "කෑම"),
    ("kama",        "කෑම"),
    ("inne",        "ඉන්නෙ"),
    ("inna",        "ඉන්න"),
    ("madi",        "මදි"),
    ("iwara",       "ඉවර"),
    ("iwra",        "ඉවර"),
    # verbal
    ("awa",         "ආවා"),
    ("aawa",        "ආවා"),
    ("giya",        "ගියා"),
    ("una",         "උනා"),
    ("wuna",        "උනා"),
    ("kiwa",        "කිව්වා"),
    ("kiwwa",       "කිව්වා"),
    ("yewwa",       "යැව්වා"),
    ("yawwa",       "යැව්වා"),
    ("damma",       "දැම්මා"),
    ("karanna",     "කරන්න"),
    ("krnna",       "කරන්න"),
    ("balanna",     "බලන්න"),
    ("blnna",       "බලන්න"),
    ("hadanna",     "හදන්න"),
    ("karamu",      "කරමු"),
    ("balamu",      "බලමු"),
    ("yamu",        "යමු"),
    ("hadamu",      "හදමු"),
    ("damu",        "දාමු"),
    ("wenawa",      "වෙනවා"),
    ("wenwa",       "වෙනවා"),
    ("thiyanawa",   "තියෙනවා"),
    ("enawa",       "එනවා"),
    ("yanawa",      "යනවා"),
]


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(tokenizer) -> Dataset:
    import csv

    pairs: list[dict] = []

    # 1. Correction pairs repeated REPEAT times
    for romanized, sinhala in CORRECTIONS:
        for _ in range(REPEAT):
            pairs.append({"romanized": romanized, "sinhala": sinhala})

    correction_count = len(pairs)
    print(f"  Correction pairs: {len(CORRECTIONS)} × {REPEAT} = {correction_count:,}")

    # 2. Background sample from original training data
    bg: list[dict] = []
    with open(DATA_PATH, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = (row.get("romanized") or "").strip()
            s = (row.get("sinhala") or "").strip()
            if r and s:
                bg.append({"romanized": r, "sinhala": s})

    random.seed(SEED)
    random.shuffle(bg)
    bg = bg[:BG_SAMPLES]
    pairs.extend(bg)
    print(f"  Background pairs: {len(bg):,}")
    print(f"  Total dataset   : {len(pairs):,}")

    random.shuffle(pairs)

    ds = Dataset.from_list(pairs)

    def tokenize(batch):
        inputs = tokenizer(
            batch["romanized"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            batch["sinhala"],
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = [
            [(t if t != tokenizer.pad_token_id else -100) for t in ids]
            for ids in targets["input_ids"]
        ]
        return inputs

    ds = ds.map(tokenize, batched=True, batch_size=5_000,
                remove_columns=["romanized", "sinhala"], desc="Tokenizing")
    ds.set_format("torch")
    return ds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ByT5 corrections on an experiment copy (GPU-only)."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Input model directory (experiment copy recommended).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for this run. If omitted, a timestamped experiment folder is created.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training (not recommended). By default training requires CUDA.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    cli = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice : {device}")
    if device != "cuda" and not cli.allow_cpu:
        raise RuntimeError(
            "CUDA GPU is required for fine-tuning. "
            "No GPU was detected, so the run was stopped to avoid CPU slowdown. "
            "If you really need CPU mode, run with --allow-cpu."
        )

    model_path = cli.model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if cli.output_dir is None:
        run_name = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        output_dir = EXPERIMENTS_ROOT / run_name
    else:
        output_dir = cli.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model     = AutoModelForSeq2SeqLM.from_pretrained(str(model_path))
    
    # Explicitly move model to GPU
    model = model.to(device)
    print(f"Model moved to: {device}")

    print("\nBuilding correction dataset ...")
    ds = build_dataset(tokenizer)

    split    = ds.train_test_split(test_size=0.02, seed=SEED)
    train_ds = split["train"]
    eval_ds  = split["test"]
    print(f"  train={len(train_ds):,}  eval={len(eval_ds):,}")

    warmup = max(100, len(train_ds) // (BATCH_SIZE * 20))

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=warmup,
        weight_decay=0.01,
        predict_with_generate=False,  # faster eval — we only care about loss
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        dataloader_num_workers=0,
        seed=SEED,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
    )

    print("\nStarting correction fine-tune ...")
    trainer.train()

    print(f"\nSaving corrected model to {output_dir} ...")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
