"""
Fine-tune google/byt5-small on Singlish → Sinhala word-level transliteration.

Input:  wsd_pairs.csv  (romanized, sinhala)
Output: byt5-singlish-sinhala/  (HuggingFace model directory)

Training approach:
  - Input  : romanized word  (e.g. "wadi")
  - Target : sinhala word    (e.g. "වැඩි")
  - Model  : ByT5-small (byte-level T5, no vocab issues with any script)
  - Beam=5 at inference → top-5 candidates for MLM reranking

  Tokenized dataset is saved to disk after first run — restarts skip
  straight to training without re-tokenizing.
"""

from pathlib import Path

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

# ── Config ─────────────────────────────────────────────────────────────────

BASE_MODEL      = "google/byt5-small"
DATA_PATH       = Path(__file__).parent / "wsd_pairs.csv"
CACHE_DIR       = Path(__file__).parent / "tokenized_cache"
OUTPUT_DIR      = Path(__file__).parent / "byt5-singlish-sinhala"

MAX_SAMPLES     = 1_000_000   # 1M pairs — more than enough for word transliteration
TRAIN_SPLIT     = 0.97
MAX_INPUT_LEN   = 64
MAX_TARGET_LEN  = 64
BATCH_SIZE      = 64    # 16GB VRAM — ByT5-small with seq_len=64
EPOCHS          = 2
LR              = 5e-4
SEED            = 42


# ── Tokenize ────────────────────────────────────────────────────────────────

def tokenize_fn(batch, tokenizer):
    # Pad to fixed max_length so all tensors have the same shape.
    # This lets set_format("torch") work and default_data_collator just stacks.
    model_inputs = tokenizer(
        batch["romanized"],
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="max_length",
    )
    labels = tokenizer(
        batch["sinhala"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding="max_length",
    )
    # Replace pad token with -100 so it's ignored in cross-entropy loss
    model_inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in ids]
        for ids in labels["input_ids"]
    ]
    return model_inputs


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")
    if device != "cuda":
        raise RuntimeError(
            "CUDA GPU is required for training. "
            "No GPU was detected, so training was stopped to avoid CPU slowdown."
        )

    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    train_cache = CACHE_DIR / "train"
    eval_cache  = CACHE_DIR / "eval"

    if train_cache.exists() and eval_cache.exists():
        print("Loading pre-tokenized dataset from disk cache …")
        train_ds = load_from_disk(str(train_cache))
        eval_ds  = load_from_disk(str(eval_cache))
        print(f"  train={len(train_ds):,}  eval={len(eval_ds):,}")
    else:
        print(f"Loading data from {DATA_PATH} …")
        ds = Dataset.from_csv(str(DATA_PATH))
        ds = ds.filter(lambda x: bool(x["romanized"]) and bool(x["sinhala"]))
        print(f"  {len(ds):,} pairs — sampling {MAX_SAMPLES:,} …")

        # Shuffle and take MAX_SAMPLES
        ds = ds.shuffle(seed=SEED).select(range(min(MAX_SAMPLES, len(ds))))

        split    = ds.train_test_split(test_size=1 - TRAIN_SPLIT, seed=SEED)
        train_raw = split["train"]
        eval_raw  = split["test"]
        print(f"  train={len(train_raw):,}  eval={len(eval_raw):,}")

        print("Tokenizing and saving to disk (one-time, ~5 min) …")
        train_ds = train_raw.map(
            lambda b: tokenize_fn(b, tokenizer),
            batched=True,
            batch_size=10_000,
            num_proc=8,
            keep_in_memory=True,
            remove_columns=["romanized", "sinhala"],
            desc="Tokenizing train",
        )
        eval_ds = eval_raw.map(
            lambda b: tokenize_fn(b, tokenizer),
            batched=True,
            batch_size=10_000,
            num_proc=8,
            keep_in_memory=True,
            remove_columns=["romanized", "sinhala"],
            desc="Tokenizing eval",
        )

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        train_ds.save_to_disk(str(train_cache))
        eval_ds.save_to_disk(str(eval_cache))
        print("  Saved to disk. Future runs will load instantly.")

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    # All sequences are pre-padded to fixed length — just stack them
    collator     = default_data_collator
    warmup_steps = int(0.05 * (len(train_ds) // BATCH_SIZE))

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=200,
        dataloader_num_workers=0,   # 0 = main process only (most stable on Windows)
        dataloader_pin_memory=True,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        seed=SEED,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
    )

    print("Starting training …")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}/final …")
    model.save_pretrained(OUTPUT_DIR / "final")
    tokenizer.save_pretrained(OUTPUT_DIR / "final")
    print("Done.")


if __name__ == "__main__":
    main()


