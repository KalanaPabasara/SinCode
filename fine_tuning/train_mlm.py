"""
Continued MLM pre-training of XLM-RoBERTa on Sinhala text.

Experiment 1 (completed): Sinhala Wikipedia (23K articles) — no improvement.
Experiment 2 (current):   9wimu9/sinhala_dataset_59m — 500K informal samples.

Usage:
    python train_mlm.py                       # full training (500K, 1 epoch)
    python train_mlm.py --samples 100 --test  # quick smoke test
    python train_mlm.py --samples 1000000     # 1M samples
"""

import argparse
import os
import math
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset

# ─── Defaults ────────────────────────────────────────────────────────────────

BASE_MODEL = "FacebookAI/xlm-roberta-base"
OUTPUT_DIR = "xlm-roberta-sinhala-v2"       # saved model directory (v2 = informal data)
DATASET = "9wimu9/sinhala_dataset_59m"      # 59M mixed-register Sinhala samples
DEFAULT_SAMPLES = 500_000                   # subset size (full 59M is ~15 days)
MAX_SEQ_LEN = 256                           # token block size
MLM_PROB = 0.15                             # mask probability (same as original)


def parse_args():
    p = argparse.ArgumentParser(description="Continue MLM pre-training on Sinhala text")
    p.add_argument("--base_model", default=BASE_MODEL, help="Base HuggingFace model")
    p.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for fine-tuned model")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs (1 is enough for 500K)")
    p.add_argument("--batch_size", type=int, default=8, help="Per-device train batch size")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--max_seq_len", type=int, default=MAX_SEQ_LEN, help="Max sequence length")
    p.add_argument("--samples", type=int, default=DEFAULT_SAMPLES, help="Number of samples to use from dataset")
    p.add_argument("--test", action="store_true", help="Quick smoke test with 100 samples")
    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    return p.parse_args()


def load_and_prepare_dataset(tokenizer, max_seq_len, num_samples, test_mode=False):
    """Download Sinhala dataset (streaming) and tokenize a subset."""
    if test_mode:
        num_samples = 100

    print(f"📥  Loading {DATASET} (streaming {num_samples:,} samples)...")
    ds = load_dataset(DATASET, split="train", streaming=True)

    # Collect samples from the stream
    texts = []
    for i, row in enumerate(ds):
        if i >= num_samples:
            break
        text = row.get("text", "")
        if len(text.strip()) >= 10:  # skip near-empty rows
            texts.append(text)
        if (i + 1) % 50_000 == 0:
            print(f"    ... loaded {i + 1:,} / {num_samples:,}")

    print(f"📊  Collected {len(texts):,} samples (after filtering empty rows)")

    # Convert to HF Dataset for .map() compatibility
    from datasets import Dataset
    raw = Dataset.from_dict({"text": texts})
    del texts  # free memory

    # Tokenize
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            return_special_tokens_mask=True,
        )

    print("🔤  Tokenizing...")
    tokenized = raw.map(
        tokenize_fn,
        batched=True,
        num_proc=4 if not test_mode else 1,
        remove_columns=raw.column_names,
        desc="Tokenizing",
    )

    # Filter out very short sequences (< 20 tokens)
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) >= 20)

    print(f"✅  {len(tokenized):,} tokenized samples ready")
    return tokenized


def main():
    args = parse_args()

    # ─── Device check ────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"🖥️  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("⚠️  No GPU detected — training will be slow!")

    # ─── Load tokenizer & model ──────────────────────────────────────────
    print(f"📦  Loading {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    # ─── Dataset ─────────────────────────────────────────────────────────
    dataset = load_and_prepare_dataset(tokenizer, args.max_seq_len, args.samples, args.test)

    # Split 95/5 for train/validation
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"🔀  Train: {len(train_dataset):,}  |  Eval: {len(eval_dataset):,}")

    # ─── Data collator (dynamic masking each epoch) ──────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROB,
    )

    # ─── Training arguments ──────────────────────────────────────────────
    # Effective batch = batch_size * grad_accum = 8 * 4 = 32
    total_steps = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum)) * args.epochs

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=max(100, total_steps // 16),
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=max(500, total_steps // 10),
        save_strategy="steps",
        save_steps=max(500, total_steps // 10),
        save_total_limit=2,
        logging_steps=50,
        fp16=device == "cuda",
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",            # no wandb/tensorboard
        seed=42,
    )

    # ─── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # ─── Train ───────────────────────────────────────────────────────────
    print("🚀  Starting training...")
    resume_checkpoint = args.resume and os.path.isdir(args.output_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint if resume_checkpoint else None)

    # ─── Save final model ────────────────────────────────────────────────
    final_path = os.path.join(args.output_dir, "final")
    print(f"💾  Saving fine-tuned model to {final_path}/")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    # ─── Final eval ──────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print(f"\n📈  Final eval loss: {metrics['eval_loss']:.4f}")
    print(f"    Perplexity:     {math.exp(metrics['eval_loss']):.2f}")
    print(f"\n✅  Model saved to: {os.path.abspath(final_path)}")
    print(f"    To use in SinCode, update DEFAULT_MODEL_NAME in core/constants.py to:")
    print(f'    DEFAULT_MODEL_NAME = r"{os.path.abspath(final_path)}"')


if __name__ == "__main__":
    main()
