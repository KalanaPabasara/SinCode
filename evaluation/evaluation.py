import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict
from typing import Dict, List, Tuple

# Ensure parent dir is on path so sincode_model can be imported from misc/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sincode_model import BeamSearchDecoder

ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*")


# ── String-level metrics ────────────────────────────────────────────────────

def levenshtein(a: str, b: str) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / max(len(ref), 1)


def wer(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not ref_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_tokens(pred_tokens, ref_tokens) / max(len(ref_tokens), 1)


def levenshtein_tokens(a: list, b: list) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, tb in enumerate(b, start=1):
            cost = 0 if ta == tb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def bleu_sentence(pred: str, ref: str, max_n: int = 4) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Cap n-gram order at the shorter sentence length
    effective_n = min(max_n, len(pred_tokens), len(ref_tokens))
    if effective_n == 0:
        return 0.0

    brevity = min(1.0, len(pred_tokens) / len(ref_tokens))
    log_avg = 0.0
    for n in range(1, effective_n + 1):
        pred_ngrams = Counter(
            tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(c, ref_ngrams[ng]) for ng, c in pred_ngrams.items())
        total = max(sum(pred_ngrams.values()), 1)
        precision = clipped / total
        if precision == 0:
            return 0.0
        log_avg += math.log(precision) / effective_n

    return brevity * math.exp(log_avg)


def token_accuracy(pred: str, ref: str) -> float:
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if not ref_tokens:
        return 0.0 if pred_tokens else 1.0

    matches = sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
    return matches / max(len(ref_tokens), 1)


def extract_english_tokens(text: str) -> List[str]:
    return [m.group(0) for m in ASCII_WORD_RE.finditer(text)]


def code_mix_preservation(input_text: str, ref_text: str, pred_text: str) -> float:
    """Measure how well English tokens from the reference are preserved.
    Only counts English words that appear in the REFERENCE (not raw input,
    since the input is all ASCII).  Returns 1.0 if no English in reference."""
    ref_eng = extract_english_tokens(ref_text)
    if not ref_eng:
        return 1.0

    pred_tokens = set(pred_text.split())
    preserved = sum(1 for token in ref_eng if token in pred_tokens)
    return preserved / len(ref_eng)


def load_dataset(csv_path: str) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "input" not in reader.fieldnames or "reference" not in reader.fieldnames:
            raise ValueError("CSV must contain 'input' and 'reference' columns")

        for row in reader:
            src = (row.get("input") or "").strip()
            ref = (row.get("reference") or "").strip()
            if src:
                rows.append((src, ref))
    return rows


def evaluate(
    decoder: BeamSearchDecoder,
    dataset: List[Tuple[str, str]],
    mode: str = "greedy",
    beam_width: int = 5,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    details: List[Dict[str, object]] = []
    exact = 0
    total_cer = 0.0
    total_wer = 0.0
    total_bleu = 0.0
    total_token_acc = 0.0
    total_code_mix = 0.0
    total_time = 0.0

    for idx, (src, ref) in enumerate(dataset, start=1):
        t0 = time.perf_counter()
        if mode == "greedy":
            pred, _, diagnostics = decoder.greedy_decode_with_diagnostics(src)
        else:
            pred, _, diagnostics = decoder.decode_with_diagnostics(
                src, beam_width=beam_width
            )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        is_exact = int(pred == ref)
        exact += is_exact

        sample_cer = cer(pred, ref)
        sample_wer = wer(pred, ref)
        sample_bleu = bleu_sentence(pred, ref)
        sample_token_acc = token_accuracy(pred, ref)
        sample_code_mix = code_mix_preservation(src, ref, pred)

        total_cer += sample_cer
        total_wer += sample_wer
        total_bleu += sample_bleu
        total_token_acc += sample_token_acc
        total_code_mix += sample_code_mix

        details.append({
            "id": idx,
            "input": src,
            "reference": ref,
            "prediction": pred,
            "exact_match": bool(is_exact),
            "cer": round(sample_cer, 4),
            "wer": round(sample_wer, 4),
            "bleu": round(sample_bleu, 4),
            "token_accuracy": round(sample_token_acc, 4),
            "code_mix_preservation": round(sample_code_mix, 4),
            "time_s": round(elapsed, 3),
        })

    n = max(len(dataset), 1)
    metrics = {
        "mode": mode,
        "samples": len(dataset),
        "exact_match": round(exact / n, 4),
        "exact_match_count": f"{exact}/{len(dataset)}",
        "avg_cer": round(total_cer / n, 4),
        "avg_wer": round(total_wer / n, 4),
        "avg_bleu": round(total_bleu / n, 4),
        "avg_token_accuracy": round(total_token_acc / n, 4),
        "avg_code_mix_preservation": round(total_code_mix / n, 4),
        "total_time_s": round(total_time, 2),
        "avg_time_per_sentence_s": round(total_time / n, 3),
    }
    return metrics, details


def write_predictions(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "input",
                "reference",
                "prediction",
                "exact_match",
                "cer",
                "wer",
                "bleu",
                "token_accuracy",
                "code_mix_preservation",
                "time_s",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def write_diagnostics(path: str, rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SinCode transliteration quality on a CSV dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CSV with columns: input,reference",
    )
    parser.add_argument(
        "--mode",
        choices=["greedy", "beam"],
        default="greedy",
        help="Decode mode (default: greedy)",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=5,
        help="Beam width used during decoding (default: 5, only for beam mode)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Hugging Face model name or local path to evaluate",
    )
    parser.add_argument(
        "--predictions-out",
        default="eval_predictions.csv",
        help="Output CSV path for per-sample predictions",
    )
    parser.add_argument(
        "--diagnostics-out",
        default="eval_diagnostics.json",
        help="Output JSON path with per-word diagnostics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset)
    if not dataset:
        raise ValueError("Dataset is empty. Add rows with input/reference values.")

    decoder = BeamSearchDecoder(model_name=args.model) if args.model else BeamSearchDecoder()
    metrics, details = evaluate(
        decoder, dataset, mode=args.mode, beam_width=args.beam_width
    )

    write_predictions(args.predictions_out, details)
    write_diagnostics(args.diagnostics_out, details)

    print("\n" + "=" * 60)
    print("  SinCode Evaluation Results")
    print("=" * 60)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nPredictions saved to: {args.predictions_out}")
    print(f"Diagnostics saved to: {args.diagnostics_out}")


if __name__ == "__main__":
    main()
