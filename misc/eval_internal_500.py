#!/usr/bin/env python3
"""
Evaluate ByT5 + XLM-RoBERTa reranker on internal_test_set_500.csv.
CSV columns: id, category, input, code_mixed_reference

Usage:
    python misc/eval_internal_500.py                   # full 500
    python misc/eval_internal_500.py --max 10          # dry run
    python misc/eval_internal_500.py --max 10 --cat colloquial
"""

import sys
import os
import argparse
import csv
import time
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.decoder import BeamSearchDecoder


# ── Metrics ─────────────────────────────────────────────────────────────────

def levenshtein(a, b):
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + (0 if ca == cb else 1))
        prev = curr
    return prev[-1]

def cer(pred, ref):
    if not ref: return 0.0 if not pred else 1.0
    return levenshtein(pred, ref) / len(ref)

def wer(pred, ref):
    pw, rw = pred.split(), ref.split()
    if not rw: return 0.0 if not pw else 1.0
    return levenshtein(pw, rw) / len(rw)

def bleu1(pred, ref):
    pt, rt = pred.split(), ref.split()
    if not rt: return 1.0 if not pt else 0.0
    matches = sum(1 for t in pt if t in rt)
    return matches / len(rt)

def exact_match(pred, ref):
    return pred.strip() == ref.strip()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--cat", type=str, default=None, help="Filter to one category")
    parser.add_argument("--out", type=str, default="misc/internal_500_results.csv", help="Output CSV path")
    args = parser.parse_args()

    csv_path = project_root / "misc" / "internal_test_set_500.csv"

    # Load samples
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.cat and row["category"] != args.cat:
                continue
            samples.append(row)
            if args.max and len(samples) >= args.max:
                break

    print(f"Loaded {len(samples)} samples" + (f" (category={args.cat})" if args.cat else ""))

    # Load decoder
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading BeamSearchDecoder on {device}...")
    decoder = BeamSearchDecoder(device=device)
    print("Ready.\n")

    results = []
    category_stats = defaultdict(list)

    for i, row in enumerate(samples):
        inp = row["input"].strip()
        ref = row["code_mixed_reference"].strip()
        cat = row["category"]
        sid = row["id"]

        t0 = time.time()
        try:
            pred, _, _ = decoder.decode(inp)
        except Exception as e:
            print(f"  ERROR id={sid}: {e}")
            pred = "[ERROR]"
        elapsed = time.time() - t0

        c = cer(pred, ref)
        w = wer(pred, ref)
        b = bleu1(pred, ref)
        em = exact_match(pred, ref)

        result = {
            "id": sid,
            "category": cat,
            "input": inp,
            "reference": ref,
            "prediction": pred,
            "cer": round(c, 4),
            "wer": round(w, 4),
            "bleu": round(b, 4),
            "exact_match": em,
            "time_s": round(elapsed, 3),
        }
        results.append(result)
        category_stats[cat].append(result)

        status = "PASS" if em else "FAIL"
        print(f"[{i+1:>4}/{len(samples)}] {status} id={sid:>4} cat={cat:<15} CER={c:.3f} WER={w:.3f} BLEU={b:.3f}")
        if not em:
            print(f"         IN:  {inp}")
            print(f"         REF: {ref}")
            print(f"         GOT: {pred}")

    # Write results CSV
    out_path = project_root / args.out
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")

    # Summary by category
    print(f"\n{'='*70}")
    print(f"{'CATEGORY':<18} {'n':>4}  {'CER':>7}  {'WER':>7}  {'BLEU':>7}  {'EM%':>7}")
    print(f"{'='*70}")

    all_results = results
    for cat in sorted(category_stats):
        rows = category_stats[cat]
        n = len(rows)
        avg_cer  = sum(r["cer"]  for r in rows) / n
        avg_wer  = sum(r["wer"]  for r in rows) / n
        avg_bleu = sum(r["bleu"] for r in rows) / n
        em_pct   = sum(1 for r in rows if r["exact_match"]) / n * 100
        print(f"{cat:<18} {n:>4}  {avg_cer:>7.4f}  {avg_wer:>7.4f}  {avg_bleu:>7.4f}  {em_pct:>6.1f}%")

    n = len(all_results)
    print(f"{'─'*70}")
    print(f"{'OVERALL':<18} {n:>4}  "
          f"{sum(r['cer'] for r in all_results)/n:>7.4f}  "
          f"{sum(r['wer'] for r in all_results)/n:>7.4f}  "
          f"{sum(r['bleu'] for r in all_results)/n:>7.4f}  "
          f"{sum(1 for r in all_results if r['exact_match'])/n*100:>6.1f}%")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
