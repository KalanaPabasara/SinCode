"""
SinCode v3 — Evaluation Script

Supports two evaluation modes selected via --mode:

  system       Full v3 pipeline (ByT5 + two-pass MLM).  Default.
  ablation     Side-by-side comparison of two configurations:
                 (A) ByT5 top-1 only  — no MLM reranking
                 (B) ByT5 + MLM       — full Code-Mixed pipeline
               Proves the contribution of the XLM-RoBERTa reranker.

Note: mBart50 is intentionally excluded from evaluation here because the
reference dataset uses code-mixed targets (English words preserved).  mBart50
produces full-Sinhala output by design, making a metric comparison against
code-mixed references invalid.  Evaluate mBart50 separately with a dataset
whose references are fully in Sinhala script.

Usage:
    python misc/evaluate.py --dataset misc/dataset_110.csv
    python misc/evaluate.py --dataset misc/dataset_110.csv --mode ablation
    python misc/evaluate.py --dataset misc/dataset_110.csv --mode ablation --out misc/results.csv

CSV columns required: id, input, reference
Optional columns (used for grouping): category, domain, has_code_mix, has_ambiguity
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

# ── Path setup ────────────────────────────────────────────────────────────────

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING)

# ── Metrics ───────────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[-1]


def _levenshtein_tokens(a: list, b: list) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ta in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, tb in enumerate(b, 1):
            cost = 0 if ta == tb else 1
            curr[j] = min(prev[j] + 1, curr[j-1] + 1, prev[j-1] + cost)
        prev = curr
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    if not ref: return 0.0 if not pred else 1.0
    return _levenshtein(pred, ref) / max(len(ref), 1)


def wer(pred: str, ref: str) -> float:
    pt, rt = pred.split(), ref.split()
    if not rt: return 0.0 if not pt else 1.0
    return _levenshtein_tokens(pt, rt) / max(len(rt), 1)


def token_accuracy(pred: str, ref: str) -> float:
    pt, rt = pred.split(), ref.split()
    if not rt: return 0.0 if pt else 1.0
    return sum(p == r for p, r in zip(pt, rt)) / max(len(rt), 1)


def bleu(pred: str, ref: str, max_n: int = 4) -> float:
    from collections import Counter
    pt, rt = pred.split(), ref.split()
    if not pt or not rt: return 0.0
    n_max = min(max_n, len(pt), len(rt))
    if n_max == 0: return 0.0
    brevity = min(1.0, len(pt) / len(rt))
    log_avg = 0.0
    for n in range(1, n_max + 1):
        pc = Counter(tuple(pt[i:i+n]) for i in range(len(pt)-n+1))
        rc = Counter(tuple(rt[i:i+n]) for i in range(len(rt)-n+1))
        clipped = sum(min(c, rc[ng]) for ng, c in pc.items())
        total = max(sum(pc.values()), 1)
        prec = clipped / total
        if prec == 0: return 0.0
        log_avg += math.log(prec) / n_max
    return brevity * math.exp(log_avg)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip() == ref.strip() else 0.0


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: int
    input: str
    reference: str
    domain: str = "general"
    has_code_mix: bool = False
    has_ambiguity: bool = False


@dataclass
class Result:
    test_case: TestCase
    system: str
    prediction: str
    cer_score: float
    wer_score: float
    token_acc: float
    bleu_score: float
    exact: float


def _score(tc: TestCase, pred: str, system: str) -> Result:
    return Result(
        test_case=tc,
        system=system,
        prediction=pred,
        cer_score=cer(pred, tc.reference),
        wer_score=wer(pred, tc.reference),
        token_acc=token_accuracy(pred, tc.reference),
        bleu_score=bleu(pred, tc.reference),
        exact=exact_match(pred, tc.reference),
    )


# ── Test set loader ───────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> List[TestCase]:
    cases = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not {"input", "reference"}.issubset(fields):
            raise ValueError(f"CSV must have 'input' and 'reference' columns. Found: {fields}")
        for row in reader:
            inp = (row.get("input") or "").strip().replace("\n", " ")
            ref = (row.get("reference") or "").strip().replace("\n", " ")
            if not inp or not ref:
                continue
            cases.append(TestCase(
                id=int(row.get("id") or 0),
                input=inp,
                reference=ref,
                domain=(row.get("domain") or row.get("category") or "general").strip(),
                has_code_mix=bool(int(row.get("has_code_mix") or 0)),
                has_ambiguity=bool(int(row.get("has_ambiguity") or 0)),
            ))
    return cases


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_v3_decoder():
    from sincode_model import BeamSearchDecoder
    print("  Loading ByT5 + XLM-RoBERTa (Code-Mixed pipeline)...")
    return BeamSearchDecoder()


def _byt5_top1_predict(decoder, sentence: str) -> str:
    """ByT5 top-1 only — pick first beam candidate, skip MLM reranking."""
    from core.constants import PUNCT_PATTERN
    from core.decoder import _is_sinhala

    words = sentence.split()
    output = []
    cores = [re.sub(r"^\W*|\W*$", "", w) for w in words]
    non_sinhala = [c for c in cores if not _is_sinhala(c) and c]

    if not non_sinhala:
        return sentence

    byt5_results = decoder.transliterator.batch_candidates(non_sinhala, k=1)
    byt5_iter = iter(byt5_results)

    for raw, core in zip(words, cores):
        m = PUNCT_PATTERN.match(raw)
        prefix, _, suffix = m.groups() if m else ("", raw, "")
        if _is_sinhala(core) or not core:
            output.append(raw)
        else:
            cands = next(byt5_iter, [core])
            output.append(prefix + (cands[0] if cands else core) + suffix)
    return " ".join(output)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _avg(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _print_table(label: str, results: List[Result]):
    print(f"\n{'='*74}")
    print(f"  {label}  (n={len(results)})")
    print(f"{'='*74}")
    print(f"  {'ID':<5} {'Domain':<14} {'CM':>3} {'Am':>3}  {'CER':>6} {'WER':>6} {'TokAcc':>7} {'BLEU':>6} {'EM':>4}")
    print(f"  {'-'*66}")
    for r in results:
        tc = r.test_case
        print(
            f"  {tc.id:<5} {tc.domain[:13]:<14} {'Y' if tc.has_code_mix else 'N':>3} "
            f"{'Y' if tc.has_ambiguity else 'N':>3}  "
            f"{r.cer_score:>6.3f} {r.wer_score:>6.3f} {r.token_acc:>7.3f} "
            f"{r.bleu_score:>6.3f} {r.exact:>4.0f}"
        )
    print(f"  {'-'*66}")
    print(
        f"  {'AVERAGE':<26}  "
        f"{_avg([r.cer_score for r in results]):>6.3f} "
        f"{_avg([r.wer_score for r in results]):>6.3f} "
        f"{_avg([r.token_acc for r in results]):>7.3f} "
        f"{_avg([r.bleu_score for r in results]):>6.3f} "
        f"{_avg([r.exact for r in results]):>4.2f}"
    )

    # Per-domain breakdown
    by_domain: Dict[str, List[Result]] = defaultdict(list)
    for r in results:
        by_domain[r.test_case.domain].append(r)
    if len(by_domain) > 1:
        print(f"\n  Per-domain averages (CER / WER / TokAcc):")
        for dom, rs in sorted(by_domain.items()):
            print(
                f"    {dom:<18}  n={len(rs):<4} "
                f"CER={_avg([r.cer_score for r in rs]):.3f}  "
                f"WER={_avg([r.wer_score for r in rs]):.3f}  "
                f"TokAcc={_avg([r.token_acc for r in rs]):.3f}"
            )

    # Code-mixed vs pure Singlish
    cm_r  = [r for r in results if r.test_case.has_code_mix]
    pure_r = [r for r in results if not r.test_case.has_code_mix]
    if cm_r and pure_r:
        print(
            f"\n  Code-mixed  (n={len(cm_r):<3}):  "
            f"CER={_avg([r.cer_score for r in cm_r]):.3f}  "
            f"WER={_avg([r.wer_score for r in cm_r]):.3f}"
        )
        print(
            f"  Pure Singlish (n={len(pure_r):<3}):  "
            f"CER={_avg([r.cer_score for r in pure_r]):.3f}  "
            f"WER={_avg([r.wer_score for r in pure_r]):.3f}"
        )


def _print_ablation(a_res: List[Result], b_res: List[Result]):
    print(f"\n{'='*74}")
    print("  ABLATION STUDY — MLM Reranking Contribution")
    print(f"  (A) ByT5 top-1 only  |  (B) ByT5 + XLM-RoBERTa MLM reranking")
    print(f"{'='*74}")
    print(f"  {'Metric':<22}  {'(A) ByT5-top1':>14}  {'(B) ByT5+MLM':>13}  {'Δ (B−A)':>10}")
    print(f"  {'-'*64}")

    metrics = [
        ("CER (↓ better)",  [r.cer_score  for r in a_res], [r.cer_score  for r in b_res], True),
        ("WER (↓ better)",  [r.wer_score  for r in a_res], [r.wer_score  for r in b_res], True),
        ("Token Acc (↑)",   [r.token_acc  for r in a_res], [r.token_acc  for r in b_res], False),
        ("BLEU (↑ better)", [r.bleu_score for r in a_res], [r.bleu_score for r in b_res], False),
        ("Exact Match (↑)", [r.exact      for r in a_res], [r.exact      for r in b_res], False),
    ]

    for label, a_vals, b_vals, lower_is_better in metrics:
        a_avg, b_avg = _avg(a_vals), _avg(b_vals)
        delta = b_avg - a_avg
        improved = (delta < 0) if lower_is_better else (delta > 0)
        print(
            f"  {label:<22}  {a_avg:>14.4f}  {b_avg:>13.4f}  "
            f"  {'✓' if improved else '✗'}{delta:>+8.4f}"
        )

    print(f"\n  ✓ B vs A isolates the contribution of XLM-RoBERTa MLM reranking.")
    print(f"  ✓ If B > A: the two-pass reranker justifies its computational cost.")

    # Subcategory breakdown
    for sublabel, filter_fn in [
        ("Code-mixed only",   lambda r: r.test_case.has_code_mix),
        ("Ambiguous only",    lambda r: r.test_case.has_ambiguity),
        ("Pure Singlish",     lambda r: not r.test_case.has_code_mix),
    ]:
        a_sub = [r for r in a_res if filter_fn(r)]
        b_sub = [r for r in b_res if filter_fn(r)]
        if not a_sub:
            continue
        print(f"\n  {sublabel} (n={len(a_sub)}):")
        print(f"    {'':20}  {'(A)':>10}  {'(B)':>10}  {'Δ':>10}")
        for ml, getter, low in [("CER", lambda r: r.cer_score, True), ("WER", lambda r: r.wer_score, True), ("TokAcc", lambda r: r.token_acc, False)]:
            av, bv = _avg([getter(r) for r in a_sub]), _avg([getter(r) for r in b_sub])
            d = bv - av
            imp = (d < 0) if low else (d > 0)
            print(
                f"    {ml:<20}  {av:>10.4f}  {bv:>10.4f}  "
                f"  {'✓' if imp else '✗'}{d:>+7.4f}"
            )


def _load_baseline(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_v2_comparison(b_res: List[Result], baseline: dict):
    n = len(b_res)
    v3 = {
        "exact_match": _avg([r.exact      for r in b_res]),
        "cer":         _avg([r.cer_score  for r in b_res]),
        "wer":         _avg([r.wer_score  for r in b_res]),
        "bleu":        _avg([r.bleu_score for r in b_res]),
        "token_acc":   _avg([r.token_acc  for r in b_res]),
    }
    v2_label = baseline.get("system", "v2 baseline")

    print(f"\n{'='*74}")
    print(f"  SinCode v2  vs  SinCode v3  —  Head-to-Head  (n={n})")
    print(f"  v2: {v2_label}")
    print(f"  v3: ByT5-small seq2seq + XLM-RoBERTa MLM reranking")
    print(f"{'='*74}")
    print(f"  {'Metric':<22}  {'v2 (baseline)':>14}  {'v3 (ours)':>10}  {'Δ (v3−v2)':>12}")
    print(f"  {'-'*62}")

    metrics = [
        ("Exact Match (↑)", "exact_match", False),
        ("CER (↓ better)",  "cer",         True),
        ("WER (↓ better)",  "wer",         True),
        ("BLEU (↑ better)", "bleu",        False),
        ("Token Acc (↑)",   "token_acc",   False),
    ]
    for label, key, lower_is_better in metrics:
        v2v = baseline.get(key, 0.0)
        v3v = v3[key]
        delta = v3v - v2v
        improved = (delta < 0) if lower_is_better else (delta > 0)
        arrow = "↑" if (delta > 0) else ("↓" if delta < 0 else "=")
        print(
            f"  {label:<22}  {v2v:>14.4f}  {v3v:>10.4f}  "
            f"  {'✓' if improved else '✗'} {arrow}{abs(delta):>+8.4f}"
        )

    if baseline.get("notes"):
        print(f"\n  Note: {baseline['notes']}")


def _save_csv(results_by_system: Dict[str, List[Result]], out_path: str):
    rows = []
    for system, results in results_by_system.items():
        for r in results:
            rows.append({
                "system": system,
                "id": r.test_case.id,
                "domain": r.test_case.domain,
                "has_code_mix": int(r.test_case.has_code_mix),
                "has_ambiguity": int(r.test_case.has_ambiguity),
                "input": r.test_case.input,
                "reference": r.test_case.reference,
                "prediction": r.prediction,
                "cer": f"{r.cer_score:.4f}",
                "wer": f"{r.wer_score:.4f}",
                "token_acc": f"{r.token_acc:.4f}",
                "bleu": f"{r.bleu_score:.4f}",
                "exact_match": f"{r.exact:.0f}",
            })
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Results saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SinCode v3 evaluation")
    parser.add_argument("--dataset", required=True,
                        help="Path to evaluation CSV (dataset_110.csv or dataset_40.csv)")
    parser.add_argument("--mode", default="system",
                        choices=["system", "ablation"],
                        help="Evaluation mode (default: system)")
    parser.add_argument("--out", default=None,
                        help="Optional path to save results CSV")
    parser.add_argument("--baseline", default=None,
                        help="Path to v2 baseline JSON (e.g. misc/v2_baseline.json) for head-to-head comparison")
    args = parser.parse_args()

    print(f"\nLoading dataset: {args.dataset}")
    test_cases = load_dataset(args.dataset)
    print(f"  {len(test_cases)} test cases loaded.")

    results_by_system: Dict[str, List[Result]] = {}
    a_results: List[Result] = []
    b_results: List[Result] = []

    decoder = _load_v3_decoder()

    if args.mode == "ablation":
        print("\nRunning (A) ByT5 top-1 only...")
        a_results = [_score(tc, _byt5_top1_predict(decoder, tc.input), "byt5_top1") for tc in test_cases]
        results_by_system["byt5_top1"] = a_results

    print("\nRunning (B) ByT5 + MLM reranking...")
    b_results = [_score(tc, decoder.decode(tc.input)[0], "byt5_mlm") for tc in test_cases]
    results_by_system["byt5_mlm"] = b_results

    if args.mode == "system":
        _print_table("v3 Code-Mixed Pipeline  (ByT5 + XLM-RoBERTa MLM)", b_results)
    elif args.mode == "ablation":
        _print_table("(A) ByT5 top-1 only", a_results)
        _print_table("(B) ByT5 + MLM reranking", b_results)
        _print_ablation(a_results, b_results)

    if args.baseline:
        baseline = _load_baseline(args.baseline)
        _print_v2_comparison(b_results, baseline)

    if args.out:
        _save_csv(results_by_system, args.out)


if __name__ == "__main__":
    main()

