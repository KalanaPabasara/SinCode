#!/usr/bin/env python3
"""
Evaluate ByT5 + XLM-RoBERTa reranker on Indo NLP Sinhala test sets.
Test Set 1: 10K formal sentences
Test Set 2: 5K informal sentences (ad-hoc, colloquial)
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd
import numpy as np
from collections import defaultdict

# Import our models
from core.decoder import BeamSearchDecoder

def load_test_set(filepath, max_samples=None):
    """
    Load Indo NLP test set.
    Format: pairs of lines (Singlish, Sinhala expected output)
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            singlish_input = lines[i]
            sinhala_expected = lines[i + 1]
            samples.append({
                'singlish': singlish_input,
                'expected': sinhala_expected
            })
            if max_samples and len(samples) >= max_samples:
                break
    
    return samples

def compute_cer(predicted, expected):
    """Character Error Rate"""
    if not expected:
        return 1.0 if predicted else 0.0
    
    # Simple character-level edit distance
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, predicted, expected)
    ratio = matcher.ratio()
    return 1.0 - ratio

def compute_wer(predicted, expected):
    """Word Error Rate (space-separated tokens)"""
    pred_words = predicted.split()
    exp_words = expected.split()
    
    if not exp_words:
        return 1.0 if pred_words else 0.0
    
    from difflib import SequenceMatcher
    matcher = SequenceMatcher(None, pred_words, exp_words)
    ratio = matcher.ratio()
    return 1.0 - ratio

def compute_em(predicted, expected):
    """Exact Match"""
    return 1.0 if predicted == expected else 0.0

def compute_bleu(predicted, expected, n=4):
    """Simple BLEU approximation (unigram overlap)"""
    pred_tokens = predicted.split()
    exp_tokens = expected.split()
    
    if not exp_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    # Count matching tokens
    matches = sum(1 for t in pred_tokens if t in exp_tokens)
    return matches / len(exp_tokens)

def evaluate_samples(decoder, samples, device, batch_size=8):
    """
    Evaluate ByT5 + MLM reranker on samples.
    Returns: list of results with metrics
    """
    results = []
    total = len(samples)
    
    for idx, sample in enumerate(samples):
        singlish_input = sample['singlish']
        expected_output = sample['expected']
        
        # Print progress every 10 samples
        if idx % 10 == 0:
            print(f"  Progress: {idx}/{total}", flush=True)
        
        try:
            # Decode using BeamSearchDecoder (includes ByT5 + MLM reranking)
            predicted, trace_logs, _ = decoder.decode(singlish_input)
            
            # Compute metrics
            cer = compute_cer(predicted, expected_output)
            wer = compute_wer(predicted, expected_output)
            bleu = compute_bleu(predicted, expected_output)
            em = compute_em(predicted, expected_output)
            
            results.append({
                'singlish': singlish_input,
                'expected': expected_output,
                'predicted': predicted,
                'cer': cer,
                'wer': wer,
                'bleu': bleu,
                'em': em
            })
            
        except Exception as e:
            print(f"  Error at {idx}/{total} processing '{singlish_input}': {e}")
            results.append({
                'singlish': singlish_input,
                'expected': expected_output,
                'predicted': '[ERROR]',
                'cer': 1.0,
                'wer': 1.0,
                'bleu': 0.0,
                'em': 0
            })
    
    print(f"  Completed: {total}/{total}", flush=True)
    return results

def print_metrics(results, subset_name):
    """Print metrics summary"""
    if not results:
        print(f"{subset_name}: No results")
        return
    
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print(f"Subset: {subset_name} (n={len(results)})")
    print(f"{'='*60}")
    print(f"CER (lower is better):  {df['cer'].mean():.4f} ± {df['cer'].std():.4f}")
    print(f"WER (lower is better):  {df['wer'].mean():.4f} ± {df['wer'].std():.4f}")
    print(f"BLEU (higher is better): {df['bleu'].mean():.4f} ± {df['bleu'].std():.4f}")
    print(f"EM (higher is better):   {df['em'].mean():.4f} ({int(df['em'].sum())} / {len(results)})")
    
    # Show sample failures
    failures = df[df['em'] == 0].head(3)
    if len(failures) > 0:
        print(f"\nSample Failures (first 3):")
        for idx, row in failures.iterrows():
            print(f"  Input:    {row['singlish']}")
            print(f"  Expected: {row['expected']}")
            print(f"  Got:      {row['predicted']}")
            print()

def main():
    import sys
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse command line args for sample limits
    max_formal = int(sys.argv[1]) if len(sys.argv) > 1 else None
    max_informal = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Initialize model
    print("Loading BeamSearchDecoder (ByT5 + MLM reranker)...")
    decoder = BeamSearchDecoder(device=device)
    
    # Load test sets
    test_dir = Path("IndoNLP-2025-Shared-Task/Test Dataset/Sinhala")
    
    print("\nLoading Test Set 1 (formal, 10K)...")
    formal_samples = load_test_set(test_dir / "Sinhala Test set 1.txt", max_samples=max_formal)
    print(f"Loaded {len(formal_samples)} formal samples")
    
    print("Loading Test Set 2 (informal, 5K)...")
    informal_samples = load_test_set(test_dir / "Sinhala Test set 2.txt", max_samples=max_informal)
    print(f"Loaded {len(informal_samples)} informal samples")
    
    # Evaluate
    print("\n" + "="*60)
    print(f"EVALUATING FORMAL SUBSET ({len(formal_samples)} samples)")
    print("="*60)
    formal_results = evaluate_samples(decoder, formal_samples, device)
    
    print("\n" + "="*60)
    print(f"EVALUATING INFORMAL SUBSET ({len(informal_samples)} samples)")
    print("="*60)
    informal_results = evaluate_samples(decoder, informal_samples, device)
    
    # Print results
    print_metrics(formal_results, f"Formal ({len(formal_results)})")
    print_metrics(informal_results, f"Informal ({len(informal_results)})")
    
    # Overall
    all_results = formal_results + informal_results
    print_metrics(all_results, f"OVERALL ({len(all_results)} samples)")
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("misc/indo_nlp_eval_results.csv", index=False)
    print(f"\nDetailed results saved to: misc/indo_nlp_eval_results.csv")
    
    # Save summary
    summary = {
        'Subset': [f'Formal ({len(formal_results)})', f'Informal ({len(informal_results)})', f'Overall ({len(all_results)})'],
        'CER': [
            f"{pd.DataFrame(formal_results)['cer'].mean():.4f}",
            f"{pd.DataFrame(informal_results)['cer'].mean():.4f}",
            f"{results_df['cer'].mean():.4f}"
        ],
        'WER': [
            f"{pd.DataFrame(formal_results)['wer'].mean():.4f}",
            f"{pd.DataFrame(informal_results)['wer'].mean():.4f}",
            f"{results_df['wer'].mean():.4f}"
        ],
        'BLEU': [
            f"{pd.DataFrame(formal_results)['bleu'].mean():.4f}",
            f"{pd.DataFrame(informal_results)['bleu'].mean():.4f}",
            f"{results_df['bleu'].mean():.4f}"
        ],
        'EM': [
            f"{pd.DataFrame(formal_results)['em'].mean():.4f}",
            f"{pd.DataFrame(informal_results)['em'].mean():.4f}",
            f"{results_df['em'].mean():.4f}"
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("misc/indo_nlp_eval_summary.csv", index=False)
    print(f"Summary saved to: misc/indo_nlp_eval_summary.csv")

if __name__ == "__main__":
    main()
