#!/usr/bin/env python3
"""Quick evaluation of ByT5 on Indo NLP test sets - simplified version."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import pandas as pd

from core.decoder import BeamSearchDecoder

def load_test_set(filepath, max_samples=None):
    """Load Indo NLP test set (pairs of lines: Singlish, Sinhala)."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            samples.append({
                'singlish': lines[i],
                'expected': lines[i + 1]
            })
            if max_samples and len(samples) >= max_samples:
                break
    return samples

def compute_metrics(predicted, expected):
    """Compute CER, WER, BLEU, EM."""
    from difflib import SequenceMatcher
    
    # CER (Character Error Rate)
    matcher_char = SequenceMatcher(None, predicted, expected)
    cer = 1.0 - matcher_char.ratio() if expected else (1.0 if predicted else 0.0)
    
    # WER (Word Error Rate)
    pred_words = predicted.split()
    exp_words = expected.split()
    matcher_word = SequenceMatcher(None, pred_words, exp_words)
    wer = 1.0 - matcher_word.ratio() if exp_words else (1.0 if pred_words else 0.0)
    
    # BLEU (simple unigram overlap)
    if exp_words:
        matches = sum(1 for t in pred_words if t in exp_words)
        bleu = matches / len(exp_words)
    else:
        bleu = 1.0 if not pred_words else 0.0
    
    # EM (Exact Match)
    em = 1 if predicted == expected else 0
    
    return {'cer': cer, 'wer': wer, 'bleu': bleu, 'em': em}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Parse command line
    max_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    print(f"Loading BeamSearchDecoder...")
    decoder = BeamSearchDecoder(device=device)
    print(f"Decoder loaded!\n")
    
    # Load test sets
    test_dir = Path("IndoNLP-2025-Shared-Task/Test Dataset/Sinhala")
    
    print(f"Loading test sets (max {max_samples} samples each)...")
    formal_samples = load_test_set(test_dir / "Sinhala Test set 1.txt", max_samples=max_samples)
    informal_samples = load_test_set(test_dir / "Sinhala Test set 2.txt", max_samples=max_samples)
    print(f"Formal: {len(formal_samples)}, Informal: {len(informal_samples)}\n")
    
    all_results = []
    
    # Evaluate formal
    print("="*60)
    print(f"FORMAL SUBSET ({len(formal_samples)} samples)")
    print("="*60)
    
    formal_results = []
    for idx, sample in enumerate(formal_samples):
        try:
            predicted, _, _ = decoder.decode(sample['singlish'])
            metrics = compute_metrics(predicted, sample['expected'])
            result = {**sample, 'predicted': predicted, 'subset': 'formal', **metrics}
            formal_results.append(result)
            all_results.append(result)
            print(f"{idx+1}/{len(formal_samples)}: EM={metrics['em']} CER={metrics['cer']:.3f} WER={metrics['wer']:.3f}")
        except Exception as e:
            print(f"{idx+1}/{len(formal_samples)}: ERROR - {e}")
            result = {**sample, 'predicted': '[ERROR]', 'subset': 'formal', 'cer': 1.0, 'wer': 1.0, 'bleu': 0.0, 'em': 0}
            formal_results.append(result)
            all_results.append(result)
    
    # Evaluate informal
    print("\n" + "="*60)
    print(f"INFORMAL SUBSET ({len(informal_samples)} samples)")
    print("="*60)
    
    informal_results = []
    for idx, sample in enumerate(informal_samples):
        try:
            predicted, _, _ = decoder.decode(sample['singlish'])
            metrics = compute_metrics(predicted, sample['expected'])
            result = {**sample, 'predicted': predicted, 'subset': 'informal', **metrics}
            informal_results.append(result)
            all_results.append(result)
            print(f"{idx+1}/{len(informal_samples)}: EM={metrics['em']} CER={metrics['cer']:.3f} WER={metrics['wer']:.3f}")
        except Exception as e:
            print(f"{idx+1}/{len(informal_samples)}: ERROR - {e}")
            result = {**sample, 'predicted': '[ERROR]', 'subset': 'informal', 'cer': 1.0, 'wer': 1.0, 'bleu': 0.0, 'em': 0}
            informal_results.append(result)
            all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    formal_df = pd.DataFrame(formal_results)
    informal_df = pd.DataFrame(informal_results)
    all_df = pd.DataFrame(all_results)
    
    for name, df in [("Formal", formal_df), ("Informal", informal_df), ("Overall", all_df)]:
        print(f"\n{name} (n={len(df)}):")
        print(f"  CER:  {df['cer'].mean():.4f} ± {df['cer'].std():.4f}")
        print(f"  WER:  {df['wer'].mean():.4f} ± {df['wer'].std():.4f}")
        print(f"  BLEU: {df['bleu'].mean():.4f} ± {df['bleu'].std():.4f}")
        print(f"  EM:   {df['em'].mean():.4f} ({int(df['em'].sum())}/{len(df)})")
    
    # Save
    all_df.to_csv("misc/quick_eval_results.csv", index=False)
    print(f"\nResults saved to: misc/quick_eval_results.csv")

if __name__ == "__main__":
    main()
