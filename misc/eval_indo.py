#!/usr/bin/env python3
"""Evaluate ByT5 on Indo NLP test sets - file-based logging version."""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import pandas as pd
import json
from datetime import datetime

from core.decoder import BeamSearchDecoder

# Redirect stderr to avoid tqdm issues
import io
sys.stderr = open(os.devnull, 'w')

LOG_FILE = Path("misc/eval_progress.log")

def log(msg):
    """Log to file and stdout."""
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")
    print(msg, flush=True)

def load_test_set(filepath, max_samples=None):
    """Load Indo NLP test set."""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            samples.append({'singlish': lines[i], 'expected': lines[i+1]})
            if max_samples and len(samples) >= max_samples:
                break
    return samples

def compute_metrics(predicted, expected):
    """Compute CER, WER, BLEU, EM."""
    from difflib import SequenceMatcher
    
    matcher_char = SequenceMatcher(None, predicted, expected)
    cer = 1.0 - matcher_char.ratio() if expected else (1.0 if predicted else 0.0)
    
    pred_words = predicted.split()
    exp_words = expected.split()
    matcher_word = SequenceMatcher(None, pred_words, exp_words)
    wer = 1.0 - matcher_word.ratio() if exp_words else (1.0 if pred_words else 0.0)
    
    if exp_words:
        matches = sum(1 for t in pred_words if t in exp_words)
        bleu = matches / len(exp_words)
    else:
        bleu = 1.0 if not pred_words else 0.0
    
    em = 1 if predicted == expected else 0
    
    return {'cer': cer, 'wer': wer, 'bleu': bleu, 'em': em}

def main():
    # Clear log
    LOG_FILE.write_text("")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    max_formal = int(sys.argv[1]) if len(sys.argv) > 1 else None
    max_informal = int(sys.argv[2]) if len(sys.argv) > 2 else None
    log(f"Max formal: {max_formal}, Max informal: {max_informal}")
    
    log("\nLoading decoder...")
    try:
        decoder = BeamSearchDecoder(device=device)
        log("Decoder loaded!")
    except Exception as e:
        log(f"ERROR loading decoder: {e}")
        return
    
    # Load test sets
    test_dir = Path("IndoNLP-2025-Shared-Task/Test Dataset/Sinhala")
    
    log(f"\nLoading test sets...")
    formal_samples = load_test_set(test_dir / "Sinhala Test set 1.txt", max_samples=max_formal)
    informal_samples = load_test_set(test_dir / "Sinhala Test set 2.txt", max_samples=max_informal)
    log(f"Formal: {len(formal_samples)}, Informal: {len(informal_samples)}")
    
    all_results = []
    
    # Evaluate formal
    log(f"\n>>> EVALUATING FORMAL ({len(formal_samples)} samples)...")
    for idx, sample in enumerate(formal_samples):
        try:
            predicted, _, _ = decoder.decode(sample['singlish'])
            metrics = compute_metrics(predicted, sample['expected'])
            result = {**sample, 'predicted': predicted, 'subset': 'formal', **metrics}
            all_results.append(result)
            
            if (idx+1) % 10 == 0:
                log(f"  Formal {idx+1}/{len(formal_samples)}: EM={metrics['em']} CER={metrics['cer']:.3f}")
        except Exception as e:
            log(f"  ERROR at formal {idx+1}: {str(e)[:100]}")
            result = {**sample, 'predicted': '[ERROR]', 'subset': 'formal', 'cer': 1.0, 'wer': 1.0, 'bleu': 0.0, 'em': 0}
            all_results.append(result)
    
    log(f"Formal complete: {len([r for r in all_results if r['subset']=='formal'])} results")
    
    # Evaluate informal
    log(f"\n>>> EVALUATING INFORMAL ({len(informal_samples)} samples)...")
    formal_count = len(all_results)
    for idx, sample in enumerate(informal_samples):
        try:
            predicted, _, _ = decoder.decode(sample['singlish'])
            metrics = compute_metrics(predicted, sample['expected'])
            result = {**sample, 'predicted': predicted, 'subset': 'informal', **metrics}
            all_results.append(result)
            
            if (idx+1) % 10 == 0:
                log(f"  Informal {idx+1}/{len(informal_samples)}: EM={metrics['em']} CER={metrics['cer']:.3f}")
        except Exception as e:
            log(f"  ERROR at informal {idx+1}: {str(e)[:100]}")
            result = {**sample, 'predicted': '[ERROR]', 'subset': 'informal', 'cer': 1.0, 'wer': 1.0, 'bleu': 0.0, 'em': 0}
            all_results.append(result)
    
    log(f"Informal complete: {len([r for r in all_results if r['subset']=='informal'])} results")
    
    # Summary
    log(f"\n>>> SUMMARY...")
    all_df = pd.DataFrame(all_results)
    
    for subset in ['formal', 'informal', None]:
        if subset:
            df = all_df[all_df['subset'] == subset]
            label = subset.upper()
        else:
            df = all_df
            label = f"OVERALL ({len(df)})"
        
        cer_mean = df['cer'].mean()
        wer_mean = df['wer'].mean()
        bleu_mean = df['bleu'].mean()
        em_sum = int(df['em'].sum())
        
        log(f"{label:20s} n={len(df):5d} | CER={cer_mean:.4f} WER={wer_mean:.4f} BLEU={bleu_mean:.4f} EM={em_sum}/{len(df)}")
    
    # Save
    all_df.to_csv("misc/indo_nlp_results.csv", index=False)
    log(f"\nResults saved: misc/indo_nlp_results.csv")
    
    log("\nDONE!")

if __name__ == "__main__":
    main()
