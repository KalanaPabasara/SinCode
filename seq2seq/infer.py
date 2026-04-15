"""
Inference helper — given a romanized word, return top-K Sinhala candidates
using beam search on the fine-tuned ByT5 model.

Usage:
    from seq2seq.infer import Transliterator
    t = Transliterator()
    print(t.candidates("videowe", k=5))
    # ['වීඩියොවේ', 'වීඩියොවී', 'වීඩියොව', ...]
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
from transformers import ByT5Tokenizer, T5ForConditionalGeneration

DEFAULT_MODEL_PATH = Path(__file__).parent / "byt5-singlish-sinhala" / "final"


class Transliterator:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, device: Optional[str] = None):
        # Keep as string — Path() would convert '/' to '\' on Windows, breaking HF Hub IDs
        model_path = str(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = ByT5Tokenizer.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def candidates(self, word: str, k: int = 5) -> list[str]:
        """Return top-k Sinhala transliteration candidates for a single word."""
        return self.batch_candidates([word], k=k)[0]

    def batch_candidates(self, words: list[str], k: int = 5) -> list[list[str]]:
        """
        Return top-k Sinhala candidates for each word in a single forward pass.
        Much faster than calling candidates() per word on a long sentence.
        """
        lowered = [w.lower() for w in words]
        inputs = self.tokenizer(
            lowered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(self.device)

        n = len(words)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=max(k, 5),
                num_return_sequences=k,
                max_new_tokens=64,
                early_stopping=True,
            )

        # outputs shape: (n * k, seq_len) — k sequences per input, grouped
        results: list[list[str]] = []
        for i in range(n):
            seen: set[str] = set()
            cands: list[str] = []
            for seq in outputs[i * k : (i + 1) * k]:
                text = self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                if text and text not in seen:
                    seen.add(text)
                    cands.append(text)
            results.append(cands)

        return results


if __name__ == "__main__":
    import sys
    words = sys.argv[1:] if len(sys.argv) > 1 else ["wadi"]
    t = Transliterator()
    for word in words:
        print(f"Candidates for '{word}':")
        for c in t.candidates(word):
            print(f"  {c}")
        print()
