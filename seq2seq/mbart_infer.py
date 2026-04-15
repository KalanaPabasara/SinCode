"""
mBart50-based Sentence Transliterator for SinCode v3.

Full-sentence Singlish → Sinhala transliteration.
Unlike the ByT5 word-by-word pipeline, mBart50 operates on the whole input
sentence and produces fully Sinhalized output — no English words are retained.

Use-case: "mn heta business ekak start karanawa"
       → "මන් හෙට ව්‍යාපාරයක් පටන් ගන්නවා"
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

from core.constants import DEFAULT_MBART_MODEL

logger = logging.getLogger(__name__)

# ── Fix-map (ZWJ / Virama composition) ───────────────────────────────────────

_FIX_MAP_PATH = Path(__file__).parent / "compose_fix_map.json"

_fix_map_cache: dict[str, str] | None = None


def _load_fix_map() -> dict[str, str]:
    global _fix_map_cache
    if _fix_map_cache is None:
        with open(_FIX_MAP_PATH, "r", encoding="utf-8") as f:
            _fix_map_cache = json.load(f)
    return _fix_map_cache


# ── Input cleaning ────────────────────────────────────────────────────────────

# Scripts that are not Sinhala, Latin, numbers, or symbols — filtered out
_UNSUPPORTED_SCRIPT = re.compile(
    r"[\u0B80-\u0BFF"   # Tamil
    r"\u0900-\u097F"    # Devanagari
    r"\u4E00-\u9FFF"    # CJK Unified Ideographs
    r"\u3040-\u309F"    # Hiragana
    r"\u30A0-\u30FF"    # Katakana
    r"\u0E00-\u0E7F"    # Thai
    r"\u0600-\u06FF"    # Arabic
    r"\u0590-\u05FF"    # Hebrew
    r"\uAC00-\uD7AF]"   # Hangul
)


def _clean(text: str) -> str | None:
    """Remove words in unsupported scripts; return None if nothing remains."""
    words = text.strip().split()
    filtered = [w for w in words if not _UNSUPPORTED_SCRIPT.search(w)]
    return " ".join(filtered) if filtered else None


def _apply_fixes(text: str) -> str:
    """Apply ZWJ/virama composition fixes to mBart50 output."""
    for pattern, replacement in _load_fix_map().items():
        text = re.sub(pattern, replacement, text)
    return text


# ── Transliterator ────────────────────────────────────────────────────────────

class SentenceTransliterator:
    """
    Full-sentence Singlish → Sinhala transliterator (mBart50).

    Loads from Hugging Face Hub on first instantiation.
    Thread-safe for inference (no mutable state after __init__).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MBART_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading mBart50 transliterator: %s", model_name)
        self.tokenizer = MBart50Tokenizer.from_pretrained(model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def transliterate(self, text: str) -> str:
        """
        Transliterate a Singlish sentence to fully-Sinhalized output.

        Args:
            text: Input Singlish sentence (Romanized Sinhala / English mix).

        Returns:
            Sinhala-script output. Returns original text if input is empty
            or consists entirely of unsupported-script characters.
        """
        cleaned = _clean(text)
        if not cleaned:
            return text

        self.tokenizer.src_lang = "si_LK"
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["si_LK"],
            )

        output = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
        return _apply_fixes(output)
