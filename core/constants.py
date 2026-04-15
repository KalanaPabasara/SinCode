"""
Configuration constants for SinCode v3.

Key difference from v2: no rule engine, no dictionary.
Candidate generation is fully handled by the ByT5 seq2seq model.
"""

import re

# ─── MLM Model Path ──────────────────────────────────────────────────────────
# XLM-RoBERTa fine-tuned on Sinhala — reranks ByT5 candidates by context
DEFAULT_MLM_MODEL = "Kalana001/xlm-roberta-base-finetuned-sinhala"

# ─── ByT5 Transliterator Model Path ──────────────────────────────────────────
# Fine-tuned on 1M Singlish→Sinhala pairs — hosted on Hugging Face Hub
DEFAULT_BYT5_MODEL = "Kalana001/byt5-small-singlish-sinhala"

# ─── mBart50 Transliterator Model Path ───────────────────────────────────────
# Full-sentence Singlish→Sinhala (no English retained) — Hugging Face Hub
DEFAULT_MBART_MODEL = "Kalana001/mbart50-large-singlish-sinhala"

# ─── Corpus ───────────────────────────────────────────────────────────────────
ENGLISH_CORPUS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
)

# ─── Scoring Weights ─────────────────────────────────────────────────────────
# Pure MLM — no manual weights needed

# ─── Decoding Parameters ─────────────────────────────────────────────────────
MAX_CANDIDATES: int = 5       # ByT5 beam=5 → 5 candidates per word
MIN_ENGLISH_LEN: int = 3      # Min word length for English detection

# ─── Regex ───────────────────────────────────────────────────────────────────
PUNCT_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$")
