"""
Configuration constants and hyperparameters for the SinCode engine.
"""

import re

# ─── Model & Data Paths ─────────────────────────────────────────────────────

# DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_MODEL_NAME = "Kalana001/xlm-roberta-sinhala-sincode"
DEFAULT_DICTIONARY_PATH = "dictionary.pkl"

ENGLISH_CORPUS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
)

# ─── Scoring Weights (tunable hyperparameters) ──────────────────────────────

W_MLM: float = 0.55           # Contextual language model probability
W_FIDELITY: float = 0.45      # Source-aware transliteration fidelity
W_RANK: float = 0.00          # Dictionary rank prior (disabled — dict is unordered)

# ─── Decoding Parameters ────────────────────────────────────────────────────

MAX_CANDIDATES: int = 8       # Max candidates per word position
DEFAULT_BEAM_WIDTH: int = 5   # Beam search width
FIDELITY_SCALE: float = 10.0  # Edit-distance penalty multiplier
DICT_FIDELITY_DAMP: float = 2.0  # Decay rate for dict bonus (higher = stricter filter)
MIN_ENGLISH_LEN: int = 3      # Min word length for 20k-corpus English detection

# ─── Unicode Constants ──────────────────────────────────────────────────────

SINHALA_VIRAMA: str = '\u0DCA'  # Sinhala virama (hal) character
ZWJ: str = '\u200D'             # Zero-width joiner (for conjuncts)

# ─── Regex ──────────────────────────────────────────────────────────────────

PUNCT_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$")
