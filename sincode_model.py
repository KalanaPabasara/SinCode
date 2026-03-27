"""
SinCode: Context-Aware Singlish-to-Sinhala Transliteration Engine

Backward-compatible entry point — all logic lives in the ``core/`` package.
This module re-exports the public API so that existing imports
(``from sincode_model import BeamSearchDecoder``) continue to work.

Author: Kalana Chandrasekara (2026)
"""

# ── Re-exports (public API) ─────────────────────────────────────────────────

from core.decoder import BeamSearchDecoder                    # noqa: F401
from core.scorer import CandidateScorer, ScoredCandidate, WordDiagnostic  # noqa: F401
from core.dictionary import DictionaryAdapter                 # noqa: F401
from core.transliterate import rule_based_transliterate       # noqa: F401
from core.english import ENGLISH_VOCAB, CORE_ENGLISH_WORDS, load_english_vocab  # noqa: F401
from core.mappings import COMMON_WORDS, CONTEXT_WORDS_STANDALONE  # noqa: F401
from core.constants import (                                  # noqa: F401
    DEFAULT_MODEL_NAME, DEFAULT_DICTIONARY_PATH,
    W_MLM, W_FIDELITY, W_RANK,
    MAX_CANDIDATES, DEFAULT_BEAM_WIDTH,
    FIDELITY_SCALE, DICT_FIDELITY_DAMP, MIN_ENGLISH_LEN,
)
