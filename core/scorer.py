"""
Data-driven candidate scorer combining MLM, fidelity, and rank signals.
"""

import math
from dataclasses import dataclass, field
from typing import List

from core.constants import (
    W_MLM, W_FIDELITY, W_RANK,
    FIDELITY_SCALE, DICT_FIDELITY_DAMP,
    SINHALA_VIRAMA, ZWJ,
)


@dataclass
class ScoredCandidate:
    """Holds a candidate word and its scoring breakdown."""
    text: str
    mlm_score: float = 0.0
    fidelity_score: float = 0.0
    rank_score: float = 0.0
    combined_score: float = 0.0
    is_english: bool = False


@dataclass
class WordDiagnostic:
    """Structured per-word diagnostics for evaluation and error analysis."""
    step_index: int
    input_word: str
    rule_output: str
    selected_candidate: str
    beam_score: float
    candidate_breakdown: List[ScoredCandidate]


class CandidateScorer:
    """
    Data-driven replacement for the old hardcoded penalty table.

    Combines three probabilistic signals to rank candidates:

    1. **MLM Score** (weight α = 0.55)
       Contextual fit from XLM-RoBERTa masked language model.

    2. **Source-Aware Fidelity** (weight β = 0.45)
       English candidates matching input → 0.0 (user intent).
       Dictionary candidates → damped Levenshtein to rule output.
       Rule-only outputs → penalised by virama/skeleton density.
       Other → full Levenshtein distance to rule output.

    3. **Rank Prior** (weight γ = 0.0, disabled)
       Dictionary rank prior is disabled because entries are unordered.
    """

    def __init__(
        self,
        w_mlm: float = W_MLM,
        w_fidelity: float = W_FIDELITY,
        w_rank: float = W_RANK,
        fidelity_scale: float = FIDELITY_SCALE,
    ):
        self.w_mlm = w_mlm
        self.w_fidelity = w_fidelity
        self.w_rank = w_rank
        self.fidelity_scale = fidelity_scale

    # ── Levenshtein distance (pure-Python, no dependencies) ──────────

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """Compute the Levenshtein edit distance between two strings."""
        if not s1:
            return len(s2)
        if not s2:
            return len(s1)

        m, n = len(s1), len(s2)
        prev_row = list(range(n + 1))

        for i in range(1, m + 1):
            curr_row = [i] + [0] * n
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[j] = min(
                    prev_row[j] + 1,       # deletion
                    curr_row[j - 1] + 1,    # insertion
                    prev_row[j - 1] + cost, # substitution
                )
            prev_row = curr_row

        return prev_row[n]

    # ── Scoring components ───────────────────────────────────────────

    def compute_fidelity(
        self, candidate: str, rule_output: str,
        original_input: str = "", is_from_dict: bool = False,
        is_ambiguous: bool = False,
    ) -> float:
        """
        Source-aware transliteration fidelity.

        - **English matching input** → 0.0  (user-intent preservation).
        - **Dict + matches rule output** → strong bonus (+2.0),
          reduced to +0.5 when *is_ambiguous* (many dict candidates
          with different meanings → let MLM context decide).
        - **Dict only** → decaying bonus (1.0 down to 0.0 with distance).
        - **Rule-only outputs not in dictionary** → penalised by
          consonant-skeleton density (high virama ratio = malformed).
        - **Other** → full Levenshtein distance to rule output.
        """
        # 1. English candidate matching the original input word
        if original_input and candidate.lower() == original_input.lower():
            return 0.0

        # 2. Dictionary-validated candidates
        if is_from_dict:
            if candidate == rule_output:
                return 0.5 if is_ambiguous else 2.0
            max_len = max(len(candidate), len(rule_output), 1)
            norm_dist = self.levenshtein(candidate, rule_output) / max_len
            return max(0.0, 1.0 - norm_dist * DICT_FIDELITY_DAMP)

        # 3. Rule-only output (not validated by dictionary)
        if candidate == rule_output:
            bare_virama = sum(
                1 for i, ch in enumerate(candidate)
                if ch == SINHALA_VIRAMA
                and (i + 1 >= len(candidate) or candidate[i + 1] != ZWJ)
            )
            density = bare_virama / max(len(candidate), 1)
            return -density * self.fidelity_scale * 2

        # 4. English word not matching input — uncertain
        if candidate.isascii():
            return -0.5

        # 5. Sinhala candidate not from dictionary — distance penalty
        max_len = max(len(candidate), len(rule_output), 1)
        norm_dist = self.levenshtein(candidate, rule_output) / max_len
        return -norm_dist * self.fidelity_scale

    @staticmethod
    def compute_rank_prior(rank: int, total: int) -> float:
        """Log-decay rank prior. First candidate → 0.0; later ones decay."""
        if total <= 1:
            return 0.0
        return math.log(1.0 / (rank + 1))

    # ── Combined score ───────────────────────────────────────────────

    def score(
        self,
        mlm_score: float,
        candidate: str,
        rule_output: str,
        rank: int,
        total_candidates: int,
        is_english: bool = False,
        original_input: str = "",
        is_from_dict: bool = False,
        is_ambiguous: bool = False,
    ) -> ScoredCandidate:
        """Return a :class:`ScoredCandidate` with full breakdown."""
        fidelity = self.compute_fidelity(
            candidate, rule_output, original_input, is_from_dict,
            is_ambiguous,
        )
        rank_prior = self.compute_rank_prior(rank, total_candidates)

        combined = (
            self.w_mlm * mlm_score
            + self.w_fidelity * fidelity
            + self.w_rank * rank_prior
        )

        return ScoredCandidate(
            text=candidate,
            mlm_score=mlm_score,
            fidelity_score=fidelity,
            rank_score=rank_prior,
            combined_score=combined,
            is_english=is_english,
        )
