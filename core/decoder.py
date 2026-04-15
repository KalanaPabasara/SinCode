"""
SinCode v3 — ByT5 Seq2Seq + XLM-RoBERTa MLM Reranker.

Pipeline (per word):
  Sinhala script  → MLM scores in context (single candidate)
  English vocab   → ByT5 generates Sinhala alternatives + English kept; MLM picks
  Everything else → ByT5 generates top-5 candidates; MLM picks best
"""

import math
import re
import torch
import logging
from typing import List, Tuple, Optional

from transformers import AutoTokenizer, AutoModelForMaskedLM

from core.constants import (
    DEFAULT_MLM_MODEL, DEFAULT_BYT5_MODEL,
    MAX_CANDIDATES, MIN_ENGLISH_LEN,
    PUNCT_PATTERN,
)
from core.english import ENGLISH_VOCAB
from seq2seq.infer import Transliterator

logger = logging.getLogger(__name__)

_SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")


class ScoredCandidate:
    __slots__ = ("text", "mlm_score")

    def __init__(self, text: str, mlm_score: float):
        self.text = text
        self.mlm_score = mlm_score


def _is_sinhala(text: str) -> bool:
    return bool(_SINHALA_RE.search(text))


class BeamSearchDecoder:
    """
    SinCode v3 contextual decoder.

    Replaces the rule engine + dictionary + hardcoded maps with a single
    ByT5-small seq2seq model fine-tuned on 1,000,000 Singlish→Sinhala pairs.
    XLM-RoBERTa reranks the top-5 beam candidates by masked-LM probability.
    """

    def __init__(
        self,
        mlm_model_name: str = DEFAULT_MLM_MODEL,
        byt5_model_path: str = DEFAULT_BYT5_MODEL,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading MLM reranker: %s", mlm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(mlm_model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(mlm_model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loading ByT5 transliterator: %s", byt5_model_path)
        self.transliterator = Transliterator(model_path=byt5_model_path, device=self.device)

    # ── Normalization ─────────────────────────────────────────────────────────

    @staticmethod
    def _softmax_normalize(raw_scores: List[float]) -> List[float]:
        if not raw_scores:
            return []
        if len(raw_scores) == 1:
            return [1.0]
        max_s = max(raw_scores)
        exps = [math.exp(s - max_s) for s in raw_scores]
        total = sum(exps)
        return [e / total for e in exps]

    # ── MLM batch scoring ─────────────────────────────────────────────────────

    def _batch_mlm_score(
        self,
        left_contexts: List[str],
        right_contexts: List[str],
        candidates: List[str],
    ) -> List[float]:
        """Score each candidate with XLM-RoBERTa multi-mask log-probability."""
        if not candidates:
            return []

        mask = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        cand_token_ids: List[List[int]] = []
        for c in candidates:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            cand_token_ids.append(ids if ids else [self.tokenizer.unk_token_id])

        batch_texts: List[str] = []
        for i in range(len(candidates)):
            n_masks = len(cand_token_ids[i])
            mask_str = " ".join([mask] * n_masks)
            parts = [p for p in [left_contexts[i], mask_str, right_contexts[i]] if p]
            batch_texts.append(" ".join(parts))

        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits

        scores: List[float] = []
        for i, target_ids in enumerate(cand_token_ids):
            token_ids = inputs.input_ids[i]
            mask_positions = (token_ids == mask_token_id).nonzero(as_tuple=True)[0]

            if mask_positions.numel() == 0 or not target_ids:
                scores.append(-100.0)
                continue

            n = min(len(target_ids), mask_positions.numel())
            total = 0.0
            for j in range(n):
                pos = mask_positions[j].item()
                log_probs = torch.log_softmax(logits[i, pos, :], dim=0)
                total += log_probs[target_ids[j]].item()

            scores.append(total / n)

        return scores

    # ── Public decode ─────────────────────────────────────────────────────────

    def decode(self, sentence: str) -> Tuple[str, List[str]]:
        """
        Decode a Singlish sentence word-by-word using ByT5 + XLM-RoBERTa MLM.
        Returns (transliterated_sentence, trace_logs).
        """
        words = sentence.split()
        if not words:
            return "", []

        # ── Phase 1: batch ByT5 candidate generation ──────────────────────────
        # Collect only the words that need ByT5 (non-Sinhala), run in one pass
        cores: List[str] = []
        core_meta: List[tuple] = []  # (index_into_words, prefix, core, suffix, core_lower)

        for i, raw in enumerate(words):
            match = PUNCT_PATTERN.match(raw)
            prefix, core, suffix = match.groups() if match else ("", raw, "")
            if not _is_sinhala(core):
                cores.append(core)
                core_meta.append((i, prefix, core, suffix, core.lower()))

        # Single ByT5 forward pass for all non-Sinhala words
        byt5_results: List[List[str]] = (
            self.transliterator.batch_candidates(cores, k=MAX_CANDIDATES)
            if cores else []
        )

        byt5_map: dict = {}  # word index → list of raw ByT5 strings
        for (i, prefix, core, suffix, core_lower), cands in zip(core_meta, byt5_results):
            byt5_map[i] = (prefix, suffix, core_lower, cands or [core])

        word_infos: List[dict] = []
        for i, raw in enumerate(words):
            match = PUNCT_PATTERN.match(raw)
            _, core, _ = match.groups() if match else ("", raw, "")

            if _is_sinhala(core):
                word_infos.append({"kind": "sinhala", "candidates": [raw]})
                continue

            prefix, suffix, core_lower, byt5_cands = byt5_map[i]
            sinhala_cands = [prefix + c + suffix for c in byt5_cands]

            if core_lower in ENGLISH_VOCAB and len(core_lower) >= MIN_ENGLISH_LEN:
                candidates = [raw] + [c for c in sinhala_cands if c != raw]
                word_infos.append({"kind": "english", "candidates": candidates[:MAX_CANDIDATES + 1]})
            else:
                word_infos.append({"kind": "singlish", "candidates": sinhala_cands})

        # ── Phase 2: greedy left-to-right pass (builds dynamic left context) ──
        # Right context is seeded from first ByT5 candidate (pre-decode estimate)
        stable_right = [info["candidates"][0] for info in word_infos]
        selected_words: List[str] = []

        for t, info in enumerate(word_infos):
            candidates = info["candidates"]
            left_ctx = " ".join(selected_words)
            right_ctx = " ".join(stable_right[t + 1:])
            raw_mlm = self._batch_mlm_score(
                [left_ctx] * len(candidates),
                [right_ctx] * len(candidates),
                candidates,
            )
            norm_mlm = self._softmax_normalize(raw_mlm)
            best = max(zip(candidates, norm_mlm), key=lambda x: x[1])
            selected_words.append(best[0])

        # ── Phase 3: re-score with full decoded sentence as context ───────────
        # Right context is now the actual decoded output, not the pre-decode estimate
        trace_logs: List[str] = []
        final_words: List[str] = []

        for t, info in enumerate(word_infos):
            raw_word = words[t]
            kind = info["kind"]
            candidates = info["candidates"]

            left_ctx = " ".join(final_words)
            right_ctx = " ".join(selected_words[t + 1:])

            raw_mlm = self._batch_mlm_score(
                [left_ctx] * len(candidates),
                [right_ctx] * len(candidates),
                candidates,
            )
            norm_mlm = self._softmax_normalize(raw_mlm)

            scored = sorted(
                [ScoredCandidate(text=c, mlm_score=norm_mlm[i]) for i, c in enumerate(candidates)],
                key=lambda x: x.mlm_score,
                reverse=True,
            )
            best = scored[0]
            final_words.append(best.text)

            if kind == "sinhala":
                trace_logs.append(
                    f"**Step {t+1}: `{raw_word}`** → `{best.text}` "
                    f"(Sinhala, MLM={best.mlm_score:.3f})\n"
                )
            else:
                trace_logs.append(
                    f"**Step {t+1}: `{raw_word}`** → `{best.text}` "
                    f"(MLM={best.mlm_score:.3f})\n"
                    + "\n".join(f"  - `{s.text}` {s.mlm_score:.3f}" for s in scored)
                )

        return " ".join(final_words), trace_logs
