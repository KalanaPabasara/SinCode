"""
Beam search and greedy decoders for Singlish → Sinhala transliteration.
"""

import math
import re
import torch
import pickle
import logging
from typing import List, Tuple, Dict, Optional, Set

from transformers import AutoTokenizer, AutoModelForMaskedLM

from core.constants import (
    DEFAULT_MODEL_NAME, DEFAULT_DICTIONARY_PATH,
    DEFAULT_BEAM_WIDTH, MAX_CANDIDATES, MIN_ENGLISH_LEN,
    PUNCT_PATTERN,
)
from core.mappings import COMMON_WORDS, CONTEXT_WORDS_STANDALONE
from core.english import ENGLISH_VOCAB
from core.scorer import CandidateScorer, ScoredCandidate, WordDiagnostic
from core.dictionary import DictionaryAdapter

logger = logging.getLogger(__name__)

# Sinhala Unicode block: U+0D80 – U+0DFF
_SINHALA_RE = re.compile(r"[\u0D80-\u0DFF]")


def _is_sinhala(text: str) -> bool:
    """Return True if the text already contains Sinhala script characters."""
    return bool(_SINHALA_RE.search(text))


class BeamSearchDecoder:
    """
    Contextual beam-search decoder for Singlish → Sinhala transliteration.

    For each word position the decoder:
        1. Generates candidates (dictionary + rule engine)
        2. Scores them with XLM-R MLM in sentence context
        3. Combines MLM score with fidelity & rank via CandidateScorer
        4. Prunes to the top-k (beam width) hypotheses
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        dictionary_path: str = DEFAULT_DICTIONARY_PATH,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading tokenizer & model: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Loading dictionary: %s", dictionary_path)
        with open(dictionary_path, "rb") as f:
            d_data = pickle.load(f)
        self.adapter = DictionaryAdapter(d_data)
        self.scorer = CandidateScorer()

    # ── Normalization ─────────────────────────────────────────────

    @staticmethod
    def _softmax_normalize(raw_scores: List[float]) -> List[float]:
        """
        Normalize raw log-probability scores to [0, 1] via softmax.

        Unlike min-max (which maps best→1.0, worst→0.0 regardless of
        the actual difference), softmax preserves the model's relative
        confidence.  When all candidates have similar log-probs the
        output values cluster together; when the model is very
        confident they spread apart.

        The raw scores are already log-probs (negative), so we use
        them directly as logits for softmax.
        """
        if not raw_scores:
            return []
        if len(raw_scores) == 1:
            return [1.0]

        # Subtract max for numerical stability (standard log-sum-exp trick)
        max_s = max(raw_scores)
        exps = [math.exp(s - max_s) for s in raw_scores]
        total = sum(exps)
        return [e / total for e in exps]

    # ── MLM batch scoring ────────────────────────────────────────────

    def _batch_mlm_score(
        self,
        left_contexts: List[str],
        right_contexts: List[str],
        candidates: List[str],
    ) -> List[float]:
        """
        Score each candidate using masked LM log-probability with proper
        multi-mask scoring for multi-subword candidates.

        Instead of placing a single <mask> and summing subword log-probs
        at that one position, this method creates one <mask> per subword
        token and scores each subword at its own position:

            score = (1/N) * Σ_i  log P(t_i | mask_position_i, context)
        """
        if not candidates:
            return []

        mask = self.tokenizer.mask_token
        mask_token_id = self.tokenizer.mask_token_id

        # Pre-tokenize every candidate to determine subword count
        cand_token_ids: List[List[int]] = []
        for c in candidates:
            ids = self.tokenizer.encode(c, add_special_tokens=False)
            cand_token_ids.append(ids if ids else [self.tokenizer.unk_token_id])

        # Build context strings with the correct number of <mask> tokens
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

            # Score each subword at its corresponding mask position
            n = min(len(target_ids), mask_positions.numel())
            total = 0.0
            for j in range(n):
                pos = mask_positions[j].item()
                log_probs = torch.log_softmax(logits[i, pos, :], dim=0)
                total += log_probs[target_ids[j]].item()

            scores.append(total / n)

        return scores

    # ── Main decode entry-point ──────────────────────────────────────

    def decode(
        self,
        sentence: str,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        mode: str = "greedy",
    ) -> Tuple[str, List[str]]:
        """
        Transliterate a full Singlish sentence into Sinhala script.

        Args:
            mode: "greedy" (accurate, uses dynamic context) or
                  "beam" (uses fixed rule-based context)

        Returns:
            result      – the best transliteration string
            trace_logs  – per-step markdown logs for the debug UI
        """
        if mode == "greedy":
            result, trace_logs, _ = self.greedy_decode_with_diagnostics(sentence)
        else:
            result, trace_logs, _ = self.decode_with_diagnostics(
                sentence=sentence,
                beam_width=beam_width,
            )
        return result, trace_logs

    # ── Greedy decode (dynamic context — more accurate) ──────────────

    def greedy_decode_with_diagnostics(
        self,
        sentence: str,
    ) -> Tuple[str, List[str], List[WordDiagnostic]]:
        """
        Greedy word-by-word decode using actual selected outputs as
        left context for subsequent MLM scoring.

        More accurate than beam search with fixed context because XLM-R
        sees the real transliteration built so far, not rule-based guesses.
        """
        words = sentence.split()
        if not words:
            return "", [], []

        # ── Phase 1: candidate generation (same as beam) ─────────────
        word_infos: List[dict] = []

        for raw in words:
            match = PUNCT_PATTERN.match(raw)
            prefix, core, suffix = match.groups() if match else ("", raw, "")

            if not core:
                word_infos.append({
                    "candidates": [raw],
                    "rule_output": raw,
                    "english_flags": [False],
                    "dict_flags": [False],
                    "prefix": prefix,
                    "suffix": suffix,
                    "sinhala_passthrough": False,
                })
                continue

            # Already-Sinhala text: pass through unchanged
            if _is_sinhala(core):
                word_infos.append({
                    "candidates": [raw],
                    "rule_output": raw,
                    "english_flags": [False],
                    "dict_flags": [False],
                    "prefix": prefix,
                    "suffix": suffix,
                    "sinhala_passthrough": True,
                })
                continue

            rule_output = self.adapter.get_rule_output(core)
            cands = self.adapter.get_candidates(core, rule_output)

            dict_entries: Set[str] = set()
            if core in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core])
            elif core.lower() in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core.lower()])

            if rule_output and rule_output not in cands:
                cands.append(rule_output)
            if not cands:
                cands = [rule_output]

            english_flags = [c.lower() in ENGLISH_VOCAB for c in cands]
            dict_flags = [c in dict_entries for c in cands]

            full_cands = [prefix + c + suffix for c in cands]

            word_infos.append({
                "candidates": full_cands[:MAX_CANDIDATES],
                "rule_output": prefix + rule_output + suffix,
                "core_rule_output": rule_output,
                "n_dict_entries": len(dict_entries),
                "dict_entries": dict_entries,
                "english_flags": english_flags[:MAX_CANDIDATES],
                "dict_flags": dict_flags[:MAX_CANDIDATES],
                "prefix": prefix,
                "suffix": suffix,
                "sinhala_passthrough": False,
            })

        # Build right-side stable context (rule outputs for future words)
        stable_right: List[str] = []
        for info in word_infos:
            eng_cands = [
                c for c, e in zip(info["candidates"], info["english_flags"]) if e
            ]
            stable_right.append(
                eng_cands[0] if eng_cands else info["rule_output"]
            )

        # ── Phase 2: greedy word-by-word with dynamic left context ───
        selected_words: List[str] = []
        trace_logs: List[str] = []
        diagnostics: List[WordDiagnostic] = []

        for t, info in enumerate(word_infos):
            candidates = info["candidates"]
            eng_flags = info["english_flags"]
            d_flags = info.get("dict_flags", [False] * len(candidates))
            rule_out = info["rule_output"]
            prefix = info.get("prefix", "")
            suffix = info.get("suffix", "")
            total_cands = len(candidates)

            # ── Sinhala passthrough ────────────────────────────────────
            if info.get("sinhala_passthrough"):
                selected_words.append(words[t])
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{words[t]}` (Sinhala passthrough)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=words[t],
                    beam_score=0.0,
                    candidate_breakdown=[],
                ))
                continue

            # ── Common-word shortcut ─────────────────────────────────
            core_lower = words[t].lower().strip()
            if core_lower in COMMON_WORDS:
                override = prefix + COMMON_WORDS[core_lower] + suffix
                selected_words.append(override)
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{override}` (common-word override)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=override,
                    beam_score=0.0,
                    candidate_breakdown=[],
                ))
                continue

            # ── Context-dependent standalone overrides ────────────────
            if core_lower in CONTEXT_WORDS_STANDALONE:
                prev_word_lower = words[t - 1].lower() if t > 0 else ""
                prev_common_val = COMMON_WORDS.get(prev_word_lower, "")
                prev_is_english = (
                    t > 0
                    and (
                        prev_word_lower in ENGLISH_VOCAB
                        or prev_common_val.isascii() and prev_common_val != ""
                    )
                )
                if not prev_is_english:
                    override = prefix + CONTEXT_WORDS_STANDALONE[core_lower] + suffix
                    selected_words.append(override)
                    trace_logs.append(
                        f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                        f"`{override}` (standalone override)\n"
                    )
                    diagnostics.append(WordDiagnostic(
                        step_index=t,
                        input_word=words[t],
                        rule_output=rule_out,
                        selected_candidate=override,
                        beam_score=0.0,
                        candidate_breakdown=[],
                    ))
                    continue

            # ── English-word shortcut ────────────────────────────────
            # Preserve English immediately UNLESS the romanisation maps
            # to a genuine Sinhala word (rule output found in the
            # dictionary with 3+ entries → multiple meanings).
            # e.g. "game" rule→ගමෙ exists in dict with 7 entries → ambiguous.
            # e.g. "meeting" rule→මීටින්ග් is in dict but only 1 entry →
            #      loanword transliteration, keep English.
            core_rule = info.get("core_rule_output", "")
            core_dict = info.get("dict_entries", set())
            is_semantically_ambiguous = (
                core_rule in core_dict and len(core_dict) >= 3
            )
            if (
                len(core_lower) >= MIN_ENGLISH_LEN
                and core_lower in ENGLISH_VOCAB
                and not is_semantically_ambiguous
            ):
                selected_words.append(words[t])
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{words[t]}` (English preserved)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=words[t],
                    beam_score=0.0,
                    candidate_breakdown=[],
                ))
                continue

            # Dynamic left context = actual selected outputs so far
            left_ctx = " ".join(selected_words) if selected_words else ""
            # Right context = rule-based stable context for future words
            right_ctx = " ".join(stable_right[t + 1:]) if t + 1 < len(words) else ""

            # Score all candidates for this position in one batch
            batch_left = [left_ctx] * total_cands
            batch_right = [right_ctx] * total_cands

            mlm_scores = self._batch_mlm_score(batch_left, batch_right, candidates)

            # ── Softmax normalise MLM scores ─────────────────────────
            # Preserves the model's relative confidence — close raw
            # log-probs yield close normalised values, unlike min-max
            # which always maps best→1.0 / worst→0.0.
            mlm_scores = self._softmax_normalize(mlm_scores)

            # MLM floor for English code-switching
            # Skip floor for semantically ambiguous words (rule output
            # found in dict with 3+ entries) so raw MLM context signal
            # can distinguish e.g. "game" (English) vs ගමේ (village).
            best_nonenglish_mlm = -1e9
            if not is_semantically_ambiguous:
                for i, mlm in enumerate(mlm_scores):
                    is_eng = eng_flags[i] if i < len(eng_flags) else False
                    if not is_eng and mlm > best_nonenglish_mlm:
                        best_nonenglish_mlm = mlm

            # Score & select best candidate
            step_log = f"**Step {t + 1}: `{words[t]}`** &nbsp;(rule → `{rule_out}`)\n\n"
            best_scored: Optional[ScoredCandidate] = None
            candidate_breakdown: List[ScoredCandidate] = []

            for i, mlm in enumerate(mlm_scores):
                cand = candidates[i]
                is_eng = eng_flags[i] if i < len(eng_flags) else False
                is_dict = d_flags[i] if i < len(d_flags) else False

                effective_mlm = mlm
                if is_eng and cand.lower() == words[t].lower() and not is_semantically_ambiguous:
                    effective_mlm = max(mlm, best_nonenglish_mlm)

                scored = self.scorer.score(
                    mlm_score=effective_mlm,
                    candidate=cand,
                    rule_output=rule_out,
                    rank=i,
                    total_candidates=total_cands,
                    is_english=is_eng,
                    original_input=words[t],
                    is_from_dict=is_dict,
                    is_ambiguous=is_semantically_ambiguous,
                )
                candidate_breakdown.append(scored)

                if best_scored is None or scored.combined_score > best_scored.combined_score:
                    best_scored = scored

                if mlm > -25.0:
                    eng_tag = " 🔤" if is_eng else ""
                    step_log += (
                        f"- `{cand}`{eng_tag} &nbsp; "
                        f"MLM={scored.mlm_score:.2f} &nbsp; "
                        f"Fid={scored.fidelity_score:.2f} &nbsp; "
                        f"Rank={scored.rank_score:.2f} → "
                        f"**{scored.combined_score:.2f}**\n"
                    )

            trace_logs.append(step_log)

            selected = best_scored.text if best_scored else rule_out
            selected_words.append(selected)

            candidate_breakdown.sort(key=lambda s: s.combined_score, reverse=True)
            diagnostics.append(WordDiagnostic(
                step_index=t,
                input_word=words[t],
                rule_output=rule_out,
                selected_candidate=selected,
                beam_score=best_scored.combined_score if best_scored else 0.0,
                candidate_breakdown=candidate_breakdown,
            ))

        result = " ".join(selected_words)
        return result, trace_logs, diagnostics

    # ── Beam decode (fixed context — legacy comparison) ──────────────

    def decode_with_diagnostics(
        self,
        sentence: str,
        beam_width: int = DEFAULT_BEAM_WIDTH,
    ) -> Tuple[str, List[str], List[WordDiagnostic]]:
        """
        Decode sentence using beam search and return detailed diagnostics.

        Uses fixed rule-based context for all beam paths. Kept for
        comparison with greedy decode in evaluation.
        """
        words = sentence.split()
        if not words:
            return "", [], []

        # ── Phase 1: candidate generation ────────────────────────────
        word_infos: List[dict] = []

        for raw in words:
            match = PUNCT_PATTERN.match(raw)
            prefix, core, suffix = match.groups() if match else ("", raw, "")

            if not core:
                word_infos.append({
                    "candidates": [raw],
                    "rule_output": raw,
                    "english_flags": [False],
                    "prefix": prefix,
                    "suffix": suffix,
                    "sinhala_passthrough": False,
                })
                continue

            # Already-Sinhala text: pass through unchanged
            if _is_sinhala(core):
                word_infos.append({
                    "candidates": [raw],
                    "rule_output": raw,
                    "english_flags": [False],
                    "prefix": prefix,
                    "suffix": suffix,
                    "sinhala_passthrough": True,
                })
                continue

            rule_output = self.adapter.get_rule_output(core)
            cands = self.adapter.get_candidates(core, rule_output)

            dict_entries: Set[str] = set()
            if core in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core])
            elif core.lower() in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core.lower()])

            if rule_output and rule_output not in cands:
                cands.append(rule_output)
            if not cands:
                cands = [rule_output]

            english_flags = [c.lower() in ENGLISH_VOCAB for c in cands]
            dict_flags = [c in dict_entries for c in cands]
            full_cands = [prefix + c + suffix for c in cands]

            word_infos.append({
                "candidates": full_cands[:MAX_CANDIDATES],
                "rule_output": prefix + rule_output + suffix,
                "core_rule_output": rule_output,
                "n_dict_entries": len(dict_entries),
                "dict_entries": dict_entries,
                "english_flags": english_flags[:MAX_CANDIDATES],
                "dict_flags": dict_flags[:MAX_CANDIDATES],
                "prefix": prefix,
                "suffix": suffix,
                "sinhala_passthrough": False,
            })

        # Build stable context (fixed for all beam paths)
        stable_context: List[str] = []
        for info in word_infos:
            eng_cands = [
                c for c, e in zip(info["candidates"], info["english_flags"]) if e
            ]
            stable_context.append(
                eng_cands[0] if eng_cands else info["rule_output"]
            )

        # ── Phase 2: beam search with data-driven scoring ────────────
        beam: List[Tuple[List[str], float]] = [([], 0.0)]
        trace_logs: List[str] = []
        diagnostics: List[WordDiagnostic] = []

        for t, info in enumerate(word_infos):
            candidates = info["candidates"]
            eng_flags = info["english_flags"]
            d_flags = info.get("dict_flags", [False] * len(candidates))
            rule_out = info["rule_output"]
            prefix = info.get("prefix", "")
            suffix = info.get("suffix", "")
            total_cands = len(candidates)

            # ── Sinhala passthrough ────────────────────────────────────
            if info.get("sinhala_passthrough"):
                next_beam_si = [(path + [words[t]], sc) for path, sc in beam]
                beam = next_beam_si[:beam_width]
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{words[t]}` (Sinhala passthrough)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=words[t],
                    beam_score=beam[0][1] if beam else 0.0,
                    candidate_breakdown=[],
                ))
                continue

            # ── Common-word shortcut ─────────────────────────────────
            core_lower = words[t].lower().strip()
            if core_lower in COMMON_WORDS:
                override = prefix + COMMON_WORDS[core_lower] + suffix
                next_beam_cw = [(path + [override], sc) for path, sc in beam]
                beam = next_beam_cw[:beam_width]
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{override}` (common-word override)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=override,
                    beam_score=beam[0][1] if beam else 0.0,
                    candidate_breakdown=[],
                ))
                continue

            # ── Context-dependent standalone overrides ────────────────
            if core_lower in CONTEXT_WORDS_STANDALONE:
                prev_word_lower = words[t - 1].lower() if t > 0 else ""
                prev_common_val = COMMON_WORDS.get(prev_word_lower, "")
                prev_is_english = (
                    t > 0
                    and (
                        prev_word_lower in ENGLISH_VOCAB
                        or prev_common_val.isascii() and prev_common_val != ""
                    )
                )
                if not prev_is_english:
                    override = prefix + CONTEXT_WORDS_STANDALONE[core_lower] + suffix
                    next_beam_ctx = [(path + [override], sc) for path, sc in beam]
                    beam = next_beam_ctx[:beam_width]
                    trace_logs.append(
                        f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                        f"`{override}` (standalone override)\n"
                    )
                    diagnostics.append(WordDiagnostic(
                        step_index=t,
                        input_word=words[t],
                        rule_output=rule_out,
                        selected_candidate=override,
                        beam_score=beam[0][1] if beam else 0.0,
                        candidate_breakdown=[],
                    ))
                    continue

            # ── English-word shortcut ────────────────────────────────
            # See greedy decode for detailed comment on criterion.
            core_rule = info.get("core_rule_output", "")
            core_dict = info.get("dict_entries", set())
            is_semantically_ambiguous = (
                core_rule in core_dict and len(core_dict) >= 3
            )
            if (
                len(core_lower) >= MIN_ENGLISH_LEN
                and core_lower in ENGLISH_VOCAB
                and not is_semantically_ambiguous
            ):
                eng_word = words[t]
                next_beam_eng = [(path + [eng_word], sc) for path, sc in beam]
                beam = next_beam_eng[:beam_width]
                trace_logs.append(
                    f"**Step {t + 1}: `{words[t]}`** &nbsp;→ "
                    f"`{eng_word}` (English preserved)\n"
                )
                diagnostics.append(WordDiagnostic(
                    step_index=t,
                    input_word=words[t],
                    rule_output=rule_out,
                    selected_candidate=eng_word,
                    beam_score=beam[0][1] if beam else 0.0,
                    candidate_breakdown=[],
                ))
                continue

            # Build left/right context pairs for multi-mask MLM scoring
            batch_left: List[str] = []
            batch_right: List[str] = []
            batch_tgt: List[str] = []
            batch_meta: List[Tuple[int, int]] = []  # (beam_idx, cand_idx)

            for p_idx, (path, _) in enumerate(beam):
                for c_idx, cand in enumerate(candidates):
                    future = stable_context[t + 1:] if t + 1 < len(words) else []
                    batch_left.append(" ".join(stable_context[:t]))
                    batch_right.append(" ".join(future))
                    batch_tgt.append(cand)
                    batch_meta.append((p_idx, c_idx))

            if not batch_tgt:
                continue

            mlm_scores = self._batch_mlm_score(batch_left, batch_right, batch_tgt)

            # ── Softmax normalise MLM scores ─────────────────────────
            mlm_scores = self._softmax_normalize(mlm_scores)

            # ── MLM floor for English code-switching ─────────────────
            # See greedy decode for detailed comment on criterion.
            best_nonenglish_mlm: Dict[int, float] = {}
            if not is_semantically_ambiguous:
                for i, mlm in enumerate(mlm_scores):
                    p_idx, c_idx = batch_meta[i]
                    is_eng = eng_flags[c_idx] if c_idx < len(eng_flags) else False
                    if not is_eng:
                        prev = best_nonenglish_mlm.get(p_idx, -1e9)
                        if mlm > prev:
                            best_nonenglish_mlm[p_idx] = mlm

            # ── Score & trace ────────────────────────────────────────
            next_beam: List[Tuple[List[str], float]] = []
            all_step_scores: List[Tuple[int, ScoredCandidate, float]] = []
            step_log = f"**Step {t + 1}: `{words[t]}`** &nbsp;(rule → `{rule_out}`)\n\n"

            for i, mlm in enumerate(mlm_scores):
                p_idx, c_idx = batch_meta[i]
                orig_path, orig_score = beam[p_idx]
                cand = batch_tgt[i]
                is_eng = eng_flags[c_idx] if c_idx < len(eng_flags) else False
                is_dict = d_flags[c_idx] if c_idx < len(d_flags) else False

                effective_mlm = mlm
                if is_eng and cand.lower() == words[t].lower() and not is_semantically_ambiguous:
                    floor = best_nonenglish_mlm.get(p_idx, mlm)
                    effective_mlm = max(mlm, floor)

                scored = self.scorer.score(
                    mlm_score=effective_mlm,
                    candidate=cand,
                    rule_output=rule_out,
                    rank=c_idx,
                    total_candidates=total_cands,
                    is_english=is_eng,
                    original_input=words[t],
                    is_from_dict=is_dict,
                    is_ambiguous=is_semantically_ambiguous,
                )

                new_total = orig_score + scored.combined_score
                next_beam.append((orig_path + [cand], new_total))
                all_step_scores.append((p_idx, scored, new_total))

                if mlm > -25.0:
                    eng_tag = " 🔤" if is_eng else ""
                    step_log += (
                        f"- `{cand}`{eng_tag} &nbsp; "
                        f"MLM={scored.mlm_score:.2f} &nbsp; "
                        f"Fid={scored.fidelity_score:.2f} &nbsp; "
                        f"Rank={scored.rank_score:.2f} → "
                        f"**{scored.combined_score:.2f}**\n"
                    )

            trace_logs.append(step_log)

            beam = sorted(next_beam, key=lambda x: x[1], reverse=True)[:beam_width]

            root_scores = [item for item in all_step_scores if item[0] == 0]
            root_scores_sorted = sorted(root_scores, key=lambda x: x[2], reverse=True)

            selected = beam[0][0][t] if beam and beam[0][0] else ""
            selected_total = beam[0][1] if beam else float("-inf")
            candidate_breakdown = [item[1] for item in root_scores_sorted]

            diagnostics.append(WordDiagnostic(
                step_index=t,
                input_word=words[t],
                rule_output=rule_out,
                selected_candidate=selected,
                beam_score=selected_total,
                candidate_breakdown=candidate_breakdown,
            ))

        result = " ".join(beam[0][0]) if beam else ""
        return result, trace_logs, diagnostics
