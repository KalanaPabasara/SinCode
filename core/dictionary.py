"""
Dictionary adapter for retrieving Sinhala transliteration candidates.
"""

from typing import Dict, List, Set

from core.constants import MAX_CANDIDATES
from core.english import ENGLISH_VOCAB
from core.scorer import CandidateScorer
from core.transliterate import rule_based_transliterate


class DictionaryAdapter:
    """Retrieves transliteration candidates from the Sinhala dictionary."""

    def __init__(self, dictionary_dict: Dict[str, List[str]]):
        self.dictionary = dictionary_dict

    def get_candidates(self, word: str, rule_output: str = "") -> List[str]:
        """
        Return candidate transliterations for a Romanized word.

        Priority:
            1. English corpus match  → keep original word
            2. Dictionary lookup     → exact / lowercase
            3. Subword decomposition → only when 1 & 2 yield nothing

        When more candidates exist than MAX_CANDIDATES, results are
        sorted by Levenshtein distance to ``rule_output`` so the most
        phonetically plausible entries survive the cut.
        """
        cands: List[str] = []
        word_lower = word.lower()

        # 1. English corpus check
        if word_lower in ENGLISH_VOCAB:
            cands.append(word)

        # 2. Sinhala dictionary check
        if word in self.dictionary:
            cands.extend(self.dictionary[word])
        elif word_lower in self.dictionary:
            cands.extend(self.dictionary[word_lower])

        # 3. Deduplicate preserving order
        if cands:
            cands = list(dict.fromkeys(cands))
            # Sort Sinhala candidates by closeness to rule output
            if rule_output and len(cands) > MAX_CANDIDATES:
                english = [c for c in cands if c.lower() in ENGLISH_VOCAB]
                sinhala = [c for c in cands if c.lower() not in ENGLISH_VOCAB]
                sinhala.sort(
                    key=lambda c: CandidateScorer.levenshtein(c, rule_output)
                )
                cands = english + sinhala
            return cands

        # 4. Subword fallback (compound words)
        length = len(word)
        if length > 3:
            for i in range(2, length - 1):
                part1, part2 = word[:i], word[i:]
                p1 = self.dictionary.get(part1) or self.dictionary.get(part1.lower())
                p2 = self.dictionary.get(part2) or self.dictionary.get(part2.lower())

                if p1 and p2:
                    for w1 in p1[:3]:
                        for w2 in p2[:3]:
                            cands.append(w1 + w2)

        return list(dict.fromkeys(cands)) if cands else []

    @staticmethod
    def get_rule_output(word: str) -> str:
        """Generate Sinhala output via the phonetic rule engine."""
        return rule_based_transliterate(word)
