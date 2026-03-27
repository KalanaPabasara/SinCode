"""
Rule-based phonetic transliteration engine (Singlish → Sinhala Unicode).
"""

from core.mappings import (
    CONSONANTS, CONSONANTS_UNI,
    VOWELS, VOWELS_UNI, VOWEL_MODIFIERS_UNI,
    SPECIAL_CONSONANTS, SPECIAL_CONSONANTS_UNI,
    SPECIAL_CHARS, SPECIAL_CHARS_UNI,
    N_VOWELS,
)


def rule_based_transliterate(text: str) -> str:
    """
    Convert Romanized Singlish text to Sinhala script using phonetic rules.

    Replacement order matters: longer patterns are consumed first so that
    greedy left-to-right substitution produces correct output.
    """
    # 1. Special consonants (anusvara, visarga, etc.)
    for pat, uni in zip(SPECIAL_CONSONANTS, SPECIAL_CONSONANTS_UNI):
        text = text.replace(pat, uni)

    # 2. Consonant + special-char combinations (e.g., kru → කෘ)
    for sc, sc_uni in zip(SPECIAL_CHARS, SPECIAL_CHARS_UNI):
        for cons, cons_uni in zip(CONSONANTS, CONSONANTS_UNI):
            text = text.replace(cons + sc, cons_uni + sc_uni)

    # 3. Consonant + ra + vowel clusters (e.g., kra → ක්‍රා)
    for cons, cons_uni in zip(CONSONANTS, CONSONANTS_UNI):
        for vow, vmod in zip(VOWELS, VOWEL_MODIFIERS_UNI):
            text = text.replace(cons + "r" + vow, cons_uni + "්‍ර" + vmod)
        text = text.replace(cons + "r", cons_uni + "්‍ර")

    # 4. Consonant + vowel combinations
    for cons, cons_uni in zip(CONSONANTS, CONSONANTS_UNI):
        for j in range(N_VOWELS):
            text = text.replace(cons + VOWELS[j], cons_uni + VOWEL_MODIFIERS_UNI[j])

    # 5. Bare consonants → consonant + hal (virama)
    for cons, cons_uni in zip(CONSONANTS, CONSONANTS_UNI):
        text = text.replace(cons, cons_uni + "්")

    # 6. Standalone vowels
    for vow, vow_uni in zip(VOWELS, VOWELS_UNI):
        text = text.replace(vow, vow_uni)

    return text
