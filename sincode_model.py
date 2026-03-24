"""
SinCode: Context-Aware Singlish-to-Sinhala Transliteration Engine

Architecture (Tiered Decoding):
    1. English Filter    – Preserves code-switched English words
    2. Dictionary Lookup – Retrieves Sinhala candidates from 5.9M-word DB
    3. Phonetic Rules    – Generates fallback transliteration for unknown words
    4. Data-Driven Scorer – Ranks ALL candidates using:
         a) XLM-R MLM contextual probability  (55%, min-max normalised)
         b) Source-aware fidelity              (45%)
    5. Common Word Override – Bypasses scoring for frequent unambiguous words
    6. Beam / Greedy Search – Finds the globally optimal word sequence

Author: Kalana Chandrasekara (2026)
"""

import torch
import math
import re
import os
import pickle
import logging
import requests
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForMaskedLM

logger = logging.getLogger(__name__)

# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
DEFAULT_DICTIONARY_PATH = "dictionary.pkl"

ENGLISH_CORPUS_URL = (
    "https://raw.githubusercontent.com/first20hours/google-10000-english/master/20k.txt"
)

# Scoring weights (tunable hyperparameters)
W_MLM: float = 0.55           # Contextual language model probability
W_FIDELITY: float = 0.45      # Source-aware transliteration fidelity
W_RANK: float = 0.00          # Dictionary rank prior (disabled — dict is unordered)

MAX_CANDIDATES: int = 8       # Max candidates per word position
DEFAULT_BEAM_WIDTH: int = 5   # Beam search width
FIDELITY_SCALE: float = 10.0  # Edit-distance penalty multiplier
DICT_FIDELITY_DAMP: float = 2.0  # Decay rate for dict bonus (higher = stricter filter)
MIN_ENGLISH_LEN: int = 3      # Min word length for 20k-corpus English detection
SINHALA_VIRAMA: str = '\u0DCA'  # Sinhala virama (hal) character
ZWJ: str = '\u200D'             # Zero-width joiner (for conjuncts)

# Precompiled regex for punctuation stripping
PUNCT_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$")

# Core English words always recognised (supplements the 20k corpus)
CORE_ENGLISH_WORDS: Set[str] = {
    "transliteration", "sincode", "prototype", "assignment", "singlish",
    "rest", "complete", "tutorial", "small", "mistakes", "game", "play",
    "type", "test", "online", "code", "mixing", "project", "demo", "today",
    "tomorrow", "presentation", "slide", "submit", "feedback", "deploy",
    "merge", "update", "delete", "download", "upload", "install", "server",
    "meeting", "backlog", "comment", "reply", "chat", "selfie", "post",
    "share", "private", "message", "group", "study", "exam", "results",
    "viva", "prepared", "site", "redo", "story", "poll",
    "hall", "exam", "PR", "DM", "page", "app", "bug", "fix",
    "log", "push", "pull", "branch", "build", "run", "save",
    "link", "edit", "file", "open", "close", "live", "view",
}


def _resolve_english_cache_path() -> str:
    """
    Resolve a writable cache path for the English corpus.

    Hugging Face Spaces may run with constrained write locations, so we prefer:
    1) explicit env override,
    2) HF_HOME cache dir,
    3) local working dir,
    4) system temp dir.
    """
    override = os.getenv("SICODE_ENGLISH_CACHE")
    if override:
        return override

    candidates = [
        os.path.join(os.getenv("HF_HOME", ""), "english_20k.txt") if os.getenv("HF_HOME") else "",
        os.path.join(os.getcwd(), "english_20k.txt"),
        os.path.join(os.getenv("TMPDIR", os.getenv("TEMP", "/tmp")), "english_20k.txt"),
    ]

    for path in candidates:
        if not path:
            continue
        parent = os.path.dirname(path) or "."
        try:
            os.makedirs(parent, exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass
            return path
        except OSError:
            continue

    return "english_20k.txt"


ENGLISH_CORPUS_CACHE = _resolve_english_cache_path()


# ─── English Vocabulary ─────────────────────────────────────────────────────

def load_english_vocab() -> Set[str]:
    """Load and cache a ~20k English word list for code-switch detection."""
    vocab = CORE_ENGLISH_WORDS.copy()

    if not os.path.exists(ENGLISH_CORPUS_CACHE):
        try:
            logger.info("Downloading English corpus...")
            response = requests.get(ENGLISH_CORPUS_URL, timeout=10)
            response.raise_for_status()
            with open(ENGLISH_CORPUS_CACHE, "wb") as f:
                f.write(response.content)
        except (requests.RequestException, OSError) as exc:
            logger.warning("Could not download English corpus: %s", exc)
            return vocab

    try:
        with open(ENGLISH_CORPUS_CACHE, "r", encoding="utf-8") as f:
            vocab.update(
                w for line in f
                if (w := line.strip().lower()) and len(w) >= MIN_ENGLISH_LEN
            )
    except OSError as exc:
        logger.warning("Could not read English corpus file: %s", exc)

    logger.info("English vocabulary loaded: %d words", len(vocab))
    return vocab


ENGLISH_VOCAB: Set[str] = load_english_vocab()


# ─── Common Word Overrides ──────────────────────────────────────────────────
# High-frequency Singlish words whose romanisation is ambiguous (long vs.
# short vowel, retroflex vs. dental, etc.).  When a word appears here the
# decoder uses the override directly, bypassing MLM/fidelity scoring.
# Only add words that are *unambiguous* — i.e. one dominant Sinhala form
# in colloquial written chat.  Context-dependent words (e.g. "eka") should
# NOT be listed so that MLM can resolve them.

COMMON_WORDS: Dict[str, str] = {
    # Pronouns & particles
    "oya":      "ඔයා",       # you
    "oyaa":     "ඔයා",
    "eya":      "ඒයා",       # he/she
    "eyaa":     "ඒයා",
    "api":      "අපි",       # we
    "mama":     "මම",        # I
    "mage":     "මගේ",       # my
    "oyage":    "ඔයාගේ",     # your
    # Common verbs (past tense)
    "awa":      "ආවා",       # came
    "aawa":     "ආවා",
    "giya":     "ගියා",       # went
    "kala":     "කළා",       # did
    "kiwa":     "කිව්වා",      # said
    "kiwwa":    "කිව්වා",
    "yewwa":    "යැව්වා",     # sent
    "gawa":     "ගැව්වා",     # hit
    "katha":    "කතා",       # talked / story
    # Time
    "heta":     "හෙට",       # tomorrow
    "ada":      "අද",        # today
    "iye":      "ඊයේ",       # yesterday
    # Common adverbs / particles
    "one":      "ඕනෙ",       # need/want
    "oney":     "ඕනේ",
    "naa":      "නෑ",        # no (long form)
    "na":       "නෑ",        # no
    "hari":     "හරි",        # ok / right
    "wage":     "වගේ",       # like
    "nisa":     "නිසා",       # because
    "inne":     "ඉන්නෙ",     # being/staying (colloquial)
    "inna":     "ඉන්න",      # stay (imperative)
    "kalin":    "කලින්",      # before / earlier
    # Common verb endings
    "giye":     "ගියේ",       # went (emphatic)
    "una":      "උනා",       # became / happened
    "wuna":     "උනා",       # became (alt spelling)
    # Locations / misc
    "gedaradi": "ගෙදරදී",     # at home
    "gedara":   "ගෙදර",       # home
    # Common adjectives / other
    "honda":    "හොඳ",       # good
    "ape":      "අපේ",       # our
    "me":       "මේ",        # this
    "passe":    "පස්සෙ",      # after / later
    "ba":       "බෑ",        # can't
    "bari":     "බැරි",       # impossible
    "bri":      "බැරි",       # can't (abbrev)
    "danne":    "දන්නෙ",     # know
    "wada":     "වැඩ",       # work (noun)
    "epa":      "එපා",       # don't
    # Common ad-hoc abbreviations
    "mta":      "මට",        # mata
    "oyta":     "ඔයාට",      # oyata
    "oyata":    "ඔයාට",      # to you
    "krnna":    "කරන්න",     # karanna
    "blnna":    "බලන්න",     # balanna
    "on":       "ඕනෙ",       # one (abbrev)
    # Common -nawa verb endings
    "thiyanawa": "තියෙනවා",   # is/has
    "wenawa":   "වෙනවා",     # becomes
    "enawa":    "එනවා",      # comes
    "yanawa":   "යනවා",      # goes
    "hithenawa":"හිතෙනවා",   # thinks/feels
    "penenawa": "පේනවා",     # appears/visible
    "karamu":   "කරමු",      # let's do
    "balamu":   "බලමු",      # let's see
    "damu":     "දාමු",       # let's put
    "yamu":     "යමු",        # let's go
    # Short English abbreviations (keys are lowercase for lookup)
    "pr":       "PR",
    "dm":       "DM",
    "ai":       "AI",
    "it":       "IT",
    "qa":       "QA",
    "ui":       "UI",
    "ok":       "OK",
    # Common ad-hoc abbreviations (contd.)
    "ek":       "එක",        # eka (short form)
    "ekta":     "එකට",       # ekata = to that one
    "ekat":     "ඒකට",       # that-thing + to (standalone form)
    "eke":      "එකේ",       # of that one
    "hta":      "හෙට",       # heta (abbrev)
    "damma":    "දැම්මා",    # put/posted
    "gannako":  "ගන්නකෝ",   # take (imperative, long ō)
    # Additional words for accuracy
    "gena":     "ගැන",       # about
    "mata":     "මට",        # to me
    "laga":     "ළඟ",        # near
    "poth":     "පොත",       # book
    "iwara":    "ඉවර",       # finished
    "karanna":  "කරන්න",     # to do
    "hadamu":   "හදමු",      # let's make
    "kiyawala":  "කියවලා",    # having read
    "baya":     "බය",        # fear/scared
}

# Context-dependent words: use this form ONLY when the previous word is
# NOT English. When "eka" follows an English noun (e.g., "assignment eka")
# the scorer resolves it to එක naturally; standalone "eka" maps to ඒක.
CONTEXT_WORDS_STANDALONE: Dict[str, str] = {
    "eka":  "ඒක",     # that thing (standalone)
    "ekak": "එකක්",   # one of (quantifier — same either way)
}


# ─── Rule-Based Transliteration Engine ───────────────────────────────────────
# Phonetic mapping tables (Singlish Romanized → Sinhala Unicode)
# Tables are ordered longest-pattern-first so greedy replacement works.

CONSONANTS: List[str] = [
    "nnd", "nndh", "nng",
    "th", "dh", "gh", "ch", "ph", "bh", "jh", "sh",
    "GN", "KN", "Lu", "kh", "Th", "Dh",
    "S", "d", "c", "th", "t", "k", "D", "n", "p", "b", "m",
    "\\y",
    "Y", "y", "j", "l", "v", "w", "s", "h",
    "N", "L", "K", "G", "P", "B", "f", "g", "r",
]

CONSONANTS_UNI: List[str] = [
    "ඬ", "ඳ", "ඟ",
    "ත", "ධ", "ඝ", "ච", "ඵ", "භ", "ඣ", "ෂ",
    "ඥ", "ඤ", "ළු", "ඛ", "ඨ", "ඪ",
    "ශ", "ද", "ච", "ත", "ට", "ක", "ඩ", "න", "ප", "බ", "ම",
    "‍ය",
    "‍ය", "ය", "ජ", "ල", "ව", "ව", "ස", "හ",
    "ණ", "ළ", "ඛ", "ඝ", "ඵ", "ඹ", "ෆ", "ග", "ර",
]

VOWELS: List[str] = [
    "oo", "o\\)", "oe", "aa", "a\\)", "Aa", "A\\)", "ae",
    "ii", "i\\)", "ie", "ee", "ea", "e\\)", "ei",
    "uu", "u\\)", "au",
    "\\a", "a", "A", "i", "e", "u", "o", "I",
]

VOWELS_UNI: List[str] = [
    "ඌ", "ඕ", "ඕ", "ආ", "ආ", "ඈ", "ඈ", "ඈ",
    "ඊ", "ඊ", "ඊ", "ඊ", "ඒ", "ඒ", "ඒ",
    "ඌ", "ඌ", "ඖ",
    "ඇ", "අ", "ඇ", "ඉ", "එ", "උ", "ඔ", "ඓ",
]

VOWEL_MODIFIERS_UNI: List[str] = [
    "ූ", "ෝ", "ෝ", "ා", "ා", "ෑ", "ෑ", "ෑ",
    "ී", "ී", "ී", "ී", "ේ", "ේ", "ේ",
    "ූ", "ූ", "ෞ",
    "ැ", "", "ැ", "ි", "ෙ", "ු", "ො", "ෛ",
]

SPECIAL_CONSONANTS: List[str] = ["\\n", "\\h", "\\N", "\\R", "R", "\\r"]
SPECIAL_CONSONANTS_UNI: List[str] = ["ං", "ඃ", "ඞ", "ඍ", "ර්\u200D", "ර්\u200D"]

SPECIAL_CHARS: List[str] = ["ruu", "ru"]
SPECIAL_CHARS_UNI: List[str] = ["ෲ", "ෘ"]

N_VOWELS: int = 26


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


# ─── Data-Driven Candidate Scorer ───────────────────────────────────────────

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

    1. **MLM Score** (weight α = 0.40)
       Contextual fit from XLM-RoBERTa masked language model.
       English candidates matching the user's input receive an
       MLM floor (best non-English score) to remove XLM-R's
       cross-script calibration bias.

    2. **Source-Aware Fidelity** (weight β = 0.60)
       English candidates matching input → 0.0 (user intent).
       Dictionary candidates → damped (50%) Levenshtein to rule
       output — validates as real word but still rewards phonetic
       closeness to the typed input.
       Rule-only outputs → penalised by virama/skeleton density.
       Other → full Levenshtein distance to rule output.

    Note: Dictionary rank prior is disabled (γ = 0.0) because the
    dictionary entries are not ordered by frequency.
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
    ) -> float:
        """
        Source-aware transliteration fidelity.

        - **English matching input** → 0.0  (user-intent preservation).
        - **Dict + matches rule output** → strong bonus (+2.0). Both
          signals agree — highest confidence.
        - **Dict only** → decaying bonus (1.0 down to 0.0 with distance
          from rule output).  Still a real word, but less certain.
        - **Rule-only outputs not in dictionary** → penalised by
          consonant-skeleton density (high virama ratio = malformed).
        - **Other** → full Levenshtein distance to rule output.
        """
        # 1. English candidate matching the original input word
        if original_input and candidate.lower() == original_input.lower():
            return 0.0

        # 2. Dictionary-validated candidates
        if is_from_dict:
            # Rule output confirmed by dictionary = highest confidence
            if candidate == rule_output:
                return 2.0
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
        """
        Log-decay rank prior.  First candidate → 0.0; later ones decay.
        """
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
    ) -> ScoredCandidate:
        """Return a :class:`ScoredCandidate` with full breakdown."""
        fidelity = self.compute_fidelity(
            candidate, rule_output, original_input, is_from_dict,
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


# ─── Dictionary Adapter ─────────────────────────────────────────────────────

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


# ─── Beam Search Decoder ────────────────────────────────────────────────────

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
        at that one position (which conflates alternative predictions at
        the same slot), this method creates one <mask> per subword token
        and scores each subword at its own position:

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
                  "beam" (faster, uses fixed rule-based context)

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
                "english_flags": english_flags[:MAX_CANDIDATES],
                "dict_flags": dict_flags[:MAX_CANDIDATES],
                "prefix": prefix,
                "suffix": suffix,
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
            # Words like "eka" that change form depending on whether the
            # previous word was English (e.g., "assignment eka" → එක)
            # or Sinhala / start of sentence ("eka heta" → ඒක).
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
            if (
                len(core_lower) >= MIN_ENGLISH_LEN
                and core_lower in ENGLISH_VOCAB
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

            # ── Min-max normalise MLM to [0, 1] ─────────────────────
            # Raw log-probs span a wide range (e.g. −5 to −25) and can
            # drown out fidelity.  Per-position normalisation makes the
            # two signals weight-comparable.
            mlm_min = min(mlm_scores)
            mlm_max = max(mlm_scores)
            mlm_range = mlm_max - mlm_min
            if mlm_range > 1e-9:
                mlm_scores = [(m - mlm_min) / mlm_range for m in mlm_scores]
            else:
                mlm_scores = [1.0] * len(mlm_scores)

            # MLM floor for English code-switching
            best_nonenglish_mlm = -1e9
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
                if is_eng and cand.lower() == words[t].lower():
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

    def decode_with_diagnostics(
        self,
        sentence: str,
        beam_width: int = DEFAULT_BEAM_WIDTH,
    ) -> Tuple[str, List[str], List[WordDiagnostic]]:
        """
        Decode sentence and return detailed per-word diagnostics.

        Returns:
            result            – best transliterated sentence
            trace_logs        – markdown logs used by Streamlit UI
            diagnostics       – structured scores and selected candidates per step
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
                })
                continue

            rule_output = self.adapter.get_rule_output(core)
            cands = self.adapter.get_candidates(core, rule_output)

            # Track which candidates are dictionary-validated
            dict_entries: Set[str] = set()
            if core in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core])
            elif core.lower() in self.adapter.dictionary:
                dict_entries.update(self.adapter.dictionary[core.lower()])

            # Always include the rule output so the model can consider it
            if rule_output and rule_output not in cands:
                cands.append(rule_output)

            # If still empty, use rule output as sole candidate
            if not cands:
                cands = [rule_output]

            english_flags = [c.lower() in ENGLISH_VOCAB for c in cands]
            dict_flags = [c in dict_entries for c in cands]

            # Apply punctuation wrappers
            full_cands = [prefix + c + suffix for c in cands]

            word_infos.append({
                "candidates": full_cands[:MAX_CANDIDATES],
                "rule_output": prefix + rule_output + suffix,
                "english_flags": english_flags[:MAX_CANDIDATES],
                "dict_flags": dict_flags[:MAX_CANDIDATES],
                "prefix": prefix,
                "suffix": suffix,
            })

        # Build stable context for ALL positions (both left and right
        # in MLM scoring).  English forms are preferred for English-
        # detected words so the context reflects the code-mixed nature
        # of the output rather than being 100% Sinhala.
        #
        # Using a FIXED context instead of the beam path for the left
        # side prevents cascade errors: different beam paths produce
        # different MLM scores for the same candidate, allowing early
        # mistakes to propagate through noisy contextual fluctuations.
        # With stable context, each candidate gets ONE consistent MLM
        # score per position, and the system picks the phonetically +
        # contextually best option at every step.
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

            # ── Common-word shortcut ─────────────────────────────────
            core_lower = words[t].lower().strip()
            if core_lower in COMMON_WORDS:
                override = prefix + COMMON_WORDS[core_lower] + suffix
                # Extend every beam path with the override
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
            if (
                len(core_lower) >= MIN_ENGLISH_LEN
                and core_lower in ENGLISH_VOCAB
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

            # ── Min-max normalise MLM to [0, 1] ─────────────────────
            mlm_min = min(mlm_scores) if mlm_scores else 0
            mlm_max = max(mlm_scores) if mlm_scores else 0
            mlm_range = mlm_max - mlm_min
            if mlm_range > 1e-9:
                mlm_scores = [(m - mlm_min) / mlm_range for m in mlm_scores]
            else:
                mlm_scores = [1.0] * len(mlm_scores)

            # ── MLM floor for English code-switching ─────────────────
            # XLM-R is not calibrated for Singlish code-mixing: English
            # tokens in Sinhala context receive disproportionately low
            # MLM scores.  For English candidates that match the user's
            # input, cap their disadvantage at the best non-English MLM
            # score in the same beam, removing cross-script bias while
            # preserving relative ranking information.
            best_nonenglish_mlm: Dict[int, float] = {}
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

                # Apply MLM floor for English candidates matching input
                effective_mlm = mlm
                if is_eng and cand.lower() == words[t].lower():
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
                )

                new_total = orig_score + scored.combined_score
                next_beam.append((orig_path + [cand], new_total))
                all_step_scores.append((p_idx, scored, new_total))

                # Trace log (skip very low scores to reduce noise)
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

            # Capture diagnostics from the root beam path (p_idx=0) so each
            # step has a stable and comparable candidate distribution.
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
