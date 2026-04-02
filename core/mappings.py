"""
Static mapping tables for the SinCode engine.

Includes common-word overrides, context-dependent overrides,
and phonetic mapping tables (consonants, vowels, modifiers).
"""

from typing import Dict, List

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
    "madi":     "මදි",        # insufficient / not enough
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
    "mn":       "මං",        # man (I, informal first person)
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
    "ok":       "ok",
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
    # Ad-hoc and alternative spellings (accuracy fixes)
    "kema":      "කෑම",       # food (colloquial spelling)
    "kama":      "කෑම",       # food (alt spelling)
    "hodai":     "හොඳයි",    # good! (no-n spelling)
    "oyge":      "ඔයාගෙ",    # your (shortened form)
    "iwra":      "ඉවර",       # finished (vowel-stripped)
    "krd":       "කරාද",      # did? (extreme abbreviation)
    "handawata": "හැන්දෑවට", # in the evening
    "wenwa":     "වෙනවා",     # becomes/happens
    "ep":        "එපා",       # epa (single-syllable abbrev)
    "prashnya":  "ප්\u200dරශ්\u200dනය",  # question (without final vowel)
    # ── Verb forms / participles (no English conflict) ────────────────────
    "penawa":    "පේනවා",     # appears/visible (alt spelling of penenawa)
    "thiyana":   "තියෙන",     # that which is/exists (relative participle)
    "enakota":   "එනකොට",    # when (you/they) come
    "hadanna":   "හදන්න",     # to make/build (imperative)
    "yawwa":     "යැව්වා",    # sent (alt spelling of yewwa)
    "gihilla":   "ගිහිල්ලා",  # having gone
    "kewata":    "කෑවට",      # having eaten / for the eating
    "kiyla":     "කියලා",     # having said (ad-hoc spelling)
    "krganna":   "කරගන්න",   # to do-and-get (ad-hoc abbreviation)
    # ── Adjectives (no English conflict) ────────────────────────────────────
    "amarui":    "අමාරුයි",   # difficult / hard
    "hodama":    "හොඳම",      # best (superlative of honda)
    # ── Particles / negation (no English conflict) ───────────────────────────
    "nathi":     "නැති",      # without / lacking (negation)
    "nati":      "නැති",      # without (alt spelling)
    "naththe":   "නැත්තෙ",   # negative participle (not ...ing)
    "dan":       "දැන්",      # now
    "oni":       "ඕනි",       # need/want (alt spelling of one)
    # ── Time ────────────────────────────────────────────────────────────────
    "udee":      "උදේ",       # morning
    # ── Ad-hoc abbreviations (no English conflict) ───────────────────────────
    "hri":       "හරි",       # ok/right (shortened hari)
    "mge":       "මගේ",       # my (shortened mage)
}

# Context-dependent words: use this form ONLY when the previous word is
# NOT English. When "eka" follows an English noun (e.g., "assignment eka")
# the scorer resolves it to එක naturally; standalone "eka" maps to ඒක.
CONTEXT_WORDS_STANDALONE: Dict[str, str] = {
    "eka":  "ඒක",     # that thing (standalone)
    "ekak": "එකක්",   # one of (quantifier — same either way)
}


# ─── Phonetic Mapping Tables ────────────────────────────────────────────────
# Singlish Romanized → Sinhala Unicode
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
