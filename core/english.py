"""
English vocabulary loader and cache management for code-switch detection.
"""

import os
import logging
import requests
from typing import Set

from core.constants import ENGLISH_CORPUS_URL, MIN_ENGLISH_LEN

logger = logging.getLogger(__name__)

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
    override = os.getenv("SINCODE_ENGLISH_CACHE")
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
