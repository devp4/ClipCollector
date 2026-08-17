"""
text_cleanup.py

Lightweight, deterministic cleanup applied to caption text ONLY (not the
raw transcript used for Gemini analysis or the .srt/.txt files) -- fixes
common ASR homophone errors (see config.WORD_CORRECTIONS) and strips
punctuation.

Why not just have Gemini rewrite the transcript instead? Because captions
are built from Whisper's WORD-LEVEL timestamps -- one timestamp per token.
An LLM rewrite can change word count/boundaries (e.g. "gonna" -> "going
to"), which breaks the timestamp alignment captions depend on. A word-level
dictionary swap preserves the token count 1:1, so timing stays exact.
"""

import re

import config


def strip_punctuation(token):
    """
    Removes ALL punctuation from a token -- periods, commas, question
    marks, exclamation points, quotes, colons, etc. -- while preserving an
    APOSTROPHE THAT SITS INSIDE A WORD (e.g. "don't", "I'm"), since that's
    part of the word's spelling rather than sentence punctuation. A leading
    or trailing apostrophe (e.g. from a stray quote) is still stripped.
    """
    cleaned = re.sub(r"[^\w\s']", "", token, flags=re.UNICODE)
    cleaned = cleaned.strip("'")
    return cleaned


def correct_word(token):
    """Applies config.WORD_CORRECTIONS to a whole token (case-insensitive match)."""
    replacement = config.WORD_CORRECTIONS.get(token.lower())
    return replacement if replacement else token


def clean_caption_token(token):
    """
    Full cleanup pipeline for one caption word: strip all punctuation, then
    apply the correction dictionary. Order matters -- stripping punctuation
    first means the dictionary match isn't thrown off by trailing marks.
    """
    token = strip_punctuation(token)
    token = correct_word(token)
    return token
