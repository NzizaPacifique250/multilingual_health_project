"""Text normalization for multilingual health QA, including Ge'ez (Amharic) handling.

Design choices (justified in notebooks/01_eda.ipynb):
  * Unicode NFC normalization — Ge'ez/Latin combining sequences collapse to canonical
    forms so the same visual string tokenizes identically (matters for ROUGE token
    overlap and for SentencePiece coverage).
  * Whitespace collapse + strip — removes the stray double-spaces / newlines that
    otherwise inflate or fragment tokens.
  * Conservative artifact stripping — zero-width chars, BOM, control chars. We do NOT
    lowercase or strip punctuation: ROUGE is computed on the reference *as written*, and
    Ge'ez has no case, so aggressive normalization would only diverge from references.

Everything here is reference-preserving: we normalize predictions and references the
same way, and we keep the target side faithful to the gold text.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Zero-width + BOM + other invisibles that creep in from web-scraped corpora.
_INVISIBLES = re.compile(r"[​‌‍\u200E\u200F﻿­]")
# C0/C1 control chars except tab/newline (which we then collapse anyway).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WS = re.compile(r"\s+")

# Ge'ez (Ethiopic) Unicode blocks — used for script sanity checks on Amharic rows.
_GEEZ = re.compile(r"[ሀ-፿ᎀ-᎟ⶀ-⷟꬀-꬯]")


def normalize_text(s: str) -> str:
    """Reference-preserving normalization applied to inputs, outputs, and predictions."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFC", s)
    s = _INVISIBLES.sub("", s)
    s = _CONTROL.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``input`` and (if present) ``output`` columns in place on a copy."""
    df = df.copy()
    df["input"] = df["input"].map(normalize_text)
    if "output" in df.columns:
        df["output"] = df["output"].map(normalize_text)
    return df


# --------------------------------------------------------------------------------------
# Script diagnostics (used by EDA + as an inference-time wrong-language guard)
# --------------------------------------------------------------------------------------


def geez_ratio(s: str) -> float:
    """Fraction of non-space characters that fall in Ge'ez/Ethiopic blocks."""
    chars = [c for c in str(s) if not c.isspace()]
    if not chars:
        return 0.0
    return sum(bool(_GEEZ.match(c)) for c in chars) / len(chars)


def looks_amharic(s: str, threshold: float = 0.5) -> bool:
    """True if the string is predominantly Ge'ez script (Amharic sanity check)."""
    return geez_ratio(s) >= threshold


def latin_ratio(s: str) -> float:
    chars = [c for c in str(s) if c.isalpha()]
    if not chars:
        return 0.0
    n = 0
    for c in chars:
        try:
            n += "LATIN" in unicodedata.name(c)
        except ValueError:
            pass
    return n / len(chars)


if __name__ == "__main__":
    samples = [
        "  Hello   world​!  ",
        "ጤና ይስጥልኝ፣ እንዴት ነዎት?",
    ]
    for s in samples:
        n = normalize_text(s)
        print(repr(n), "geez=%.2f" % geez_ratio(n), "latin=%.2f" % latin_ratio(n))
