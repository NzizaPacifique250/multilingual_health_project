"""Data loading, language tagging, and task-prefixing for Multilingual Health QA.

The corpus has 8 ``subset`` codes of the form ``<Lang>_<Country>``. Each maps to a
(language, script, country) triple. We build task-prefixed inputs of the form
``"<subset>: <question>"`` so a single seq2seq model can condition generation on the
target language/script without a separate model per language (see PLAN.md, Phase 1).

Verified dataset facts (datasets/Train.csv, Val.csv, Test.csv):
    Train 29,815 rows | Val 6,686 | Test 2,618 (no ``output`` column)
    Columns: ID, input, output, subset  (Test: ID, input, subset)
    No nulls. No input overlap train<->val or train<->test (no leakage).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------------------
# Subset -> language metadata. The dataset uses ISO-ish 3-letter prefixes.
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LangInfo:
    code: str          # subset code, e.g. "Amh_Eth"
    language: str      # human-readable language
    script: str        # Latin | Ge'ez
    country: str
    is_english: bool


SUBSETS: dict[str, LangInfo] = {
    "Eng_Uga": LangInfo("Eng_Uga", "English", "Latin", "Uganda", True),
    "Eng_Gha": LangInfo("Eng_Gha", "English", "Latin", "Ghana", True),
    "Eng_Eth": LangInfo("Eng_Eth", "English", "Latin", "Ethiopia", True),
    "Eng_Ken": LangInfo("Eng_Ken", "English", "Latin", "Kenya", True),
    "Aka_Gha": LangInfo("Aka_Gha", "Akan", "Latin", "Ghana", False),
    "Lug_Uga": LangInfo("Lug_Uga", "Luganda", "Latin", "Uganda", False),
    "Swa_Ken": LangInfo("Swa_Ken", "Swahili", "Latin", "Kenya", False),
    "Amh_Eth": LangInfo("Amh_Eth", "Amharic", "Ge'ez", "Ethiopia", False),
}

# Low-resource non-English targets we watch most closely (smallest + hardest).
LOW_RESOURCE = ["Amh_Eth", "Swa_Ken", "Lug_Uga", "Aka_Gha"]

DATA_DIR_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_raw(split: str, data_dir: str = DATA_DIR_DEFAULT) -> pd.DataFrame:
    """Load one raw split CSV. ``split`` in {"train", "val", "test"}."""
    fname = {"train": "Train.csv", "val": "Val.csv", "test": "Test.csv"}[split.lower()]
    df = pd.read_csv(os.path.join(data_dir, fname))
    expected = {"train": 29815, "val": 6686, "test": 2618}[split.lower()]
    if len(df) != expected:
        # Not fatal (organizers may revise data) but worth surfacing loudly.
        print(f"[data] WARNING: {split} has {len(df)} rows, expected {expected}.")
    return df


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach language/script/country columns derived from ``subset``."""
    df = df.copy()
    df["language"] = df.subset.map(lambda s: SUBSETS[s].language)
    df["script"] = df.subset.map(lambda s: SUBSETS[s].script)
    df["country"] = df.subset.map(lambda s: SUBSETS[s].country)
    df["is_english"] = df.subset.map(lambda s: SUBSETS[s].is_english)
    return df


# --------------------------------------------------------------------------------------
# Task prefixing
# --------------------------------------------------------------------------------------


def make_prefixed_input(question: str, subset: str) -> str:
    """Build the seq2seq source string: ``"<subset>: <question>"``.

    The subset tag steers the model toward the correct language *and script* — the two
    things the lexical-overlap metric rewards most (PLAN.md guiding insight).
    """
    return f"{subset}: {question}"


def add_prefixed_input(df: pd.DataFrame, col: str = "src") -> pd.DataFrame:
    df = df.copy()
    df[col] = [make_prefixed_input(q, s) for q, s in zip(df.input, df.subset)]
    return df


# --------------------------------------------------------------------------------------
# One-call convenience
# --------------------------------------------------------------------------------------


def load_split(
    split: str,
    data_dir: str = DATA_DIR_DEFAULT,
    clean: bool = True,
    prefix: bool = True,
) -> pd.DataFrame:
    """Load a split with metadata, optional cleaning, and task-prefixed ``src`` column.

    ``clean=True`` applies :func:`src.preprocess.normalize_df` (Unicode NFC, whitespace,
    artifact stripping). Imported lazily so ``data.py`` stays importable without the
    preprocess module during minimal EDA.
    """
    df = add_metadata(load_raw(split, data_dir))
    if clean:
        from .preprocess import normalize_df

        df = normalize_df(df)
    if prefix:
        df = add_prefixed_input(df)
    return df


def leakage_report(
    train: pd.DataFrame, val: pd.DataFrame, test: Optional[pd.DataFrame] = None
) -> dict[str, int]:
    """Count exact-input overlaps across splits (train↔val, train↔test)."""
    rep = {
        "train_dup_inputs": int(train.input.duplicated().sum()),
        "train_dup_outputs": int(train.output.duplicated().sum()),
        "train_val_input_overlap": len(set(train.input) & set(val.input)),
    }
    if test is not None:
        rep["train_test_input_overlap"] = len(set(train.input) & set(test.input))
    return rep


if __name__ == "__main__":  # quick smoke test
    tr = load_split("train")
    va = load_split("val")
    te = load_split("test")
    print(tr[["subset", "language", "script", "src"]].head(3).to_string())
    print("\nsubset counts:\n", tr.subset.value_counts().to_string())
    print("\nleakage:", leakage_report(tr, va, te))
