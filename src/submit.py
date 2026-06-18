"""Format a Zindi submission CSV for Multilingual Health QA — LOCAL FILE ONLY.

IMPORTANT: nothing here uploads or submits anywhere. These helpers only assemble and
validate a local CSV against the competition's column contract so YOU can review the
predictions and upload manually when satisfied. There is no network call in this module.

Required: exactly 4 columns ``ID, TargetRLF1, TargetR1F1, TargetLLM`` where the three
target columns hold the **identical** predicted answer text for each row (the platform
duplicates one answer across all three metrics — see COMPETITION.md).
"""
from __future__ import annotations

import os

import pandas as pd

SUB_COLS = ["ID", "TargetRLF1", "TargetR1F1", "TargetLLM"]


def build_submission(ids, predictions) -> pd.DataFrame:
    """Assemble a valid submission frame from row IDs and predicted answers."""
    ids = list(ids)
    preds = [("" if p is None else str(p)) for p in predictions]
    if len(ids) != len(preds):
        raise ValueError(f"ids ({len(ids)}) and predictions ({len(preds)}) length mismatch")
    # Empty predictions score 0 and may be rejected; replace with a single space.
    preds = [p if p.strip() else " " for p in preds]
    return pd.DataFrame(
        {"ID": ids, "TargetRLF1": preds, "TargetR1F1": preds, "TargetLLM": preds}
    )[SUB_COLS]


def validate_submission(df: pd.DataFrame, sample_path: str | None = None) -> pd.DataFrame:
    """Assert the frame matches the contract; optionally check IDs vs SampleSubmission."""
    if list(df.columns) != SUB_COLS:
        raise ValueError(f"columns must be exactly {SUB_COLS}, got {list(df.columns)}")
    if df[["TargetRLF1", "TargetR1F1", "TargetLLM"]].nunique(axis=1).max() != 1:
        raise ValueError("the three Target* columns must be identical per row")
    if df.ID.duplicated().any():
        raise ValueError("duplicate IDs in submission")
    if df[SUB_COLS].isna().any().any():
        raise ValueError("submission contains NaN values")
    if sample_path and os.path.exists(sample_path):
        want = set(pd.read_csv(sample_path).ID)
        got = set(df.ID)
        if want != got:
            raise ValueError(
                f"ID mismatch vs sample: missing {len(want - got)}, extra {len(got - want)}"
            )
    return df


def save_submission(df: pd.DataFrame, path: str, sample_path: str | None = None) -> str:
    """Validate then write the CSV locally. Does NOT upload anywhere — review first."""
    validate_submission(df, sample_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[submit] wrote {len(df)} rows -> {path} (LOCAL ONLY — review before uploading)")
    return path
