"""Local leaderboard replication for Multilingual Health QA.

The organizers score with the ``rouge-score`` library using a **whitespace tokenizer**
and ``use_stemmer=False``, calling ``scorer.score(reference, prediction)`` and averaging
the per-row F-measures (verified from the official starter notebook, cell 13). We mirror
that byte-for-byte so our Val numbers track the hidden leaderboard.

Final leaderboard = 0.37·ROUGE-1 F1 + 0.37·ROUGE-L F1 + 0.26·LLM-judge.
We cannot reproduce the LLM judge, so our offline north-star is the lexical proxy
``0.5·R1 + 0.5·RL`` (the judge slice is held out). :func:`weighted_score` also accepts an
optional judge score for completeness / ablations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rouge_score import rouge_scorer

W_R1, W_RL, W_LLM = 0.37, 0.37, 0.26


class WhitespaceTokenizer:
    """Language-agnostic tokenizer — splits on whitespace, safe for African scripts.

    Identical to the organizers' tokenizer: aggressive (English-centric) tokenization
    would diverge from the hidden grader on Ge'ez and other non-Latin text.
    """

    def tokenize(self, text):
        if text is None:
            return []
        return str(text).strip().split()


def _make_scorer() -> rouge_scorer.RougeScorer:
    return rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        tokenizer=WhitespaceTokenizer(),
        use_stemmer=False,
    )


@dataclass
class RougeResult:
    rouge1_f1: float
    rougeL_f1: float

    @property
    def proxy(self) -> float:
        """Offline north-star: lexical half of the leaderboard (0.5·R1 + 0.5·RL)."""
        return 0.5 * self.rouge1_f1 + 0.5 * self.rougeL_f1

    def as_dict(self) -> dict[str, float]:
        return {
            "rouge1_f1": self.rouge1_f1,
            "rougeL_f1": self.rougeL_f1,
            "proxy": self.proxy,
        }


def compute_rouge(predictions, references) -> RougeResult:
    """Mean ROUGE-1 / ROUGE-L F1 over aligned (prediction, reference) lists."""
    scorer = _make_scorer()
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(str(ref), str(pred))  # (target, prediction) order matters
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return RougeResult(
        rouge1_f1=float(np.mean(r1)) if r1 else 0.0,
        rougeL_f1=float(np.mean(rl)) if rl else 0.0,
    )


def compute_rouge_by_group(predictions, references, groups):
    """Per-group (e.g. per-subset / per-language) ROUGE breakdown.

    Returns a dict ``{group: {rouge1_f1, rougeL_f1, proxy, n}}``, plus an ``ALL`` row.
    """
    preds = np.asarray(predictions, dtype=object)
    refs = np.asarray(references, dtype=object)
    grp = np.asarray(groups)
    out: dict[str, dict[str, float]] = {}
    for g in sorted(np.unique(grp)):
        m = grp == g
        res = compute_rouge(preds[m].tolist(), refs[m].tolist())
        out[str(g)] = {**res.as_dict(), "n": int(m.sum())}
    overall = compute_rouge(predictions, references)
    out["ALL"] = {**overall.as_dict(), "n": int(len(grp))}
    return out


def weighted_score(rouge1_f1: float, rougeL_f1: float, llm_judge: float | None = None) -> float:
    """Full leaderboard formula. If ``llm_judge`` is None, returns the lexical proxy
    rescaled to the 0.74 lexical weight (i.e. assumes judge contributes nothing)."""
    if llm_judge is None:
        return W_R1 * rouge1_f1 + W_RL * rougeL_f1  # judge slice = 0
    return W_R1 * rouge1_f1 + W_RL * rougeL_f1 + W_LLM * llm_judge


if __name__ == "__main__":
    preds = ["the cat sat on the mat", "ጤና ይስጥልኝ እንዴት ነዎት"]
    refs = ["a cat sat on the mat", "ጤና ይስጥልኝ እንዴት ነህ"]
    r = compute_rouge(preds, refs)
    print(r.as_dict())
    print("weighted (proxy):", weighted_score(r.rouge1_f1, r.rougeL_f1))
