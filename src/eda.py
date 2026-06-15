"""Exploratory data analysis -> figures + findings for Multilingual Health QA.

Generates the 8 figures referenced in the report (reports/figures/) and prints a findings
summary. Pure-CPU, matplotlib-only. The notebook notebooks/01_eda.ipynb narrates the same
analysis; this module exists so figures are reproducible from one command and stay in sync.

    python -m src.eda            # -> reports/figures/*.png + console findings
"""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import data as D
from .preprocess import geez_ratio, latin_ratio

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "figures")
SUBSET_ORDER = ["Eng_Uga", "Aka_Gha", "Eng_Gha", "Eng_Eth",
                "Lug_Uga", "Eng_Ken", "Swa_Ken", "Amh_Eth"]


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {path}")


def _wc(s):
    return s.fillna("").str.split().map(len)


def fig_subset_distribution(train, test):
    tr = train.subset.value_counts().reindex(SUBSET_ORDER)
    te = test.subset.value_counts().reindex(SUBSET_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(SUBSET_ORDER))
    w = 0.4
    ax.bar(x - w / 2, tr.values, w, label="Train", color="#4C72B0")
    ax.bar(x + w / 2, te.values * (tr.sum() / te.sum()), w,
           label="Test (scaled to train total)", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(SUBSET_ORDER, rotation=30, ha="right")
    ax.set_ylabel("rows"); ax.set_title("Subset distribution — Train vs Test (test mirrors train)")
    ax.legend()
    _save(fig, "01_subset_distribution.png")


def fig_length_distributions(train):
    iw, ow = _wc(train.input), _wc(train.output)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(iw, bins=50, color="#55A868"); axes[0].set_title("Input length (words)")
    axes[0].axvline(iw.median(), color="k", ls="--", label=f"median {iw.median():.0f}")
    axes[0].legend()
    axes[1].hist(ow.clip(upper=300), bins=50, color="#C44E52")
    axes[1].set_title("Output length (words, clipped@300)")
    axes[1].axvline(ow.median(), color="k", ls="--", label=f"median {ow.median():.0f}")
    axes[1].legend()
    for a in axes: a.set_ylabel("count"); a.set_xlabel("words")
    _save(fig, "02_length_distributions.png")


def fig_output_length_by_subset(train):
    data = [_wc(train[train.subset == s].output).clip(upper=300).values for s in SUBSET_ORDER]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot(data, labels=SUBSET_ORDER, showfliers=False)
    ax.set_xticklabels(SUBSET_ORDER, rotation=30, ha="right")
    ax.set_ylabel("output words"); ax.set_title("Answer length by subset (decides gen max_length)")
    _save(fig, "03_output_length_by_subset.png")


def fig_script_composition(train):
    rows = []
    for s in SUBSET_ORDER:
        sub = train[train.subset == s].output
        rows.append({"subset": s,
                     "geez": sub.map(geez_ratio).mean(),
                     "latin": sub.map(latin_ratio).mean()})
    d = pd.DataFrame(rows).set_index("subset").reindex(SUBSET_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(SUBSET_ORDER)); w = 0.4
    ax.bar(x - w / 2, d.latin, w, label="Latin char ratio", color="#4C72B0")
    ax.bar(x + w / 2, d.geez, w, label="Ge'ez char ratio", color="#8172B3")
    ax.set_xticks(x); ax.set_xticklabels(SUBSET_ORDER, rotation=30, ha="right")
    ax.set_ylabel("mean ratio of answer chars")
    ax.set_title("Script composition of answers (Amharic = Ge'ez; rest = Latin)")
    ax.legend()
    _save(fig, "04_script_composition.png")
    return d


def fig_duplicate_outputs(train):
    vc = train.output.value_counts()
    dup = vc[vc > 1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(vc.values, bins=range(1, 25), color="#937860", align="left")
    axes[0].set_title("Answer reuse (how many times each unique answer appears)")
    axes[0].set_xlabel("occurrences"); axes[0].set_ylabel("# unique answers")
    top = dup.head(10)[::-1]
    axes[1].barh([f"{t[:30]}…" for t in top.index], top.values, color="#DA8BC3")
    axes[1].set_title("Top-10 most repeated answers"); axes[1].set_xlabel("count")
    _save(fig, "05_duplicate_outputs.png")
    return {"unique_outputs": int(vc.size), "rows": int(len(train)),
            "dup_output_rows": int(len(train) - vc.size),
            "share_dup": float(1 - vc.size / len(train))}


def fig_qa_vocab_overlap(train, n=4000):
    s = train.sample(min(n, len(train)), random_state=42)
    ov = []
    for q, a in zip(s.input, s.output):
        qt, at = set(str(q).lower().split()), set(str(a).lower().split())
        ov.append(len(qt & at) / len(qt) if qt else 0.0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ov, bins=30, color="#4C72B0")
    ax.set_title("Question-word recall in the answer (lexical reuse)")
    ax.set_xlabel("fraction of question words appearing in answer"); ax.set_ylabel("count")
    _save(fig, "06_qa_vocab_overlap.png")
    return {"mean_q_recall_in_answer": float(np.mean(ov))}


def fig_baseline_by_language():
    """TF-IDF B0 per-language ROUGE — if the baseline has been run."""
    from .baseline_tfidf import run

    res = run()
    bl = pd.DataFrame(res["by_lang"]).T.drop(index="ALL")
    bl = bl.reindex([s for s in SUBSET_ORDER if s in bl.index])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(bl)); w = 0.4
    ax.bar(x - w / 2, bl.rouge1_f1, w, label="ROUGE-1 F1", color="#55A868")
    ax.bar(x + w / 2, bl.rougeL_f1, w, label="ROUGE-L F1", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(bl.index, rotation=30, ha="right")
    ax.set_ylabel("F1"); ax.set_title("TF-IDF B0 baseline — Val ROUGE by subset (where we win/lose)")
    ax.legend()
    _save(fig, "07_baseline_rouge_by_language.png")
    return res["overall"]


def fig_text_length_tokens_note(train):
    """Words->tokens rule-of-thumb scatter to justify truncation lengths."""
    iw, ow = _wc(train.input), _wc(train.output)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(iw, ow, s=4, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("input words"); ax.set_ylabel("output words")
    ax.set_title("Input vs output length (no strong correlation)")
    ax.set_xlim(0, iw.quantile(0.99)); ax.set_ylim(0, ow.quantile(0.99))
    _save(fig, "08_input_vs_output_length.png")


def main():
    train = D.load_split("train", clean=True, prefix=False)
    val = D.load_split("val", clean=True, prefix=False)
    test = D.load_split("test", clean=True, prefix=False)

    findings = {}
    findings["shapes"] = {"train": len(train), "val": len(val), "test": len(test)}
    findings["leakage"] = D.leakage_report(train, val, test)
    findings["input_words"] = {
        "median": float(_wc(train.input).median()),
        "p90": float(_wc(train.input).quantile(0.9)),
        "max": int(_wc(train.input).max()),
    }
    findings["output_words"] = {
        "median": float(_wc(train.output).median()),
        "p90": float(_wc(train.output).quantile(0.9)),
        "max": int(_wc(train.output).max()),
    }

    fig_subset_distribution(train, test)
    fig_length_distributions(train)
    fig_output_length_by_subset(train)
    script_df = fig_script_composition(train)
    findings["script"] = script_df.round(3).to_dict("index")
    findings["dups"] = fig_duplicate_outputs(train)
    findings.update(fig_qa_vocab_overlap(train))
    fig_input_vs_output = fig_text_length_tokens_note(train)
    findings["baseline_tfidf_val"] = fig_baseline_by_language()

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(os.path.dirname(FIG_DIR), "eda_findings.json"), "w") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)
    print("\n[eda] findings:\n", json.dumps(findings, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
