"""B0 baseline: TF-IDF character n-gram nearest-neighbour retrieval (GPU-free).

For each question we retrieve the most similar *training* question (per subset, with a
global fallback) and return its answer verbatim. Because ~74% of the leaderboard is pure
lexical overlap and many training answers are templated/duplicated (see EDA), copying a
real reference answer in the correct language is a strong, zero-GPU anchor — the floor
that any fine-tuned seq2seq model must beat.

This module is fully runnable on CPU. It produces (a) Val ROUGE so we can rank it on our
offline north-star, and (b) Test predictions for manual review before any submission.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from . import data as D
from .metrics import compute_rouge, compute_rouge_by_group


class TfidfRetriever:
    """Per-subset TF-IDF char-n-gram retriever with a global fallback model."""

    def __init__(self, group_col="subset", ngram_range=(3, 5), max_features=200_000):
        self.group_col = group_col
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.models: dict[str, dict] = {}
        self.global_model: dict | None = None

    def _fit_single(self, questions, answers):
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=self.ngram_range,
            min_df=1,
            max_features=self.max_features,
            lowercase=False,  # preserve case + non-Latin scripts
        )
        X = vec.fit_transform(questions)
        nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(X)
        return {"vec": vec, "nn": nn, "answers": np.array(answers, dtype=object)}

    def fit(self, df: pd.DataFrame):
        q = df.input.fillna("").astype(str).tolist()
        a = df.output.fillna("").astype(str).tolist()
        self.global_model = self._fit_single(q, a)
        for g, sub in df.groupby(self.group_col):
            if len(sub) >= 2:
                self.models[g] = self._fit_single(
                    sub.input.fillna("").astype(str).tolist(),
                    sub.output.fillna("").astype(str).tolist(),
                )
        print(f"[tfidf] fitted global + {len(self.models)} subset models")
        return self

    def predict(self, df: pd.DataFrame):
        preds, sims = [], []
        for _, row in df.iterrows():
            m = self.models.get(row[self.group_col], self.global_model)
            Xq = m["vec"].transform([str(row.input)])
            dist, idx = m["nn"].kneighbors(Xq, n_neighbors=1)
            preds.append(m["answers"][idx[0][0]])
            sims.append(1.0 - float(dist[0][0]))
        return preds, sims


def run(data_dir: str = D.DATA_DIR_DEFAULT, out_dir: str = "outputs"):
    """Fit on train, evaluate on val, generate test predictions. Returns a result dict."""
    train = D.load_split("train", data_dir, clean=True, prefix=False)
    val = D.load_split("val", data_dir, clean=True, prefix=False)
    test = D.load_split("test", data_dir, clean=True, prefix=False)

    # --- validate on val ---
    r = TfidfRetriever().fit(train)
    val_pred, _ = r.predict(val)
    overall = compute_rouge(val_pred, val.output.tolist())
    by_lang = compute_rouge_by_group(val_pred, val.output.tolist(), val.subset.tolist())
    print(f"[tfidf] VAL  R1={overall.rouge1_f1:.4f}  RL={overall.rougeL_f1:.4f}  proxy={overall.proxy:.4f}")

    # --- test predictions (refit on train; train has no test leakage) ---
    test_pred, test_sim = r.predict(test)

    os.makedirs(out_dir, exist_ok=True)
    val_df = val[["ID", "subset"]].copy()
    val_df["prediction"] = val_pred
    val_df["reference"] = val.output.values
    val_df.to_csv(os.path.join(out_dir, "tfidf_val_predictions.csv"), index=False)

    test_df = test[["ID", "subset"]].copy()
    test_df["prediction"] = test_pred
    test_df["similarity"] = test_sim
    test_df.to_csv(os.path.join(out_dir, "tfidf_test_predictions.csv"), index=False)
    print(f"[tfidf] wrote val + test predictions to {out_dir}/ for manual review")

    return {"overall": overall.as_dict(), "by_lang": by_lang, "test_df": test_df}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=D.DATA_DIR_DEFAULT)
    ap.add_argument("--out_dir", default="outputs")
    args = ap.parse_args()
    res = run(args.data_dir, args.out_dir)
    print("\nPer-language ROUGE:")
    print(pd.DataFrame(res["by_lang"]).T.round(4).to_string())
