"""Multilingual Health QA (HASH / Zindi) — source package.

Modules:
    data        load/clean/split CSVs, language tagging, task-prefixing
    preprocess  Unicode/script normalization for multilingual + Ge'ez text
    metrics     ROUGE-1/L replication mirroring the leaderboard + weighted proxy
    train       seq2seq fine-tuning (HF Trainer)
    infer       batch generation -> predictions
    submit      enforce the 4-column Zindi submission format
"""
