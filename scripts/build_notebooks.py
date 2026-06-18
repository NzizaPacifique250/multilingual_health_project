"""Generate notebooks/01_eda.ipynb and notebooks/colab_run.ipynb as nbformat-4 JSON.

Kept as a script (not run on Colab) so the notebooks are reproducible artifacts and stay
in sync with the src/ modules they call into. Run: python scripts/build_notebooks.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write(name, cells):
    path = os.path.join(NB_DIR, name)
    os.makedirs(NB_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(notebook(cells), f, indent=1, ensure_ascii=False)
    print("wrote", path)


# =====================================================================================
# 01_eda.ipynb
# =====================================================================================
eda = [
    md("""# 01 — Exploratory Data Analysis & Preprocessing
**Multilingual Health QA (HASH / Zindi)** · maps to Rubric criterion 3 (Data/EDA, 10 pts).

This notebook profiles the corpus and justifies every preprocessing decision. All heavy
lifting lives in `src/` (imported below) so the analysis is reproducible from one command
(`python -m src.eda`) and never drifts from the training/inference code.

**Guiding insight (see `PLAN.md`):** the leaderboard is `0.37·ROUGE-1 + 0.37·ROUGE-L +
0.26·LLM-judge` — **74% is pure lexical overlap**. So EDA centers on the two levers that
move lexical overlap: *(1) emitting the correct language & script*, and *(2) reusing the
reference vocabulary/length*.
"""),
    code("""# Colab setup — clone repo + install. On local run, skip the clone.
import sys, os
if 'google.colab' in sys.modules:
    !git clone -q https://github.com/<your-user>/multilingual_health_qa.git
    %cd multilingual_health_qa
    !pip install -q rouge-score scikit-learn matplotlib
# Ensure repo root on path
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, os.getcwd())
print('cwd:', os.getcwd())"""),
    code("""import pandas as pd, numpy as np
from src import data as D
from src.preprocess import geez_ratio, latin_ratio, normalize_text

train = D.load_split('train', clean=True, prefix=False)
val   = D.load_split('val',   clean=True, prefix=False)
test  = D.load_split('test',  clean=True, prefix=False)
print(train.shape, val.shape, test.shape)
train.head(2)"""),
    md("""## 1. Splits, columns, integrity
We verify row counts, nulls, and **leakage** across splits (a contaminated split would
make our offline ROUGE optimistic)."""),
    code("""print('nulls (train):', train.isna().sum().to_dict())
print('leakage report:', D.leakage_report(train, val, test))"""),
    md("""**Finding.** Train 29,815 / Val 6,686 / Test 2,618. No nulls. **Zero exact-input
overlap** train↔val and train↔test → our Val ROUGE is an honest proxy for the leaderboard.
Note though: **1,482 duplicate inputs** and **11,750 duplicate outputs (~39%)** *within*
train — answers are heavily templated (analyzed in §4)."""),
    md("""## 2. Subset / language distribution
8 subsets of the form `<Lang>_<Country>`. The test set mirrors train proportions, so a
single jointly-trained model is the right structure — but low-resource languages
(Amharic, Swahili, Luganda, Akan) are small and need explicit attention."""),
    code("""disp = pd.DataFrame({'train': train.subset.value_counts(),
                     'test':  test.subset.value_counts()})
disp['train_%'] = (disp.train / disp.train.sum() * 100).round(1)
disp['test_%']  = (disp.test  / disp.test.sum()  * 100).round(1)
disp.loc[:, ['language','script']] = train.groupby('subset')[['language','script']].first()
disp"""),
    md("![subset distribution](../reports/figures/01_subset_distribution.png)\n\n"
       "**Finding.** ~61% of train is English. The four target African languages total ~39%; "
       "**Amharic is the smallest (1,845 rows) and the only Ge'ez-script language** — our hardest case."),
    md("""## 3. Length distributions → truncation budgets
Decides `max_input_len` and generation `max_length` (a direct ROUGE lever — truncated
answers lose recall)."""),
    code("""for col in ['input','output']:
    w = train[col].fillna('').str.split().map(len)
    print(f'{col:7s} words  median={w.median():.0f}  p90={w.quantile(.9):.0f}  p99={w.quantile(.99):.0f}  max={w.max()}')"""),
    md("![lengths](../reports/figures/02_length_distributions.png)\n"
       "![lengths by subset](../reports/figures/03_output_length_by_subset.png)\n\n"
       "**Decision.** Inputs: median 13 words, max 83 → **`max_input_len=128` tokens** covers ~all. "
       "Outputs: median 61, p90 153 words → **generation `max_length=256` tokens** (≈ p90) balances "
       "recall vs. runaway repetition. We log this choice as a tunable in Phase 3."),
    md("""## 4. Script handling (Ge'ez vs Latin) + duplication
The single most important multilingual check: Amharic answers must be **Ge'ez script**.
A model that emits Latin/English for Amharic scores ~0 on ROUGE there."""),
    code("""rows = []
for s in D.SUBSETS:
    o = train[train.subset==s].output
    rows.append({'subset': s, 'geez': o.map(geez_ratio).mean().round(3),
                 'latin': o.map(latin_ratio).mean().round(3)})
pd.DataFrame(rows)"""),
    md("![script composition](../reports/figures/04_script_composition.png)\n\n"
       "**Finding.** Amharic answers are **96.6% Ge'ez**; every other subset is ~100% Latin. "
       "→ We (a) keep Unicode **NFC** normalization so Ge'ez sequences tokenize canonically, "
       "(b) pick a tokenizer with Ge'ez coverage (mT5/NLLB do), and (c) add a *wrong-language "
       "guard* at inference that flags Amharic predictions which aren't Ge'ez (`src/infer.py`)."),
    code("""vc = train.output.value_counts()
print(f'unique answers: {vc.size:,} / {len(train):,} rows  ->  {1-vc.size/len(train):.1%} are duplicates')
vc.head(5)"""),
    md("![duplicate outputs](../reports/figures/05_duplicate_outputs.png)\n\n"
       "**Finding & implication.** ~39% of answers are repeated templates. This is *why a "
       "retrieval baseline is strong* (copying a real template often matches the reference) and "
       "*why we must dedup carefully*: we keep duplicates in training (they reflect the true "
       "answer distribution the metric rewards) but watch for them inflating apparent val scores."),
    md("""## 5. Question→answer lexical reuse
How much of the answer vocabulary is already in the question? Informs whether copying /
extraction helps."""),
    code("""s = train.sample(4000, random_state=42)
rec = [len(set(str(q).lower().split()) & set(str(a).lower().split())) / max(1,len(set(str(q).lower().split())))
       for q,a in zip(s.input, s.output)]
print('mean question-word recall in answer:', round(float(np.mean(rec)),3))"""),
    md("![qa overlap](../reports/figures/06_qa_vocab_overlap.png)\n"
       "![input vs output](../reports/figures/08_input_vs_output_length.png)\n\n"
       "**Finding.** ~40% of question words reappear in the answer, but input/output lengths are "
       "uncorrelated — answers are elaborations, not echoes. Pure extraction won't suffice; we need "
       "generation conditioned on the question + language tag."),
    md("""## 6. Preprocessing decisions (summary)
| Decision | Choice | Why |
|---|---|---|
| Unicode | **NFC** | canonical Ge'ez/Latin → stable tokenization & ROUGE counts |
| Whitespace | collapse + strip | removes scrape artifacts that fragment tokens |
| Case / punctuation | **keep** | ROUGE scores refs as-written; Ge'ez has no case |
| Invisible/control chars | strip | zero-width & BOM inflate token counts |
| Task prefix | `"<subset>: <q>"` | steers language + script with one shared model |
| `max_input_len` | 128 | covers max 83-word inputs |
| gen `max_length` | 256 | ≈ p90 output length |
All implemented in `src/preprocess.py` + `src/data.py`."""),
    md("""## 7. Baseline anchor (B0) — TF-IDF retrieval
A GPU-free floor: for each question, return the nearest training answer (per subset).
Scored with our leaderboard-mirroring ROUGE (`src/metrics.py`)."""),
    code("""from src.baseline_tfidf import run
res = run()
pd.DataFrame(res['by_lang']).T.round(4)"""),
    md("![baseline rouge](../reports/figures/07_baseline_rouge_by_language.png)\n\n"
       "**B0 result: Val proxy ≈ 0.394** (R1 0.421 / RL 0.366). Per-language: easy = Swahili/Eng-Kenya "
       "(~0.58), **hard = Amharic (0.14)** and Ghanaian subsets (Akan/Eng-Gha ~0.22). This sets the bar "
       "the fine-tuned seq2seq model must beat, and tells us *where* to spend effort (Amharic, Akan)."),
    md("""## Next
→ `notebooks/colab_run.ipynb` fine-tunes mT5/NLLB end-to-end on a GPU and logs results to
`experiments/`. EDA findings above drive its config (prefix, lengths, low-resource upsampling)."""),
]
write("01_eda.ipynb", eda)


# =====================================================================================
# colab_run.ipynb
# =====================================================================================
colab = [
    md("""# Colab — End-to-End Fine-Tune & Predict
**Multilingual Health QA (HASH / Zindi).** Reproducible run: install → train seq2seq →
evaluate on Val (leaderboard-mirroring ROUGE) → generate a **reviewable** Test prediction
CSV. Maps to rubric crit. 1, 2, 9 (leaderboard, tracking, reproducibility).

> ⚠️ This notebook does **not** upload anything. It writes a local submission CSV for you
> to review and submit manually."""),
    md("## 0. Runtime check (use a GPU runtime: Runtime → Change runtime type → T4/A100)"),
    code("""import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),
          f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"""),
    md("## 1. Get the code + data"),
    code("""import sys, os
if 'google.colab' in sys.modules:
    !git clone -q https://github.com/<your-user>/multilingual_health_qa.git
    %cd multilingual_health_qa
    !pip install -q -r requirements.txt
sys.path.insert(0, os.getcwd())
# Data must be in ./datasets (Train.csv, Val.csv, Test.csv, SampleSubmission.csv).
# On Colab, upload them or mount Drive if not tracked in the repo.
assert os.path.exists('datasets/Train.csv'), 'place data CSVs in datasets/'"""),
    md("""## 2. Train
Model ladder by VRAM: `google/mt5-small` (fast) → `google/mt5-base` (default) →
`facebook/nllb-200-distilled-600M` (best African coverage). One variable at a time — log
each run in `experiments/`."""),
    code("""# ~2-4h for mt5-base/3 epochs on a T4. Drop to mt5-small or fewer epochs to iterate fast.
!python -m src.train \\
    --model_name google/mt5-base \\
    --output_dir outputs/mt5base_v1 \\
    --epochs 3 --train_bs 8 --eval_bs 16 --grad_accum 2 \\
    --lr 3e-4 --max_input_len 128 --max_target_len 256 \\
    --num_beams 4 --gen_max_len 256 --no_repeat_ngram 3 \\
    --eval_subsample 1500 --fp16"""),
    md("## 3. Evaluate on Val (full per-language ROUGE)"),
    code("""!python -m src.infer --model_dir outputs/mt5base_v1 --split val \\
    --num_beams 4 --gen_max_len 256 --no_repeat_ngram 3 --tag mt5base_v1"""),
    md("## 4. Generate Test predictions → reviewable submission CSV (no upload)"),
    code("""!python -m src.infer --model_dir outputs/mt5base_v1 --split test \\
    --num_beams 4 --gen_max_len 256 --no_repeat_ngram 3 --tag mt5base_v1"""),
    code("""import pandas as pd
sub = pd.read_csv('outputs/submission_mt5base_v1.csv')
print(sub.shape); sub.head()  # REVIEW before uploading to Zindi"""),
    md("""## 5. Manual review checklist (do this before submitting)
- [ ] Spot-check Amharic rows are **Ge'ez script** (infer.py prints a wrong-language count).
- [ ] No empty / single-token answers; lengths look reasonable per language.
- [ ] Val proxy **beats the TF-IDF B0 (0.394)** and prior runs (`experiments/results.csv`).
- [ ] IDs match `SampleSubmission.csv` (submit.py validates this).
Then upload `outputs/submission_*.csv` to Zindi yourself and record the LB score in the log."""),
    md("""## 6. Log the run
Append a row to `experiments/results.csv` and a note to `experiments/log.md`:
config (model, epochs, lr, decoding), Val R1/RL/proxy, LB score, and one-line takeaway."""),
]
write("colab_run.ipynb", colab)
print("done")
