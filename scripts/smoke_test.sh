#!/usr/bin/env bash
# Local CPU smoke test: validates train -> infer -> score end-to-end on a tiny subset.
# NOT a real model — it proves the pipeline runs locally before committing to a cloud GPU run.
# Expect ~25 min on a modern CPU (training is the slow part). Usage: bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-/usr/bin/python3}
OUT=outputs/smoke_mt5small
# First run downloads google/mt5-small (~1.2 GB); afterwards everything is offline.
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
$PY -c "import os; from huggingface_hub import snapshot_download; snapshot_download('google/mt5-small')" \
  && export HF_HUB_OFFLINE=1

echo "== [1/2] train (mt5-small, ~2k stratified rows, 1 epoch, CPU) =="
$PY -m src.train \
  --model_name google/mt5-small \
  --output_dir "$OUT" \
  --max_train_samples 2000 \
  --epochs 1 --train_bs 4 --eval_bs 8 --grad_accum 1 \
  --lr 5e-4 --max_input_len 128 --max_target_len 128 \
  --num_beams 1 --gen_max_len 128 --eval_subsample 200

echo "== [2/2] eval on a small Val slice (ROUGE + Amharic script guard) =="
$PY -m src.infer --model_dir "$OUT" --split val --num_beams 1 \
  --gen_max_len 128 --tag smoke --limit 200

echo ""
echo "== Pipeline works end-to-end locally. =="
echo "Scores will be LOW (undertrained on purpose). For real models -> notebooks/colab_run.ipynb on a cloud GPU."
