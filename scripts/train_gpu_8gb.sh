#!/usr/bin/env bash
# =============================================================================
# Full local fine-tuning run for an 8GB NVIDIA GPU (e.g. RTX 2070/3060/4060).
#
# Strategy: mt5-base + LoRA (parameter-efficient fine-tuning). Full fine-tuning
# of mt5-base needs ~9-10GB just for optimizer state and would OOM at 8GB, so we
# freeze the base model and train low-rank adapters instead. This trains every
# language jointly via task-prefixed inputs (see src/data.py) and selects
# checkpoints on the leaderboard-mirroring ROUGE proxy (src/metrics.py).
#
# This script is SELF-BOOTSTRAPPING: it creates a venv, installs the pinned
# deps + a CUDA build of PyTorch, trains, then scores on the Val split.
#
# Usage:
#   bash scripts/train_gpu_8gb.sh                 # mt5-base + LoRA, full data
#   MODEL=google/mt5-small bash scripts/train_gpu_8gb.sh   # smaller/faster
#   EPOCHS=4 TRAIN_BS=2 bash scripts/train_gpu_8gb.sh      # override anything
#   SMOKE=1 bash scripts/train_gpu_8gb.sh         # 2k-row dry run to test setup
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# ---- tunables (override via env) --------------------------------------------
MODEL="${MODEL:-google/mt5-base}"
OUT="${OUT:-outputs/mt5base_lora_8gb}"
EPOCHS="${EPOCHS:-3}"
TRAIN_BS="${TRAIN_BS:-4}"          # per-device train batch; lower to 2 if you OOM
EVAL_BS="${EVAL_BS:-4}"            # generation eval is memory-heavy; keep small
INFER_BS="${INFER_BS:-32}"          # standalone inference doesn't keep grad states; safe to use larger batch size
GRAD_ACCUM="${GRAD_ACCUM:-4}"      # effective batch = TRAIN_BS * GRAD_ACCUM = 16
LR="${LR:-3e-4}"                   # higher LR is normal for LoRA
MAX_INPUT_LEN="${MAX_INPUT_LEN:-128}"
MAX_TARGET_LEN="${MAX_TARGET_LEN:-256}"
NUM_BEAMS="${NUM_BEAMS:-4}"
EVAL_SUBSAMPLE="${EVAL_SUBSAMPLE:-800}"  # score on 800 random Val rows during training (speed)
VENV="${VENV:-.venv-gpu}"
PYBIN="${PYBIN:-python3}"

# ---- 1) environment ---------------------------------------------------------
if [ ! -d "$VENV" ]; then
  echo "== [setup] creating virtualenv at $VENV =="
  "$PYBIN" -m venv "$VENV"
fi
# shellcheck disable=SC1091
if [ -f "$VENV/Scripts/activate" ]; then
  source "$VENV/Scripts/activate"
else
  source "$VENV/bin/activate"
fi
python -m pip install --upgrade pip wheel >/dev/null

echo "== [setup] installing PyTorch (CUDA) + pinned requirements =="
# Default PyPI torch wheels on Linux x86_64 ship bundled CUDA — no system CUDA needed.
# python -c "import torch" 2>/dev/null || pip install "torch==2.4.0"
# pip install -q -r requirements.txt

export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# Reduce CUDA fragmentation OOMs on small cards.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- 2) sanity check the GPU and pick a precision flag ----------------------
echo "== [check] GPU / precision =="
PREC_FLAG="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[check] No CUDA GPU visible to PyTorch. "
                     "Install the NVIDIA driver, then re-run on the GPU machine.")
name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
# bf16 needs Ampere+ (RTX 30xx/40xx). Older cards (RTX 20xx) fall back to fp16.
flag = "--bf16" if torch.cuda.is_bf16_supported() else "--fp16"
import sys
print(f"[check] {name}  {vram:.1f} GB VRAM  -> using {flag}", file=sys.stderr)
print(flag)
PY
)"

# ---- 3) optional smoke flags ------------------------------------------------
SMOKE_ARGS=()
if [ "${SMOKE:-0}" = "1" ]; then
  echo "== [smoke] 2k-row / 1-epoch dry run to validate the full pipeline =="
  SMOKE_ARGS=(--max_train_samples 2000 --epochs 1 --eval_subsample 200)
  OUT="outputs/smoke_${MODEL##*/}_lora"
fi

# ---- 4) train ---------------------------------------------------------------
echo "== [train] $MODEL + LoRA -> $OUT =="
python -m src.train \
  --model_name "$MODEL" \
  --output_dir "$OUT" \
  --use_lora --gradient_checkpointing \
  --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --epochs "$EPOCHS" \
  --train_bs "$TRAIN_BS" --eval_bs "$EVAL_BS" --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --max_input_len "$MAX_INPUT_LEN" --max_target_len "$MAX_TARGET_LEN" \
  --num_beams "$NUM_BEAMS" --gen_max_len "$MAX_TARGET_LEN" \
  --eval_subsample "$EVAL_SUBSAMPLE" \
  "$PREC_FLAG" \
  "${SMOKE_ARGS[@]}"

# ---- 5) score on the full Val split (per-language ROUGE) --------------------
echo "== [eval] full Val ROUGE + Amharic script guard =="
python -m src.infer \
  --model_dir "$OUT" --split val \
  --batch_size "$INFER_BS" --num_beams "$NUM_BEAMS" \
  --gen_max_len "$MAX_TARGET_LEN" --tag mt5base_lora

echo ""
echo "== Done. =="
echo "Model + adapters: $OUT"
echo "Val predictions:  outputs/mt5base_lora_val_predictions.csv"
echo "To build a Zindi submission from the Test split, run:"
echo "  source $VENV/bin/activate && python -m src.infer --model_dir $OUT --split test --tag mt5base_lora"
