"""Seq2seq fine-tuning for Multilingual Health QA (HF Trainer).

Trains a multilingual encoder-decoder (default ``google/mt5-base``; swap to
``facebook/nllb-200-distilled-600M`` or ``google/mt5-small`` via ``--model_name``) on all
subsets jointly, using task-prefixed inputs ``"<subset>: <question>"`` so one model serves
every language. Designed to run on a single Colab GPU.

Eval during training uses our leaderboard-mirroring ROUGE (src.metrics) so checkpoints are
selected on the same lexical signal the leaderboard rewards (PLAN.md Phase 2/3).

Usage (Colab):
    python -m src.train --model_name google/mt5-base --output_dir outputs/mt5base_v1 \
        --epochs 3 --train_bs 8 --eval_bs 16 --lr 3e-4 --fp16
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# We train with PyTorch only. Stop transformers from importing TensorFlow/Flax — on Colab
# their pre-installed TF clashes with our pinned protobuf (3.20.3, needed by the mT5
# sentencepiece tokenizer) and raises "cannot import name 'runtime_version'".
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np

# ---- Python 3.14 compatibility shim for datasets/dill fingerprinting ----
# On Python 3.14, pickle._Pickler._batch_setitems gained an `obj` parameter and datasets'
# own dict-key-sorting patch breaks; we restore deterministic fingerprinting there. On
# earlier Pythons (e.g. Colab's 3.12) _batch_setitems takes only (self, items) and datasets
# works natively — applying this patch would pass an extra arg and raise TypeError, so skip.
if sys.version_info >= (3, 14):
    import pickle
    import dill
    import datasets.utils._dill

    dill.Pickler._batch_setitems = lambda self, items, obj=None: pickle._Pickler._batch_setitems(self, items, obj)

    def patched_datasets_setitems(self, items, obj=None):
        if getattr(self, "_legacy_no_dict_keys_sorting", False):
            return pickle._Pickler._batch_setitems(self, items, obj)
        try:
            sorted_items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            sorted_items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        return pickle._Pickler._batch_setitems(self, sorted_items, obj)

    datasets.utils._dill.Pickler._batch_setitems = patched_datasets_setitems
# ------------------------------------------------------------------

from . import data as D
from .metrics import compute_rouge


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="google/mt5-base")
    p.add_argument("--data_dir", default=D.DATA_DIR_DEFAULT)
    p.add_argument("--output_dir", default="outputs/seq2seq_run")
    p.add_argument("--max_input_len", type=int, default=128)   # input p~max ~83 words
    p.add_argument("--max_target_len", type=int, default=256)  # output p90 153 words
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--train_bs", type=int, default=8)
    p.add_argument("--eval_bs", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--gen_max_len", type=int, default=256)
    p.add_argument("--gen_min_len", type=int, default=8)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--upsample_low_resource", type=float, default=1.0,
                   help="repeat factor for LOW_RESOURCE subsets (1.0 = off)")
    p.add_argument("--eval_subsample", type=int, default=0,
                   help="if >0, evaluate generation on a random N-row val subset (speed)")
    p.add_argument("--max_train_samples", type=int, default=0,
                   help="if >0, train on a random N-row subset (CPU smoke test)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="trade compute for memory (recommended on <=8GB GPUs)")
    # ---- LoRA / PEFT (parameter-efficient fine-tuning for small GPUs) ----
    p.add_argument("--use_lora", action="store_true",
                   help="freeze the base model and train low-rank adapters only "
                        "(fits mt5-base full fine-tuning into ~8GB VRAM)")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", default="q,k,v,o,wi_0,wi_1,wo",
                   help="comma-separated module names to adapt (mT5/T5 attention+FFN)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = build_args()
    # Heavy imports kept inside main so the module imports cheaply for tooling/tests.
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    on_gpu = torch.cuda.is_available()
    print(f"[train] model={args.model_name} device={'cuda' if on_gpu else 'cpu'}")
    if not on_gpu and (args.fp16 or args.bf16):
        # Mixed precision needs a GPU; silently fall back to fp32 on CPU.
        print("[train] CPU detected -> disabling fp16/bf16 (fp32 training).")
        args.fp16 = args.bf16 = False

    # ---- data ----
    train_df = D.load_split("train", args.data_dir, clean=True, prefix=True)
    val_df = D.load_split("val", args.data_dir, clean=True, prefix=True)

    if args.upsample_low_resource > 1.0:
        import pandas as pd

        extra = train_df[train_df.subset.isin(D.LOW_RESOURCE)]
        reps = int(args.upsample_low_resource) - 1
        train_df = pd.concat([train_df] + [extra] * reps, ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        print(f"[train] upsampled low-resource x{args.upsample_low_resource} -> {len(train_df)} rows")

    if args.max_train_samples and args.max_train_samples < len(train_df):
        # Stratify by subset so a smoke test still touches every language.
        train_df = (
            train_df.groupby("subset", group_keys=True)
            .apply(lambda g: g.sample(
                max(1, round(args.max_train_samples * len(g) / len(train_df))),
                random_state=args.seed))
            .reset_index(level="subset")
            .reset_index(drop=True)
        )
        print(f"[train] smoke subset -> {len(train_df)} train rows across {train_df.subset.nunique()} subsets")

    eval_df = val_df
    if args.eval_subsample and args.eval_subsample < len(val_df):
        eval_df = val_df.sample(args.eval_subsample, random_state=args.seed).reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    if args.use_lora:
        # Wrap the frozen base model with LoRA adapters. Only the adapters (a few M
        # params) get gradients/optimizer state, so mt5-base trains in ~8GB VRAM.
        from peft import LoraConfig, TaskType, get_peft_model

        if args.gradient_checkpointing:
            # PEFT + checkpointing: inputs must require grad or the adapter gets no signal.
            model.enable_input_require_grads()
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    def tokenize(batch):
        model_inputs = tok(
            batch["src"], max_length=args.max_input_len, truncation=True
        )
        labels = tok(
            text_target=batch["output"], max_length=args.max_target_len, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    keep = ["src", "output"]
    train_ds = Dataset.from_pandas(train_df[keep]).map(
        tokenize, batched=True, remove_columns=keep, desc="tok-train"
    )
    eval_ds = Dataset.from_pandas(eval_df[["src", "output"]]).map(
        tokenize, batched=True, remove_columns=["src", "output"], desc="tok-eval"
    )

    collator = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tok.pad_token_id)
        labels = np.where(labels != -100, labels, tok.pad_token_id)
        dec_preds = tok.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tok.batch_decode(labels, skip_special_tokens=True)
        r = compute_rouge(dec_preds, dec_labels)
        return {"rouge1_f1": r.rouge1_f1, "rougeL_f1": r.rougeL_f1, "proxy": r.proxy}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # mT5 shares/transposes weights -> safetensors refuses non-contiguous tensors.
        # Save as pytorch_model.bin instead (works for mT5 and NLLB alike).
        save_safetensors=False,
        predict_with_generate=True,
        generation_max_length=args.gen_max_len,
        generation_num_beams=args.num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="proxy",
        greater_is_better=True,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        # Re-entrant checkpointing warns/errors with frozen-base PEFT; use the new path.
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        report_to="none",
        seed=args.seed,
    )
    # Decoding knobs that aren't direct TrainingArguments fields.
    model.generation_config.no_repeat_ngram_size = args.no_repeat_ngram
    model.generation_config.length_penalty = args.length_penalty
    model.generation_config.min_length = args.gen_min_len

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("[train] final val metrics:", metrics)

    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2, ensure_ascii=False)
    print(f"[train] saved model + metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
