"""Batch generation for a fine-tuned seq2seq model -> prediction CSVs.

Generates answers for a split (val for scoring, or test for a candidate submission). For
``test`` it writes BOTH a raw predictions CSV and a contract-valid 4-column file via
src.submit — but it NEVER uploads. Review the output before submitting manually.

Usage (Colab):
    python -m src.infer --model_dir outputs/mt5base_v1 --split val   # -> ROUGE report
    python -m src.infer --model_dir outputs/mt5base_v1 --split test  # -> submission CSV
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

from . import data as D
from .metrics import compute_rouge, compute_rouge_by_group
from .preprocess import looks_amharic, normalize_text
from .submit import build_submission, save_submission


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--data_dir", default=D.DATA_DIR_DEFAULT)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_input_len", type=int, default=128)
    p.add_argument("--gen_max_len", type=int, default=256)
    p.add_argument("--gen_min_len", type=int, default=8)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--no_repeat_ngram", type=int, default=3)
    p.add_argument("--length_penalty", type=float, default=1.0)
    p.add_argument("--tag", default="seq2seq", help="filename tag for outputs")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only generate for the first N rows (smoke test; skips full submission)")
    return p.parse_args()


def generate(model_dir, srcs, batch_size, max_input_len, gen_kwargs):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir)
    # LoRA runs save only an adapter; reload the base model and attach it.
    if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        import json

        from peft import PeftModel

        with open(os.path.join(model_dir, "adapter_config.json")) as f:
            base_name = json.load(f)["base_model_name_or_path"]
        base = AutoModelForSeq2SeqLM.from_pretrained(base_name)
        model = PeftModel.from_pretrained(base, model_dir)
        model = model.merge_and_unload()  # fold adapters into base for faster generation
        model = model.to(device).eval()
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()

    preds: list[str] = []
    for i in range(0, len(srcs), batch_size):
        chunk = srcs[i : i + batch_size]
        enc = tok(
            chunk, max_length=max_input_len, truncation=True,
            padding=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)
        preds.extend(tok.batch_decode(out, skip_special_tokens=True))
        print(f"\r[infer] {min(i + batch_size, len(srcs))}/{len(srcs)}", end="", flush=True)
    print()
    return [normalize_text(p) for p in preds]


def main():
    args = build_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = D.load_split(args.split, args.data_dir, clean=True, prefix=True)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)
        print(f"[infer] LIMIT={args.limit} (smoke test — partial output, not a full submission)")

    gen_kwargs = dict(
        max_length=args.gen_max_len,
        min_length=args.gen_min_len,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram,
        length_penalty=args.length_penalty,
    )
    preds = generate(args.model_dir, df.src.tolist(), args.batch_size, args.max_input_len, gen_kwargs)

    # Wrong-language guard diagnostic: Amharic predictions should be Ge'ez script.
    amh = df.subset == "Amh_Eth"
    if amh.any():
        wrong = sum(not looks_amharic(p) for p, m in zip(preds, amh) if m)
        print(f"[infer] Amharic rows not in Ge'ez script: {wrong}/{int(amh.sum())}")

    if args.split == "val":
        overall = compute_rouge(preds, df.output.tolist())
        by_lang = compute_rouge_by_group(preds, df.output.tolist(), df.subset.tolist())
        print(f"[infer] VAL R1={overall.rouge1_f1:.4f} RL={overall.rougeL_f1:.4f} proxy={overall.proxy:.4f}")
        print("\nPer-language ROUGE:")
        print(pd.DataFrame(by_lang).T.round(4).to_string())
        out = df[["ID", "subset"]].copy()
        out["prediction"] = preds
        out["reference"] = df.output.values
        path = os.path.join(args.out_dir, f"{args.tag}_val_predictions.csv")
        out.to_csv(path, index=False)
        print(f"[infer] wrote {path}")
    else:
        raw = df[["ID", "subset"]].copy()
        raw["prediction"] = preds
        raw_path = os.path.join(args.out_dir, f"{args.tag}_test_predictions.csv")
        raw.to_csv(raw_path, index=False)
        print(f"[infer] wrote raw predictions -> {raw_path}")
        if args.limit:
            print("[infer] LIMIT set -> skipping submission file (partial predictions only).")
            return
        sub = build_submission(df.ID.tolist(), preds)
        sub_path = os.path.join(args.out_dir, f"submission_{args.tag}.csv")
        save_submission(sub, sub_path, os.path.join(args.data_dir, "SampleSubmission.csv"))
        print("[infer] REVIEW predictions before uploading to Zindi (no upload was performed).")


if __name__ == "__main__":
    main()
