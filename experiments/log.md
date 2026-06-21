# Experiment Log — Multilingual Health QA

> One entry per run. Machine-readable mirror in [results.csv](results.csv).
> Offline north-star = **proxy** = `0.5·ROUGE-1 + 0.5·ROUGE-L` on Val (the lexical 74% of
> the leaderboard; the 26% LLM-judge slice can't be reproduced locally — see PLAN.md §4).
> All ROUGE computed with `src/metrics.py`, which mirrors the organizers' scorer exactly
> (whitespace tokenizer, `use_stemmer=False`, mean per-row F1).

Change **one variable at a time** and record the takeaway. LB score is filled in *after*
manual review + manual upload (we do not auto-submit).

---

## B0 — TF-IDF retrieval baseline (anchor)  ·  2026-06-17
- **Config:** char_wb (3,5)-gram TF-IDF, cosine 1-NN, one model per subset + global fallback.
  CPU-only, no training. `python -m src.baseline_tfidf`.
- **Val:** R1 **0.4212** · RL **0.3660** · proxy **0.3936**
- **Per-language proxy:** Swa_Ken 0.585 · Eng_Ken 0.580 · Eng_Eth 0.512 · Lug_Uga 0.505 ·
  Eng_Uga 0.494 · Aka_Gha 0.225 · Eng_Gha 0.214 · **Amh_Eth 0.140 (worst)**.
- **Takeaway:** Surprisingly strong because ~39% of answers are templated duplicates, so
  copying the nearest real answer often overlaps the reference. This is the **floor** any
  fine-tuned model must beat. Effort should concentrate on **Amharic and the Ghanaian
  subsets (Akan, Eng_Gha)**, where retrieval collapses.

---

## B1 — mt5-base LoRA joint task-prefixed fine-tuning  ·  2026-06-19
- **Hypothesis:** Jointly fine-tuning mt5-base using LoRA adapters across all subsets with task-prefixed inputs will establish a generative baseline.
- **Config:** google/mt5-base + LoRA (r=16, alpha=32, dropout=0.05), 3 epochs, lr=3e-4, train_bs=4, eval_bs=4 (grad_accum=4), num_beams=4, max_input_len=128, gen_max_len=256, no_repeat_ngram=3, length_penalty=1.0, bf16 precision.
- **Val:** R1 **0.2036** · RL **0.1583** · proxy **0.1809**
- **Per-language proxy:** Eng_Eth 0.2908 · Eng_Gha 0.2555 · Aka_Gha 0.2407 · Swa_Ken 0.1880 · Eng_Ken 0.1275 · Amh_Eth 0.1251 · Eng_Uga 0.1239 · Lug_Uga 0.0961 (worst).
- **Takeaway:** The generative baseline scores below the retrieval baseline (0.1809 vs 0.3936), primarily due to retrieval benefiting heavily from exact template duplicates. However, this is our first baseline for synthesizing new answers. Amharic script-guard diagnostic: 3/462 rows not in Ge'ez script. We will proceed to explore upsampling low-resource subsets and testing other architectures like NLLB.

---

## B2 — mt5-base LoRA + low-resource upsampling + recall decoding  ·  2026-06-21 (running)
- **Hypothesis:** B1 underfit (3 epochs) and under-generated, starving ROUGE recall. Holding
  the model + LoRA config fixed, three changes should lift the proxy above B1 and toward B0:
  (1) **5 epochs** for more adapter steps, (2) **low-resource upsampling ×3** (Amh/Swa/Lug/Aka)
  to rescue the weakest subsets, (3) **recall-oriented decoding** (beams 5, length_penalty 1.3,
  gen_min_len 16) since B1's outputs were too short.
- **Config:** google/mt5-base + LoRA (r=16, alpha=32, dropout=0.05), 5 epochs, lr=3e-4,
  train_bs=8, eval_bs=8 (grad_accum=2 → eff. 16), max_input_len=128, max_target_len=256,
  upsample_low_resource=3.0, num_beams=5, gen_max_len=256, gen_min_len=16, no_repeat_ngram=3,
  length_penalty=1.3, bf16 (A100/L4) or fp16 fallback, gradient_checkpointing.
- **Run:** `notebooks/colab_run.ipynb` (cell 6). Tag `B2_mt5base_lora`.
- **Val:** R1 _ · RL _ · proxy _   (fill after run)
- **LB:** _ (after manual review + upload)
- **Takeaway:** _ (did epochs/upsampling/decoding close the gap to B0? next: B5 NLLB-600M.)

---

## Template for the next entry

## <run_id> — <one-line description>  ·  <date>
- **Hypothesis:** what single change vs. the previous best, and why we expect it to help.
- **Config:** model, epochs, lr, max_input_len/gen_max_len, decoding (beams, no_repeat_ngram,
  length_penalty), data tweaks (upsampling/dedup).
- **Val:** R1 _ · RL _ · proxy _   (per-language table if notable)
- **LB:** _ (after manual review + upload)
- **Takeaway:** did the hypothesis hold? what to try next?

---

## Planned progression (Phase 3 — one variable at a time)
1. **B1** mt5-base, greedy, defaults → first fine-tuned anchor vs B0.
2. **B2** + beam search (4) + `no_repeat_ngram=3` → decoding lever (usually big ROUGE gain).
3. **B3** tune `gen_max_len` / `length_penalty` (recall vs. over-generation).
4. **B4** low-resource **upsampling** (Amharic/Swahili/Luganda/Akan) → lift the weak subsets.
5. **B5** model swap: **NLLB-600M** (African coverage) vs mt5-base on same recipe.
6. **B6** scale up (mt5-large / NLLB-1.3B) if VRAM allows.
7. **Reserve** LoRA decoder LLM (Aya-101/Gemma) — only if judge-slice headroom justifies it.
