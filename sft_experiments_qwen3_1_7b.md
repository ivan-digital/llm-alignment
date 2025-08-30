# Qwen3-1.7B LoRA SFT: Reproduction Plan (deterministic, full MT-Bench)

This plan reproduces Experiments 1–4 from `experiment_log.md` on Qwen/Qwen3-1.7B with:
- Deterministic MT-Bench generation: temperature=0 (greedy)
- Full MT-Bench (80 questions) at baseline and final
- Same T-Wix training logic (LoRA, optional KL/anchor), same evaluation pipeline

Notes
- Commands use PowerShell-friendly syntax (single line). Poetry is used per `pyproject.toml`.
- Early-stopping mid-train MT-Bench is disabled to avoid long full-benchmark evaluations. We’ll run full MT-Bench only at baseline and final.
- If you prefer mid-train checks, set `--early_stop_patience 3 --eval_steps <N>`; be aware this will run the full benchmark each eval with the current code.

Global evaluation settings
- `--mtbench_ru auto` (downloads dataset if absent)
- `--mtbench_temperature 0.0` (greedy)
- `--mtbench_questions_limit 0` (full 80 Q at baseline/final)

## Exp1-1.7B — Aggressive (expect forgetting)
Assumptions (not fully specified in original): subset=general, sample_ratio=0.25, max_seq_length=768, epochs=1.

Command
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_exp1_aggressive --subset general --sample_ratio 0.25 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 768 --learning_rate 5e-5 --lora_r 16 --lora_alpha 32 --kl_coef 0.0 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --baseline_eval_samples 40 --num_proc 4

Record
- Baseline: val loss/ppl, MT-Bench avg (80Q, greedy)
- Final: val loss/ppl, MT-Bench avg (80Q, greedy)

## Exp2-1.7B — Conservative (partial success in 8B; here reproduce with full data)
Mirrors Exp2 with the sampling bug (100%). Early stopping off.

Command
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_exp2_conservative_full --subset general --sample_ratio 1.0 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 768 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.0 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --baseline_eval_samples 40 --num_proc 4

Record
- Baseline/final metrics as above.

## Exp3-1.7B — 25% sampling + KL (coef=0.02, start=150, tail=128)
Replicates Exp3 but with deterministic, full MT-Bench at baseline/final.

Command
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_exp3_kl002_g025 --subset general --sample_ratio 0.25 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 768 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.02 --kl_start_step 150 --kl_warmup_steps 1 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --baseline_eval_samples 40 --num_proc 4

Record
- Baseline/final metrics as above. Note: warmup set to 1 step to match “no warmup” behavior.

## Exp4-1.7B — both + reasoning mix 20% + stronger KL warmup
Replicates successful Exp4 configuration, but evaluated deterministically and on the full benchmark.

Command
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_exp4_mix_reasoning20 --subset both --mix_reasoning_ratio 0.2 --sample_ratio 0.25 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 768 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.03 --kl_start_step 200 --kl_warmup_steps 300 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --baseline_eval_samples 40 --num_proc 4

Record
- Baseline/final metrics as above, plus total runtime.

## Results Table (fill as runs complete)
| Exp | Subset | Sample | KL (coef/start/warmup/tail) | Val loss→ | PPL→ | MT-Bench avg→ |
|-----|--------|--------|------------------------------|-----------|------|----------------|
| 1   | general | 0.25 | none | b: 1.5023 → f: 1.2063 | b: 4.4919 → f: 3.3411 | b: 1.4585 → f: 1.5683 |
| 2   | general | 0.25 | none | b: … → f: … | b: … → f: … | b: … → f: … |
| 3   | general | 0.25 | 0.02/150/1/128 | b: … → f: … | b: … → f: … | b: … → f: … |
| 4   | both    | 0.25 | 0.03/200/300/128 | b: … → f: … | b: … → f: … | b: … → f: … |

## Notes
- Deterministic evaluation removes sampling noise; expect slightly different numbers vs 8B runs.
- If full MT-Bench runtime is too high, we can add a separate flag for mid-train subset checks; for now, we run only baseline+final.

---

## Small-model optimized path (Qwen3‑1.7B)
Rationale: smaller capacity benefits from gentler updates, KL regularization, shorter sequences, and explicit reasoning mix. Also, long answers were previously capped at 256 tokens in the simple evaluator; this has been lifted to 1024 to avoid truncation during MT-Bench.

S1 — Gentle SFT + KL (10% general, 2 epochs)
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_s1_gentle --subset general --sample_ratio 0.10 --epochs 2 --batch 8 --grad_acc 16 --max_seq_length 512 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.02 --kl_start_step 150 --kl_warmup_steps 300 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --mtbench_backend simple --baseline_eval_samples 40 --num_proc 4

S2 — Add reasoning mix (20%)
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_s2_mix20 --subset both --mix_reasoning_ratio 0.2 --sample_ratio 0.25 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 512 --learning_rate 8e-6 --lora_r 8 --lora_alpha 16 --kl_coef 0.03 --kl_start_step 200 --kl_warmup_steps 300 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --mtbench_backend simple --baseline_eval_samples 40 --num_proc 4

S3 (tuned batch + gentler KL/LR) — completed
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_s3_tune_b12ga11 --subset both --mix_reasoning_ratio 0.2 --sample_ratio 0.25 --epochs 1 --batch 12 --grad_acc 11 --max_seq_length 512 --learning_rate 1.2e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.035 --kl_start_step 200 --kl_warmup_steps 400 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 32 --mtbench_temperature 0.0 --mtbench_backend simple --early_stop_patience 2 --eval_steps 200 --baseline_eval_samples 40 --num_proc 4

Results: val loss 1.4698 → 1.2949; PPL 4.3482 → 3.6507; MT-Bench avg (32Q subset during final eval) 2.5893 → 3.3673; Full 80Q confirmation: 2.7721.

S4 — Two epochs, conservative KL (safe)
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_1p7b_s4_e2_mix20_kl003_lr1e-5_b12ga11 --subset both --mix_reasoning_ratio 0.2 --sample_ratio 0.25 --epochs 2 --batch 12 --grad_acc 11 --max_seq_length 512 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.03 --kl_start_step 200 --kl_warmup_steps 400 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --mtbench_backend simple --baseline_eval_samples 40 --num_proc 4

Results: val loss 1.4698 → 1.2672; PPL 4.3482 → 3.5509; MT-Bench avg (80Q) 2.4949 → 2.8033; total runtime ≈ 401 min.

Optional: If acceptable, start from Qwen/Qwen3-1.7B-Instruct to boost baseline MT-Bench; then apply S1–S3 with halved learning rate.

### Results (S-path)

| Run | Subset  | Sample | KL (coef/start/warmup/tail) | Val loss→             | PPL→                    | MT-Bench avg→              | Notes                      |
|-----|---------|--------|------------------------------|-----------------------|-------------------------|----------------------------|----------------------------|
| S1  | general | 0.10   | 0.02/150/300/128            | b: 1.7862 → f: 1.4188 | b: 5.9664 → f: 4.1323  | b: 2.4949 → f: 2.6585      | full 80Q, greedy, no ES    |
| S2  | both    | 0.25   | 0.03/200/300/128            | b: 1.4698 → f: 1.3185 | b: 4.3482 → f: 3.7378  | b: 2.4949 → f: 2.6294      | LR=8e-6; 20% reasoning mix; full 80Q, greedy |
| S3  | both    | 0.25   | 0.035/200/400/128           | b: 1.4698 → f: 1.2949 | b: 4.3482 → f: 3.6507  | b: 2.5893 → f: 2.7721 (80Q)| batch=12, grad_acc=11; baseline 32Q=2.5893; final 80Q=2.7721; ES subset dip not representative |
| S4  | both    | 0.25   | 0.03/200/400/128            | b: 1.4698 → f: 1.2672 | b: 4.3482 → f: 3.5509  | b: 2.4949 → f: 2.8033     | epochs=2; batch=12, grad_acc=11; full 80Q at baseline & final |

