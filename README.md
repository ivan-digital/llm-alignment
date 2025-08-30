# llm-alignment

Brief recipes and results for aligning Qwen3 models with LoRA-based SFT and DPO. This README distills the training scripts and experiment logs in this repo into a practical, minimal guide.

## What’s included
- SFT: `sft_train.py` (LoRA SFT on T-Wix, MT-Bench-RU eval, optional KL regularization and early stopping)
- DPO (single-task): `dpo_train.py` (auto preference pairs from TAT-QA + TAT-QA/MT-Bench/Instruction eval)
- DPO (multi-task): `dpo_train_mt.py` (Reddit Finance + Finance-Instruct; optional 3-way with general instructions)
- DPO (progressive multi-stage): `dpo_train_mt_v2.py` (Stage 1 general → Stage 2 finance → Stage 3 social finance)
- Experiment logs: `sft_experiments_qwen3_*.md`, `dpo_experiments*.md`, `dpo_experiments*.jsonl`

## Datasets and benchmarks
- T-Wix (t-tech/T-Wix): Russian mixed general/reasoning chat data (SFT)
- TAT-QA: Finance QA over tables + passages (DPO pair construction; EM-like accuracy)
- Reddit Finance 43 (winddude/reddit_finance_43_250k): Post→comment (DPO pairs via comment score)
- Finance-Instruct-500k: Financial instruction-following (pairs by using reference as chosen)
- General DPO (e.g., Intel/orca_dpo_pairs): General capability preservation in progressive flow
- MT-Bench (RU/EN): General capability scoring via reward model on a 1–10 scale (avg score)

## Metrics (how we score)
- SFT validation: eval_loss, perplexity
- TAT-QA: EM-like accuracy (numeric tolerance for arithmetic)
- MT-Bench: average reward-model score (1–10)
- Instruction following: average reward-model score and success rate (≥7/10)
- Reddit reward: average reward-model score on generated replies

## Approach 1 — SFT (LoRA)
- Script: `sft_train.py`
- Core ideas: 4-bit base + LoRA; KL(student||reference) anchoring to reduce forgetting (mode-seeking D_KL; optional tail-only and warmup); optional MT-Bench-RU early-stopping; dataset subsetting and reasoning mix.
- Useful flags: `--subset [general|reasoning|both]`, `--mix_reasoning_ratio`, `--kl_coef/start/warmup/limit_seq`, `--mtbench_ru auto`, `--mtbench_questions_limit`, `--early_stop_patience`.

Recommended SFT config (balanced, small model):
- Model: Qwen/Qwen3-1.7B
- Data: subset=both, mix_reasoning_ratio≈0.2, sample_ratio≈0.25, max_seq_length≈512–768
- LoRA: r=8, α=16, dropout=0.05; lr≈1e-5; epochs 1–2; pack off
- KL regularization: coef≈0.03, start≈200, warmup≈300–400, limit_seq=128 (tail-only)
- MT-Bench eval: greedy (temperature=0), full 80Q for final

Observed SFT results (from logs):
- S4 (2 epochs, conservative KL): val loss 1.4698 → 1.2672; PPL 4.3482 → 3.5509; MT-Bench (80Q) 2.4949 → 2.8033.
- S3 (1 epoch, tuned): val loss 1.4698 → 1.2949; PPL 4.3482 → 3.6507; MT-Bench 2.5893 → 3.3673 (32Q subset); full-80Q confirmation: 2.7721.

Quick start (PowerShell, single line):
```
poetry run python sft_train.py --model Qwen/Qwen3-1.7B --output_dir qwen3_sft_mix20 --subset both --mix_reasoning_ratio 0.2 --sample_ratio 0.25 --epochs 1 --batch 8 --grad_acc 16 --max_seq_length 512 --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --kl_coef 0.03 --kl_start_step 200 --kl_warmup_steps 300 --kl_limit_seq 128 --mtbench_ru auto --mtbench_questions_limit 0 --mtbench_temperature 0.0 --num_proc 4
```

## Approach 2 — DPO (single-task TAT-QA)
- Script: `dpo_train.py`
- Core ideas: stream TAT-QA JSON; build preference pairs per question (chosen=gold; rejected=plausible); fresh LoRA on 4-bit base; evaluate on TAT-QA, MT-Bench (RU), instruction following.
- Useful flags: `--beta`, `--learning_rate`, `--max_prompt_length/max_length`, `--use_chat_template`, `--do_mtbench`, `--do_instruct`.

Best single-stage DPO config (from experiments):
- Model: Qwen/Qwen3-1.7B; β=0.02; lr=5e-6; epochs=1; LoRA r=16, α=32; batch=1, grad_acc=32; context≈768/2560
- Results snapshot (validation): TAT-QA +5.4pp; MT-Bench +0.38; Instruction small positive
- Notes: 2 epochs overfit; lr=2.5e-6 undertrained

Quick start (PowerShell, single line):
```
poetry run python dpo_train.py --output_dir qwen3_tatqa_dpo_beta002_lora16 --beta 0.02 --epochs 1 --batch 1 --grad_acc 32 --learning_rate 5e-6 --lora_r 16 --lora_alpha 32 --max_prompt_length 768 --max_length 2560 --eval_split validation --eval_limit 1000 --do_mtbench --mtbench_file mt_bench_ru.json --mtbench_limit 50
```

## Approach 3 — DPO (multi-task hybrid)
- Script: `dpo_train_mt.py`
- Core ideas: Reddit Finance explicit pairs + Finance-Instruct reference pairs; optional 3-way with general DPO. Evaluates on TAT-QA, MT-Bench (EN), instruction, and Reddit reward.

Balanced config (recommended in logs):
- β=0.0175, lr=4e-6, batch=1, grad_acc=64, LoRA r=8/α=16
- Hybrid targeting 50–50 Reddit/Finance-Instruct; actual example: ≈81/19 (after sampling)
- Results snapshot: TAT-QA +1.9%; MT-Bench −0.2%; Finance-Instruct −4.3%; Reddit reward −5.9% (best overall stability)

Quick start (PowerShell, single line):
```
poetry run python dpo_train_mt.py --use_hybrid --output_dir qwen3_reddit_finance_dpo --hybrid_ratio 0.5 --beta 0.0175 --learning_rate 4e-6 --finance_instruct_pairs 1800 --reddit_sample_rate 0.2 --mtbench_file mt_bench_en.json --mtbench_limit 50
```

## Approach 4 — Progressive multi-stage DPO
- Script: `dpo_train_mt_v2.py`
- Stages: (1) general capability preservation → (2) finance foundation → (3) social finance integration
- Results snapshot (from logs): all four domains improved simultaneously (TAT-QA ≈+2.4%, MT-Bench ≈+2.5%, Finance-Instruct ≈+3.3%, Reddit ≈+2.9%)

Quick start (PowerShell, single line):
```
poetry run python dpo_train_mt_v2.py --progressive --output_dir qwen3_3stage_progressive_experiment
```

## Practical tips
- Prefer greedy decoding (temperature=0) for MT-Bench to reduce variance
- Keep DPO to 1 epoch unless validated continuously—2+ epochs often overfit
- For SFT, KL regularization with warmup and limiting to tail tokens helps preserve general ability
- Use 4-bit load for base + LoRA for memory; adjust `--grad_acc` to fit

## Best options at a glance
- SFT (Qwen3-1.7B): subset=both, mix_reasoning_ratio≈0.2, lr≈1e-5, LoRA r=8/α=16, KL 0.03 (warmup), 1–2 epochs; MT-Bench full at final
- DPO single-task (TAT-QA): β=0.02, lr=5e-6, 1 epoch, LoRA r=16/α=32, short max lengths
- DPO multi-task (balanced): β≈0.0175, lr≈4e-6, hybrid≈50–50 (Reddit/Finance-Instruct)
- DPO progressive 3-stage: small Stage 1 preserve → Stage 2 finance-only → Stage 3 hybrid; achieved all-positive gains

## Repo notes
- Trained adapters and eval JSONLs live under directories like `qwen3_tatqa_dpo_*`, `qwen3_reddit_finance_dpo*`, and `qwen3_3stage_progressive_experiment/`
- MT-Bench RU/EN question sets: `mt_bench_ru.json`, `mt_bench_en.json`; RU file auto-downloads if `--mtbench_ru auto` in SFT

---
Requirements coverage
- Summary of SFT and DPO approaches: done
- Brief experiment setup and configs: done
- Datasets and benchmark metrics: done
- Results with best options and how to run: done
