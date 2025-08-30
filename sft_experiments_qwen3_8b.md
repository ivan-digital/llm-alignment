# Experiment Results Log

## Experiment 1: Catastrophic Forgetting (FAILED ❌)
**Date:** 2025-08-05 to 2025-08-07 | **Duration:** 47 hours

### Parameters
| Parameter | Value | Issue |
|-----------|--------|-------|
| Learning Rate | 5e-5 | Too high ❌ |
| LoRA Rank | 16 | Too aggressive ❌ |
| LoRA Alpha | 32 | Too aggressive ❌ |
| Early Stopping | None | Missing ❌ |
| Regularization | None | Missing ❌ |
| Generation Config | Default | Poor quality ❌ |

### Results
| Metric | Baseline | Final | Delta | Status |
|--------|----------|-------|-------|--------|
| T-Wix Loss | 1.2848 | 0.8430 | -34.4% | ✅ Good |
| T-Wix Perplexity | 3.6141 | 2.3234 | -35.7% | ✅ Good |
| MT-Bench Average | 2.0357 | 1.4924 | **-26.7%** | ❌ **Catastrophic Forgetting** |

**Root Cause:** Aggressive parameters caused severe overfitting - model learned training data but forgot general capabilities.

---

## Experiment 4: Mixed Subset + Stronger KL Warmup (PLANNED ▶)
**Objective:** Reduce MT-Bench forgetting by mixing reasoning data and slightly stronger KL with warmup.

### Planned Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Subset | both | Allow mixing |
| Mix Reasoning Ratio | 0.2 | 20% reasoning to protect skills |
| Sample Ratio | 0.25 | Keep runtime similar |
| KL | coef=0.03, start=200, warmup=300, limit_seq=128 | Stronger but ramped |
| Eval Steps | 250 | Fewer MT-Bench evals to save time |
| MT-Bench Qs | 16 | Faster mid-train checks |
| Skip Baseline | true | Faster rerun |

### Command
```powershell
poetry run python sft_train.py `
	--output_dir qwen3_twix_lora_sft_ru_exp4_mix `
	--subset both `
	--mix_reasoning_ratio 0.2 `
	--sample_ratio 0.25 `
	--learning_rate 1e-5 `
	--lora_r 8 --lora_alpha 16 `
	--epochs 1 `
	--early_stop_patience 3 `
	--eval_steps 250 `
	--mtbench_questions_limit 16 `
	--kl_coef 0.03 --kl_start_step 200 --kl_warmup_steps 300 --kl_limit_seq 128 `
	--max_seq_length 768 `
	--skip_baseline_eval `
	--baseline_eval_samples 30 `
	--num_proc 4
```
### Quick Run (1% sampling) — Completed ✅
Date: 2025-08-14 | Dataset: 1% (mix reasoning 20%) | Baseline: skipped

Parameters (delta vs planned):
- sample_ratio: 0.01 (vs 0.25 planned)
- mtbench_questions_limit: 16 (subset, cached reward on second pass)
- kl: coef=0.03, start=200, warmup=300, limit_seq=128
- max_seq_length: 768; epochs: 1; batch: 8; grad_acc: 16; early_stop_patience: 3

Results
- Validation: eval_loss=1.1831, perplexity=3.2645 (n/a baseline)
- MT-Bench-RU (16 Q): avg=3.2246
- Train runtime: 1513.41s (25.2m), steps/s≈0.026
- Final MT-Bench pass: 180.09s (cached reward)

Output dir: `qwen3_twix_lora_sft_ru_exp4_mix_001`

## Experiment 2: Conservative Training (PARTIAL SUCCESS ⚠️)
**Date:** 2025-08-08 to 2025-08-10 | **Duration:** 47 hours | **Dataset:** Full (100% - sampling bug)

### Parameters Used
| Parameter | Planned | Actual | Issue |
|-----------|---------|--------|-------|
| Learning Rate | 1e-5 | ✅ 1e-5 | Correct |
| LoRA Rank | 8 | ✅ 8 | Correct |
| LoRA Alpha | 16 | ✅ 16 | Correct |
| Early Stopping | 3 patience | ❌ Not triggered | No MT-Bench during training |
| Sample Ratio | 25% | ❌ 100% | **Sampling bug - used full dataset** |
| Regularization | ✅ Weight decay, grad clipping | Applied |

### Results
| Metric | Baseline | Final | Delta | Status |
|--------|----------|-------|-------|--------|
| T-Wix Loss | 1.2848 | 0.9043 | -29.6% | ✅ Good improvement |
| T-Wix Perplexity | 3.6141 | 2.4701 | -31.6% | ✅ Good improvement |
| MT-Bench Average | **2.6565** | **2.4430** | **-8.0%** | ⚠️ **Mild Forgetting** |

### Analysis
✅ **Better than Exp 1:** MT-Bench degradation reduced from -26.7% to -8.0%  
✅ **Conservative params worked:** Less aggressive overfitting  
❌ **Still some forgetting:** Many questions dropped to 1.0 (minimum score)  

---

## Experiment 3: Fast 25% Sampling + KL Regularization (COMPLETED ✅)
**Date:** 2025-08-11 to 2025-08-12 | **Duration:** ~11 hours | **Dataset:** 25% of general split

### Parameters
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| LoRA Rank / Alpha | r=8, alpha=16 |
| Sample Ratio | 0.25 (fixed bug) |
| Epochs | 1 |
| Grad Accum / Batch | 16 / 8 |
| KL Regularization | coef=0.02, start_step=150, limit_seq=128 |
| MT-Bench During Train | patience=3, eval_steps=150, subset=24 Qs |
| Max Seq Length | 768 |
| Baseline Eval Samples | 40 |

### Results
| Metric | Baseline | Final | Delta | Status |
|--------|----------|-------|-------|--------|
| T-Wix Loss | 1.1034 | 0.9659 | -12.4% | ✅ Improved |
| T-Wix Perplexity | 3.0143 | 2.6273 | -12.8% | ✅ Improved |
| MT-Bench Average (24 Qs) | 2.8939 | 2.6824 | -7.3% | ⚠️ Mild forgetting |

Train runtime: 39,592s (~11.0h), steps/s: ~0.021 (GPU underutilized). MT-Bench evals took ~8.6 min each (24 Q subset with cached reward model).

---

## Experiment 4 (cont.): Main Run (25% sampling) — Completed ✅
Date: 2025-08-15 | Dataset: 25% of both (mix reasoning 20%) | Baseline: enabled

Parameters (key):
- subset=both; mix_reasoning_ratio=0.2; sample_ratio=0.25
- lr=1e-5; LoRA r=8, alpha=16; epochs=1; max_seq_length=768
- early_stop_patience=3; eval_steps=250; mtbench_questions_limit=16
- KL: coef=0.03, start=200, warmup=300, limit_seq=128
- baseline_eval_samples=40; num_proc=4

Artifacts:
- Output dir: `qwen3_twix_lora_sft_ru_exp4_main_025`
- Checkpoints: 500, 750, 967 (final); adapter saved at run root

Results
- Baseline (val): loss=1.0577, perplexity=2.8798
- Final (val): loss=0.9169, perplexity=2.5015  (≈ -13.3% loss, -13.1% ppl)
- Baseline MT-Bench-RU (16 Q): avg=2.0744
- Final MT-Bench-RU (16 Q): avg=2.4899  (≈ +20.0%)
- Mid-train best MT-Bench (subset 16 Q): 2.8352

Runtime
- Total: 48,408s (806.8 min)
- Training: 47,432s; Evaluation: 340s; MT-Bench: 355s; Model I/O: ~0.8s

Takeaway: Mixing in 20% reasoning plus KL warmup improved MT-Bench significantly while also lowering validation loss/perplexity vs baseline.

---

## Experiment 4 (full MT-Bench): 25% sampling + 20% reasoning mix + KL warmup — Completed ✅
Date: 2025-08-16 | Dataset: 25% of both (mix reasoning 20%) | Baseline: enabled | MT-Bench: full (80 Q)

Parameters (key differences vs earlier Exp4 run)
- mtbench_questions_limit=0 (full benchmark for baseline and final)
- subset=both; mix_reasoning_ratio=0.2; sample_ratio=0.25
- lr=1e-5; LoRA r=8, alpha=16; epochs=1; max_seq_length=768
- early_stop_patience=3; eval_steps=250
- KL: coef=0.03, start=200, warmup=300, limit_seq=128
- baseline_eval_samples=40; num_proc=4

Artifacts
- Output dir: `qwen3_twix_lora_sft_ru_exp4_main_025_full`

Results (full MT-Bench)
- Baseline (val): loss=1.0577, perplexity=2.8798
- Final (val): loss=0.9163, perplexity=2.5001  (≈ -13.4% loss, -13.2% ppl)
- Baseline MT-Bench-RU (80 Q): avg=2.8469
- Final MT-Bench-RU (80 Q): avg=2.7252  (Δ -0.1217, mild regression)

Runtime
- Total: 50,015s (833.6 min)
- Training: 47,055s; Evaluation: 1,126.8s; MT-Bench: 1,808.2s; Model I/O: ~0.4s

Observations & analysis
- Validation improved strongly (loss/ppl down), confirming learning on T-Wix.
- Full MT-Bench dropped slightly (-0.12) despite subset evals previously showing gains. This suggests the 16Q subset overestimated generalization; the full set exposes regressions on some tasks.
- Per-question deltas show a mix of large gains (e.g., Q2, Q15, Q16, Q25, Q27, Q30, Q38, Q44, Q58–62, Q70, Q76) and notable drops to the floor (1.0) on others (e.g., Q3, Q5–6, Q9, Q12, Q19, Q28, Q32, Q34, Q41, Q54, Q61, Q64, Q69, Q72–73, Q75, Q78–79). This pattern points to uneven transfer: improvements on some reasoning/instruction types with regressions on others.
- Early stopping didn’t trigger (score never exceeded baseline best=2.8469; patience logged as 1/3 at final eval path).

Likely causes
- Data mix may still overweight domains that don’t translate to certain MT-Bench categories; 20% reasoning helped subset metrics but not all categories in full eval.
- Anchor (KL forward, coef=0.03) constrains drift overall, but warmup and tail restriction focus on later tokens; some instruction formats might still shift.
- Reward-model evaluation uses sampling (temperature=0.7, top_p=0.9) which adds variance; however, both baseline and final used the same settings, so the consistent -0.12 suggests a real effect.

Actions to try next
- Data: increase mix_reasoning_ratio to 0.25–0.30, or stratify sampling by T-Wix categories to balance skills reflected in MT-Bench. Alternatively, curate a small “anchor” set closer to MT-Bench styles and upsample it.
- Anchor loss: test anchor_type=js and anchor_type=kl_rev; try slightly higher anchor_coef 0.04–0.05 with the same warmup to see if it reduces regressions without hurting learning.
- Evaluation stability: for reward-model scoring, reduce sampling variance (e.g., temperature=0.2 or greedy) during MT-Bench runs to lower noise and better detect real regressions.
- Curriculum: start with more general data in the first 30–40% steps, then blend in reasoning, keeping KL higher early to preserve base skills; decay KL slowly at the end.
- Run an ablation: full MT-Bench at baseline/mid/final with n=80 to monitor category trends (we can add category-wise aggregation if we tag questions).
