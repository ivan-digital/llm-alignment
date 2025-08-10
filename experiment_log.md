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
❌ **Sampling failed:** Used full dataset instead of 25% (47 hours vs planned 3 hours)  
❌ **Early stopping unused:** MT-Bench evaluation not triggered during training  

---

## Experiment 3: Fast Sampling (READY TO TEST 🔄)

### Fixed Parameters for Next Run
| Parameter | Value | Reason |
|-----------|-------|---------|
| Sample Ratio | **0.25** (fix default) | 4x faster training |
| Early Stopping | **Enable during training** | Catch forgetting early |
| Eval Steps | **100** | More frequent MT-Bench checks |
| Other Params | Same as Exp 2 | Keep conservative settings |

**Command to run:**
```bash
poetry run python sft_train.py --learning_rate 1e-5 --lora_r 8 --lora_alpha 16 --early_stop_patience 3 --eval_steps 100
```
(Should now use 25% sampling by default)