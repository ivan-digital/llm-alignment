#!/usr/bin/env python
"""
sft_train_mtbench_ru.py  –  LoRA SFT on the Russian T-Wix dataset
with automatic *before & after* evaluation on

  • T-Wix test-split perplexity
  • MT-Bench-RU average score (using Skywork reward model)

Key options
-----------
--mtbench_ru  auto | /path/to/mt_bench_ru.json
    auto  -> download + convert t-tech/ru-mt-bench/raw/question.jsonl

Example
~~~~~~~
python sft_train_mtbench_ru.py \
  --output_dir qwen3_twix_lora_sft_ru \
  --epochs 2 \
  --val_split 0.01 \
  --mtbench_ru auto
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import time
import warnings
import random
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sft_train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------#
# Early Stopping Callback for MT-Bench                                       #
# ---------------------------------------------------------------------------#
from transformers import TrainerCallback, TrainerState, TrainerControl

class MTBenchEarlyStoppingCallback(TrainerCallback):
    """Early stopping based on MT-Bench scores to prevent catastrophic forgetting.

    NOTE: Trainer callbacks must use the signature on_evaluate(self, args, state, control, **kwargs).
    We store model/tokenizer references provided at construction because the callback handler does
    not pass them as positional arguments (previous version caused TypeError).

    Stops when:
      • Score fails to improve by `min_delta` for `patience` evaluations, OR
      • Score drops more than `catastrophic_drop` below baseline.
    """

    def __init__(
        self,
        mtbench_path: str,
        baseline_score: float,
        patience: int = 3,
        min_delta: float = 0.05,
        catastrophic_drop: float = 0.30,
        questions_limit: int | None = None,
        questions_seed: int = 42,
        model=None,
        tokenizer=None,
        device: str = "cuda",
        gen_temperature: float = 0.0,
        gen_backend: str = "simple",
        gen_max_new_tokens: int = 512,
    ):
        self.mtbench_path = mtbench_path
        self.baseline_score = baseline_score
        self.patience = patience
        self.min_delta = min_delta
        self.catastrophic_drop = catastrophic_drop
        self.questions_limit = questions_limit
        self.questions_seed = questions_seed
        self.best_score = baseline_score
        self.patience_counter = 0
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gen_temperature = gen_temperature
        self.gen_backend = gen_backend
        self.gen_max_new_tokens = gen_max_new_tokens

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Expect metrics in kwargs (ignored here). Use stored model/tokenizer.
        if not (self.mtbench_path and state.epoch and state.epoch > 0.1):
            return control
        if self.model is None or self.tokenizer is None:
            logger.warning("EarlyStoppingCallback missing model/tokenizer references; skipping MT-Bench eval.")
            return control

        logger.info("Running MT-Bench evaluation for early stopping (subset=%s)...", self.questions_limit)
        try:
            scores = run_mtbench_ru(
                self.model,
                self.tokenizer,
                self.device,
                self.mtbench_path,
                max_tokens=self.gen_max_new_tokens,
                questions_limit=self.questions_limit,
                questions_seed=self.questions_seed,
                temperature=self.gen_temperature,
                mtbench_backend=self.gen_backend,
            )
            current_score = scores.get('mtbench_ru_avg', 0)
            logger.info(
                "MT-Bench score: %.4f (best: %.4f, baseline: %.4f)",
                current_score, self.best_score, self.baseline_score
            )

            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.patience_counter = 0
                logger.info("MT-Bench improved. New best: %.4f", self.best_score)
            else:
                self.patience_counter += 1
                logger.info("No improvement. Patience: %d/%d", self.patience_counter, self.patience)

            if current_score < self.baseline_score - self.catastrophic_drop:
                logger.warning(
                    "Catastrophic forgetting: %.4f < baseline %.4f - %.2f",
                    current_score, self.baseline_score, self.catastrophic_drop,
                )
                control.should_training_stop = True

            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered (no MT-Bench improvement)")
                control.should_training_stop = True
        except Exception as e:
            logger.error(f"MT-Bench evaluation failed: {e}")
        return control

# ---------------------------------------------------------------------------#
# MT-Bench-RU evaluation via FastChat                                         #
# ---------------------------------------------------------------------------#
try:
    from fastchat.eval import mtbench_ru as mtb  # ≥ v0.8.0-ru  # type: ignore[import-not-found]
except ImportError:                                    # degrade gracefully
    mtb = None

# Fallback MT-Bench evaluation implementation
class SimpleMTBenchEvaluator:
    _reward_model = None
    _reward_tokenizer = None

    @staticmethod
    def _suppress_transformers_warnings():
        """Context manager to temporarily silence Transformers generation warnings."""
        import contextlib
        @contextlib.contextmanager
        def _ctx():
            tr_logger = logging.getLogger("transformers")
            gen_logger = logging.getLogger("transformers.generation.utils")
            gen_config_logger = logging.getLogger("transformers.generation.configuration")
            modeling_logger = logging.getLogger("transformers.modeling_utils")
            prev_tr = tr_logger.level
            prev_gen = gen_logger.level
            prev_gc = gen_config_logger.level
            prev_mu = modeling_logger.level
            try:
                tr_logger.setLevel(logging.ERROR)
                gen_logger.setLevel(logging.ERROR)
                gen_config_logger.setLevel(logging.ERROR)
                modeling_logger.setLevel(logging.ERROR)
                yield
            finally:
                tr_logger.setLevel(prev_tr)
                gen_logger.setLevel(prev_gen)
                gen_config_logger.setLevel(prev_gc)
                modeling_logger.setLevel(prev_mu)
        return _ctx()

    @classmethod
    def _ensure_reward_model(cls):
        if cls._reward_model is not None:
            return
        logger.info("Loading Skywork reward model (cached)")
        reward_qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        from transformers import AutoModelForSequenceClassification
        cls._reward_tokenizer = AutoTokenizer.from_pretrained(
            "Skywork/Skywork-Reward-V2-Qwen3-0.6B", trust_remote_code=True
        )
        cls._reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            quantization_config=reward_qcfg,
            trust_remote_code=True,
            device_map="auto",
        )
        logger.info("Skywork reward model ready")

    @classmethod
    def run_evaluation(cls, model, tokenizer, questions, device, max_new_tokens=1024, temperature: float = 0.0):
        """MT-Bench evaluation using Skywork reward model (cached)."""
        logger.info(
            "Using custom MT-Bench evaluation (questions=%d, cached_reward=%s, temp=%.2f)",
            len(questions), cls._reward_model is not None, temperature,
        )
        cls._ensure_reward_model()
        reward_model = cls._reward_model
        reward_tokenizer = cls._reward_tokenizer

        scores = {}
        for i, question in enumerate(questions):
            if isinstance(question, dict):
                question_text = question.get('question', question.get('text', str(question)))
            else:
                question_text = str(question)

            inputs = tokenizer(question_text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Always use greedy decode for deterministic evaluation; avoid passing GenerationConfig
                safe_max_new = int(max(1, min(max_new_tokens, 1024)))
                # Suppress spurious warnings about unused sampling flags
                with cls._suppress_transformers_warnings():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=safe_max_new,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                        return_dict_in_generate=False,
                    )

            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            score = cls._score_with_reward_model(question_text, response, reward_model, reward_tokenizer)
            scores[f"question_{i+1}"] = score

            if (i + 1) % 5 == 0 or i == len(questions) - 1:
                logger.info("Processed %d/%d MT-Bench questions", i + 1, len(questions))
        return scores

    @staticmethod
    def _score_with_reward_model(question, response, reward_model, reward_tokenizer):
        """Score a response using the Skywork reward model."""
        # Format the conversation for the reward model
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]

        # Apply chat template
        conversation_text = reward_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize for the reward model
        inputs = reward_tokenizer(
            conversation_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        )
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get reward score
            outputs = reward_model(**inputs)
            reward_score = outputs.logits.squeeze().float().item()

            # Convert reward score to 1-10 scale
            if reward_score < 0:
                # Map negative scores (e.g., [-5, 5]) to [0, 1]
                normalized_score = (reward_score + 5) / 10
            else:
                # Map positive scores to [0, 1]
                normalized_score = min(1.0, reward_score / 5.0) if reward_score > 1 else reward_score

            # Scale to 1-10 range
            final_score = 1.0 + (normalized_score * 9.0)
            final_score = max(1.0, min(10.0, final_score))

            return final_score



# Use fallback if mtb is not available
if mtb is None:
    mtb = SimpleMTBenchEvaluator()


def run_mtbench_ru(
    model,
    tokenizer,
    device: str,
    bench_file: str | Path,
    max_tokens: int = 1024,
    questions_limit: int | None = None,
    questions_seed: int = 42,
    temperature: float = 0.0,
    mtbench_backend: str = "simple",
) -> Dict[str, float]:
    """Return per-question scores + overall average.

    Optionally evaluate only a random subset (`questions_limit`) for faster
    iterative monitoring. Uses a fixed seed for reproducibility.
    """
    logger.info("Starting MT-Bench-RU evaluation (limit=%s)", questions_limit)
    logger.info(f"Using benchmark file: {bench_file}")

    # Choose evaluator backend
    if mtbench_backend == "simple":
        evaluator = SimpleMTBenchEvaluator
    else:
        if mtb is None:
            evaluator = SimpleMTBenchEvaluator
        else:
            evaluator = mtb

    logger.info("Loading MT-Bench questions...")
    with open(bench_file, encoding="utf-8") as f:
        questions = json.load(f)
    original_q_count = len(questions)

    if questions_limit and questions_limit < original_q_count:
        rng = random.Random(questions_seed)
        questions = rng.sample(questions, questions_limit)
        logger.info(
            "Sampled %d/%d MT-Bench questions (seed=%d)",
            len(questions), original_q_count, questions_seed,
        )
    else:
        logger.info(f"Loaded {len(questions)} questions from MT-Bench (no subsampling)")

    logger.info("Running MT-Bench evaluation...")
    start_time = time.time()
    # Try temperature-aware evaluator; fall back if signature differs
    try:
        scores = evaluator.run_evaluation(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            device=device,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
    except TypeError:
        # Older FastChat or different signature without temperature
        scores = evaluator.run_evaluation(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            device=device,
            max_new_tokens=max_tokens,
        )
    elapsed = time.time() - start_time
    logger.info(f"MT-Bench evaluation completed in {elapsed:.2f} seconds")

    scores["mtbench_ru_avg"] = sum(scores.values()) / len(scores)
    logger.info(f"MT-Bench average score: {scores['mtbench_ru_avg']:.4f}")
    return scores


# ---------------------------------------------------------------------------#
# ru-MT-Bench fetch & convert                                                 #
# ---------------------------------------------------------------------------#
def _download_and_prepare_mtbench(out: Path) -> Path:
    """Download question.jsonl -> write JSON list file."""
    logger.info(f"Downloading MT-Bench-RU dataset to {out}")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        logger.error("huggingface_hub not available for downloading MT-Bench")
        raise RuntimeError(
            "huggingface_hub required – pip install huggingface_hub"
        ) from e

    logger.info("Downloading MT-Bench questions from HuggingFace Hub...")
    jsonl = hf_hub_download(
        repo_id="t-tech/ru-mt-bench",
        filename="raw/question.jsonl",
        repo_type="dataset",
    )
    logger.info(f"Downloaded questions file: {jsonl}")

    logger.info("Converting JSONL to JSON format...")
    data = [json.loads(l) for l in open(jsonl, encoding="utf-8")]
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"MT-Bench questions prepared and saved to {out}")
    return out


def ensure_mtbench_ru(spec: Optional[str]) -> Optional[str]:
    """Return existing JSON list path or download one."""
    if spec is None:
        logger.info("MT-Bench-RU evaluation disabled (no path specified)")
        return None

    logger.info(f"Ensuring MT-Bench-RU is available: {spec}")
    p = Path(spec)
    if spec == "auto" or not p.exists():
        logger.info("MT-Bench file not found, downloading automatically...")
        p = _download_and_prepare_mtbench(Path("mt_bench_ru.json"))
    else:
        logger.info(f"Using existing MT-Bench file: {p}")
        json.load(p.open())           # simple validation
        logger.info("MT-Bench file validation successful")
    return str(p)


# ---------------------------------------------------------------------------#
# Data helpers                                                                #
# ---------------------------------------------------------------------------#
def _to_chatml(example, tok):
    user = example["messages"][0]["content"]
    assistant = example["messages"][1]["content"]
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    example["text"] = tok.apply_chat_template(msgs, tokenize=False)
    return example


def prepare_datasets(tok, subset: str, val_split: float, streaming: bool, sample_ratio: float = 1.0, sample_seed: int = 42, num_proc: int = 1, mix_reasoning_ratio: float | None = None):
    """Return (train_ds, val_ds)."""
    logger.info(f"Preparing datasets with subset='{subset}', val_split={val_split}, streaming={streaming}")
    logger.info(f"Dataset sampling: {sample_ratio:.1%} of training data (seed={sample_seed})")

    if streaming:
        logger.info("Using streaming dataset mode")
        pct = int(val_split * 100)
        logger.info(f"Loading train split: train[{pct}%:] and validation split: train[:{pct}%]")
        train_ds = load_dataset("t-tech/T-Wix", split=f"train[{pct}%:]", streaming=True)
        val_ds = load_dataset("t-tech/T-Wix", split=f"train[:{pct}%]")

        # Note: Sampling with streaming mode is complex, so we'll log a warning
        if sample_ratio < 1.0:
            logger.warning("Dataset sampling with streaming mode is not fully implemented. Using full dataset.")
    else:
        logger.info("Loading full T-Wix dataset into memory")
        full = load_dataset("t-tech/T-Wix", split="train")
        logger.info(f"Loaded dataset with {len(full)} examples")

        if subset != "both":
            logger.info(f"Filtering dataset for subset: {subset}")
            full = full.filter(lambda ex: ex["subset"] == subset)
            logger.info(f"After filtering: {len(full)} examples")

        # Apply dataset sampling BEFORE train/val split
        if sample_ratio < 1.0 and subset == "both" and mix_reasoning_ratio is not None and 0.0 < mix_reasoning_ratio < 1.0:
            logger.info(f"Sampling with enforced mix (reasoning={mix_reasoning_ratio:.0%}, general={1-mix_reasoning_ratio:.0%}) ...")
            original_size = len(full)
            general_ds = full.filter(lambda ex: ex["subset"] == "general")
            reasoning_ds = full.filter(lambda ex: ex["subset"] == "reasoning")
            total_target = max(1, int(original_size * sample_ratio))
            n_reasoning = min(len(reasoning_ds), int(total_target * mix_reasoning_ratio))
            n_general = max(0, total_target - n_reasoning)
            general_take = general_ds.shuffle(seed=sample_seed).select(range(min(n_general, len(general_ds))))
            reasoning_take = reasoning_ds.shuffle(seed=sample_seed + 1).select(range(n_reasoning))
            from datasets import concatenate_datasets
            full = concatenate_datasets([general_take, reasoning_take]).shuffle(seed=sample_seed)
            sampled_size = len(full)
            logger.info(
                "Dataset sampled with mix: %d -> %d examples (general=%d, reasoning=%d)",
                original_size, sampled_size, len(general_take), len(reasoning_take)
            )
        elif sample_ratio < 1.0:
            logger.info(f"Sampling {sample_ratio:.1%} of the dataset (seed={sample_seed}) ...")
            original_size = len(full)
            split = full.train_test_split(
                train_size=sample_ratio,
                seed=sample_seed,
                shuffle=True,
            )
            sampled_ds = split["train"]  # explicit key to avoid ordering issues
            full = sampled_ds
            sampled_size = len(full)
            logger.info(
                "Dataset sampled: %d -> %d examples (%.1f%%)",
                original_size, sampled_size, 100 * sampled_size / original_size,
            )

        logger.info(f"Splitting dataset: {1-val_split:.1%} train, {val_split:.1%} validation")
        train_ds, val_ds = full.train_test_split(test_size=val_split, seed=42).values()

    logger.info("Converting examples to ChatML format...")
    train_ds = train_ds.map(lambda ex: _to_chatml(ex, tok), num_proc=num_proc)
    val_ds   = val_ds.map(lambda ex: _to_chatml(ex, tok), num_proc=max(1, min(2, num_proc)))

    # Log dataset sizes if not streaming
    if not streaming:
        logger.info(f"Final dataset sizes - Train: {len(train_ds)}, Validation: {len(val_ds)}")
    else:
        logger.info("Dataset preparation complete (streaming mode - sizes unknown)")

    return train_ds, val_ds


# ---------------------------------------------------------------------------#
# CLI                                                                         #
# ---------------------------------------------------------------------------#
def parse_args():
    p = argparse.ArgumentParser("LoRA SFT for Qwen-3 with MT-Bench-RU eval")
    p.add_argument("--model",      default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", default="qwen3_twix_lora_sft_ru")
    p.add_argument("--subset",     choices=["general", "reasoning", "both"], default="general")
    p.add_argument("--val_split",  type=float, default=0.01)
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--batch",      type=int,   default=8)
    p.add_argument("--grad_acc",   type=int,   default=16)
    p.add_argument("--streaming",  action="store_true")
    p.add_argument("--mtbench_ru", type=str,   default="auto",
                   help="'auto' or path to MT-Bench-RU JSON list")
    p.add_argument("--mtbench_backend", type=str, choices=["auto", "simple"], default="simple",
                   help="MT-Bench evaluator backend: 'simple' (built-in, quiet, deterministic) or 'auto' (use FastChat if available)")

    # Anti-catastrophic forgetting parameters
    p.add_argument("--learning_rate", type=float, default=1e-5,
                   help="Learning rate (lower helps prevent forgetting)")
    p.add_argument("--lora_r", type=int, default=8,
                   help="LoRA rank (lower = less adaptation)")
    p.add_argument("--lora_alpha", type=int, default=16,
                   help="LoRA alpha (lower = less aggressive adaptation)")
    p.add_argument("--early_stop_patience", type=int, default=None,
                   help="Stop if MT-Bench doesn't improve for N evaluations")
    p.add_argument("--eval_steps", type=int, default=200,
                   help="Evaluate every N steps for early stopping")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Warmup ratio for learning rate scheduler")

    # Dataset sampling parameters
    p.add_argument("--sample_ratio", type=float, default=0.25,
                   help="Fraction of training data to use (0.25 = 25% sampling)")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Random seed for dataset sampling reproducibility")
    p.add_argument("--mix_reasoning_ratio", type=float, default=None,
                   help="If subset='both', enforce a mix by sampling this fraction from 'reasoning' (e.g., 0.2 -> 20% reasoning, 80% general)")

    # MT-Bench speed / cost controls
    p.add_argument("--mtbench_questions_limit", type=int, default=0,
                   help="If set, randomly sample this many MT-Bench questions per evaluation (default: 32). Use a larger number (e.g. 80) for final eval or set to 0/all for full benchmark.")
    p.add_argument("--mtbench_questions_seed", type=int, default=42,
                   help="Random seed for MT-Bench question subsampling")
    p.add_argument("--mtbench_temperature", type=float, default=0.0,
                   help="Generation temperature for MT-Bench scoring (0=greedy)")

    # KL regularisation (reference model) to mitigate forgetting
    p.add_argument("--kl_coef", type=float, default=0.02,
                   help="Coefficient for KL(student||reference) added to loss (0 disables). Typical range 0.01–0.1")
    p.add_argument("--kl_start_step", type=int, default=0,
                   help="Start applying KL after this global step (warmup)")
    p.add_argument("--kl_log_interval", type=int, default=50,
                   help="Log KL diagnostics every N steps")
    p.add_argument("--kl_limit_seq", type=int, default=0,
                   help="If >0, compute KL only on the last N target tokens (reduces memory)")
    p.add_argument("--kl_warmup_steps", type=int, default=300,
                   help="Linearly ramp KL coef from 0 to target over this many steps after kl_start_step")

    # Performance / throughput controls
    p.add_argument("--max_seq_length", type=int, default=1024,
                   help="Max sequence length for training/eval")
    p.add_argument("--packing", action="store_true",
                   help="Enable sequence packing (if supported by TRL version)")
    p.add_argument("--num_proc", type=int, default=4,
                   help="Processes for dataset map (set 1 on Windows if spawn issues)")
    p.add_argument("--baseline_eval_samples", type=int, default=60,
                   help="Validation samples for baseline perplexity (smaller=faster startup)")
    p.add_argument("--skip_baseline_eval", action="store_true",
                   help="Skip baseline eval and MT-Bench to save time when rerunning")

    return p.parse_args()


# ---------------------------------------------------------------------------#
# Utils                                                                       #
# ---------------------------------------------------------------------------#
def compute_metrics(eval_pred):
    loss = eval_pred[0] if isinstance(eval_pred, tuple) else eval_pred
    scalar = float(loss.mean()) if hasattr(loss, "mean") else float(loss)
    ppl = math.exp(scalar) if scalar < 20 else float("inf")
    return {"eval_loss": scalar, "perplexity": ppl}


def print_delta(baseline: Dict[str, Any] | None, final: Dict[str, Any] | None, tag: str):
    """Print baseline -> final metrics safely (ASCII only).
    - Handles missing baseline (e.g., when --skip_baseline_eval is used).
    - Prints only numeric keys present in both dicts when baseline provided.
    """
    if not final:
        return

    # If no baseline, print finals only
    if not baseline:
        for k, v in sorted(final.items()):
            if isinstance(v, (int, float)):
                print(f"{tag:12} {k:>13}: n/a -> {v:.4f}")
        return

    keys = [k for k in sorted(set(baseline) & set(final)) if isinstance(baseline[k], (int, float)) and isinstance(final[k], (int, float))]
    for k in keys:
        base = baseline[k]
        fin = final[k]
        delta = fin - base
        print(f"{tag:12} {k:>13}: {base:.4f} -> {fin:.4f}   (d {delta:+.4f})")


# ---------------------------------------------------------------------------#
# Main                                                                        #
# ---------------------------------------------------------------------------#
def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting SFT training with MT-Bench-RU evaluation")
    logger.info("=" * 60)

    args = parse_args()
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Subset: {args.subset}")
    logger.info(f"  Validation split: {args.val_split}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch}")
    logger.info(f"  Gradient accumulation: {args.grad_acc}")
    logger.info(f"  Streaming: {args.streaming}")
    logger.info(f"  MT-Bench-RU: {args.mtbench_ru}")
    logger.info(f"  MT-Bench question limit: {args.mtbench_questions_limit}")
    logger.info(f"  MT-Bench question seed: {args.mtbench_questions_seed}")
    logger.info(f"  KL coef: {args.kl_coef}")
    logger.info(f"  KL start step: {args.kl_start_step}")
    logger.info(f"  Max seq length: {args.max_seq_length}")
    logger.info(f"  Packing: {args.packing}")
    logger.info(f"  Dataset map processes: {args.num_proc}")
    logger.info(f"  Baseline eval samples: {args.baseline_eval_samples}")
    if args.subset == "both" and args.mix_reasoning_ratio is not None:
        logger.info(f"  Enforced mix (reasoning): {args.mix_reasoning_ratio:.0%}")

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created: {out_dir}")

    # Model setup
    logger.info("Setting up model and tokenizer...")
    logger.info("Configuring 4-bit quantization for memory efficiency")
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

    logger.info(f"Loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    logger.info(f"Loading model from {args.model} with quantization")
    model_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=qcfg,
        trust_remote_code=True, device_map="auto",
    )
    model_load_time = time.time() - model_start
    logger.info(f"Model loaded in {model_load_time:.2f} seconds")

    # LoRA setup
    logger.info("Setting up LoRA adapter...")
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    logger.info(f"LoRA config: r={lora.r}, alpha={lora.lora_alpha}, dropout={lora.lora_dropout}")
    logger.info(f"Target modules: {lora.target_modules}")

    model = get_peft_model(model, lora)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    logger.info("LoRA adapter applied and model configured for training")

    # Load frozen reference model for KL if enabled
    ref_model = None
    if args.kl_coef > 0:
        try:
            logger.info("Loading frozen reference model for KL regularisation...")
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=qcfg,
                trust_remote_code=True,
                device_map="auto",
            )
            ref_model.eval()
            for p_ref in ref_model.parameters():
                p_ref.requires_grad_(False)
            logger.info("Reference model ready (parameters frozen)")
        except Exception as e:
            logger.warning(f"KL disabled – failed to load reference model: {e}")
            ref_model = None

    # Data preparation
    logger.info("Preparing training and validation datasets...")
    data_start = time.time()
    train_ds, val_ds = prepare_datasets(
        tok,
        args.subset,
        args.val_split,
        args.streaming,
        args.sample_ratio,
        args.sample_seed,
    args.num_proc,
    args.mix_reasoning_ratio,
    )
    data_prep_time = time.time() - data_start
    logger.info(f"Dataset preparation completed in {data_prep_time:.2f} seconds")

    # MT-Bench setup (before trainer to avoid optimizer memory allocation)
    logger.info("Setting up MT-Bench-RU evaluation...")
    mtbench_path = ensure_mtbench_ru(args.mtbench_ru)
    # Compute effective question limit once for reuse
    effective_limit = None if args.mtbench_questions_limit in (None, 0, -1) else args.mtbench_questions_limit

    base_val = {}
    base_mt = {}
    # Default timings in case baseline eval is skipped
    baseline_eval_time = 0.0
    mtbench_baseline_time = 0.0
    if not args.skip_baseline_eval:
        # Baseline evaluation WITHOUT trainer (no optimizer memory allocated)
        logger.info("=" * 40)
        logger.info("BASELINE EVALUATION")
        logger.info("=" * 40)
        logger.info("Running baseline validation evaluation (without optimizer)...")
        baseline_start = time.time()
        model.eval()
        total_loss = 0.0
        num_samples = 0
        from torch.utils.data import DataLoader
        eval_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for i, batch in enumerate(eval_dataloader):
                if i >= args.baseline_eval_samples:
                    break
                inputs = tok(batch['text'], return_tensors="pt", truncation=True, max_length=1024, padding=True)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                outputs = model(**inputs, labels=inputs['input_ids'])
                total_loss += outputs.loss.item()
                num_samples += 1
                if (i + 1) % 20 == 0:
                    logger.info(f"Evaluated {i + 1} samples...")
        baseline_loss = total_loss / num_samples if num_samples > 0 else float('inf')
        baseline_perplexity = math.exp(baseline_loss) if baseline_loss < 20 else float('inf')
        base_val = {'eval_loss': baseline_loss, 'perplexity': baseline_perplexity}
        baseline_eval_time = time.time() - baseline_start
        logger.info(f"Baseline validation completed in {baseline_eval_time:.2f} seconds")
        logger.info(f"Baseline loss: {baseline_loss:.4f}")
        logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")
        if mtbench_path:
            logger.info("Running baseline MT-Bench-RU evaluation (may use subset)...")
            mtbench_start = time.time()
            effective_limit = None if args.mtbench_questions_limit in (None, 0, -1) else args.mtbench_questions_limit
            base_mt = run_mtbench_ru(
                model,
                tok,
                "cuda",
                mtbench_path,
                questions_limit=effective_limit,
                questions_seed=args.mtbench_questions_seed,
                temperature=args.mtbench_temperature,
                mtbench_backend=args.mtbench_backend,
            )
            mtbench_baseline_time = time.time() - mtbench_start
            logger.info(f"Baseline MT-Bench evaluation completed in {mtbench_baseline_time:.2f} seconds")
        else:
            logger.info("Skipping MT-Bench-RU baseline (not configured)")
    else:
        logger.info("Skipping baseline evaluations as requested.")

    # Training - NOW initialize trainer (after baseline evaluation to save memory)
    logger.info("=" * 40)
    logger.info("STARTING TRAINING")
    logger.info("=" * 40)

    logger.info("Initializing SFT trainer (with optimizer)...")

    # Determine strategies (must match when load_best_model_at_end=True)
    if args.early_stop_patience:
        evaluation_strategy = "steps"
        save_strategy = "steps"
        eval_steps = args.eval_steps
        save_steps = args.eval_steps
        logging_steps = max(1, min(50, args.eval_steps // 2))
        want_load_best = True
    else:
        evaluation_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
        save_steps = None
        logging_steps = 50
        want_load_best = False

    # Introspect TrainingArguments signature for compatibility
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    training_args_kwargs = dict(
        output_dir=out_dir.as_posix(),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        fp16=True,
        save_strategy=save_strategy,
        logging_steps=logging_steps,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        prediction_loss_only=True,
        eval_accumulation_steps=8,
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        do_eval=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
    )

    # Only pass evaluation_strategy if supported
    if "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = evaluation_strategy
    # load_best_model_at_end only if both strategies set at init & supported
    if want_load_best and "load_best_model_at_end" in ta_params and "evaluation_strategy" in ta_params:
        training_args_kwargs["load_best_model_at_end"] = True
    else:
        # We'll simulate best model externally (early stopping) if not supported
        training_args_kwargs["load_best_model_at_end"] = False

    if eval_steps is not None:
        if "eval_steps" in ta_params:
            training_args_kwargs["eval_steps"] = eval_steps
        if "save_steps" in ta_params:
            training_args_kwargs["save_steps"] = save_steps

    training_args = TrainingArguments(**training_args_kwargs)

    # For older versions where evaluation_strategy wasn't in init, set attributes after
    if "evaluation_strategy" not in ta_params:
        setattr(training_args, "evaluation_strategy", evaluation_strategy)
        if eval_steps is not None:
            setattr(training_args, "eval_steps", eval_steps)
            setattr(training_args, "save_steps", save_steps)
    # Ensure save/eval alignment logged
    logger.info(
        f"TrainingArguments strategies -> evaluation: {getattr(training_args,'evaluation_strategy',None)} | save: {getattr(training_args,'save_strategy',None)}"
    )

    class KLSFTTrainer(SFTTrainer):
        def __init__(self, *t_args, ref_model=None, kl_coef=0.0, kl_start_step=0, kl_log_interval=50, kl_limit_seq=0, kl_warmup_steps=300, **t_kwargs):
            super().__init__(*t_args, **t_kwargs)
            self.ref_model = ref_model
            self.kl_coef = kl_coef
            self.kl_start_step = kl_start_step
            self.kl_log_interval = kl_log_interval
            self.kl_limit_seq = kl_limit_seq
            self.kl_warmup_steps = max(1, int(kl_warmup_steps))

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss
            apply_kl = (
                self.ref_model is not None and
                self.kl_coef > 0 and
                getattr(self.state, 'global_step', 0) >= self.kl_start_step and
                model.training
            )
            if apply_kl:
                # Linear warmup of KL coef
                steps_since = max(0, self.state.global_step - self.kl_start_step)
                ramp = min(1.0, steps_since / float(self.kl_warmup_steps))
                eff_kl_coef = self.kl_coef * ramp
                with torch.no_grad():
                    ref_out = self.ref_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs.get('attention_mask')
                    )
                    ref_logits_full = ref_out.logits
                # Free container ASAP
                del ref_out

                # Shift for causal LM (drop last logit position)
                student_logits = outputs.logits[:, :-1, :]
                ref_logits = ref_logits_full[:, :-1, :]
                del ref_logits_full  # free

                labels = inputs.get('labels')
                if labels is None:
                    labels = inputs['input_ids']
                labels = labels[:, 1:]

                # Optional sequence tail restriction for KL
                if self.kl_limit_seq and self.kl_limit_seq > 0:
                    tail_len = min(self.kl_limit_seq, student_logits.shape[1])
                    student_logits = student_logits[:, -tail_len:, :]
                    ref_logits = ref_logits[:, -tail_len:, :]
                    labels = labels[:, -tail_len:]

                mask = (labels != -100)

                # Log-softmax in-place friendly pattern
                student_logprob = F.log_softmax(student_logits, dim=-1)
                ref_logprob = F.log_softmax(ref_logits, dim=-1)
                # Avoid building separate softmax tensor: exp(log p)
                student_prob = student_logprob.exp()
                kl_token = (student_prob * (student_logprob - ref_logprob)).sum(-1)
                # Free intermediate large tensors early
                del student_logits, ref_logits, student_prob
                kl_token = kl_token * mask.float()
                denom = mask.float().sum().clamp_min(1.0)
                kl_mean = kl_token.sum() / denom
                del kl_token
                loss = loss + eff_kl_coef * kl_mean

                if self.state.global_step % self.kl_log_interval == 0:
                    logger.info(
                        "KL step=%d base_loss=%.4f kl=%.4f coef=%.3f total=%.4f (limit_seq=%s)", \
                        self.state.global_step, outputs.loss.item(), kl_mean.item(), eff_kl_coef, loss.item(), self.kl_limit_seq or 'full'
                    )
            return (loss, outputs) if return_outputs else loss

    trainer = KLSFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
        ref_model=ref_model,
        kl_coef=args.kl_coef,
        kl_start_step=args.kl_start_step,
        kl_log_interval=args.kl_log_interval,
        kl_limit_seq=args.kl_limit_seq,
    kl_warmup_steps=args.kl_warmup_steps,
        packing=args.packing,
    )
    logger.info("KLSFT trainer initialized (KL coef=%.3f)", args.kl_coef)

    # Register early stopping callback
    if args.early_stop_patience and mtbench_path:
        baseline_score = base_mt.get('mtbench_ru_avg', 0)
        logger.info(f"Registering early stopping callback with baseline score: {baseline_score:.4f}")
        trainer.add_callback(
            MTBenchEarlyStoppingCallback(
                mtbench_path=mtbench_path,
                baseline_score=baseline_score,
                patience=args.early_stop_patience,
                min_delta=0.05,
                questions_limit=effective_limit,
                questions_seed=args.mtbench_questions_seed,
                model=model,
                tokenizer=tok,
                device="cuda",
                gen_temperature=args.mtbench_temperature,
                gen_backend=args.mtbench_backend,
            )
        )

    training_start = time.time()
    logger.info(f"Beginning LoRA fine-tuning for {args.epochs} epoch(s)...")
    trainer.train()
    training_time = time.time() - training_start
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.1f} minutes)")

    # Final evaluation
    logger.info("=" * 40)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 40)

    logger.info("Running final validation evaluation...")
    final_eval_start = time.time()
    fin_val = trainer.evaluate()
    final_eval_time = time.time() - final_eval_start
    logger.info(f"Final validation completed in {final_eval_time:.2f} seconds")

    # Calculate perplexity manually from eval_loss
    final_loss = fin_val.get('eval_loss', float('inf'))
    final_perplexity = math.exp(final_loss) if final_loss < 20 else float('inf')
    fin_val['perplexity'] = final_perplexity
    logger.info(f"Final loss: {final_loss:.4f}")
    logger.info(f"Final perplexity: {final_perplexity:.4f}")

    fin_mt = {}
    if mtbench_path:
        logger.info("Running final MT-Bench-RU evaluation (full if limit disabled)...")
        final_mtbench_start = time.time()
        final_effective_limit = None if args.mtbench_questions_limit in (None, 0, -1) else args.mtbench_questions_limit
        fin_mt = run_mtbench_ru(
            model,
            tok,
            "cuda",
            mtbench_path,
            questions_limit=final_effective_limit,
            questions_seed=args.mtbench_questions_seed,
            temperature=args.mtbench_temperature,
            mtbench_backend=args.mtbench_backend,
        )
        final_mtbench_time = time.time() - final_mtbench_start
        logger.info(f"Final MT-Bench evaluation completed in {final_mtbench_time:.2f} seconds")
    else:
        logger.info("Skipping final MT-Bench-RU evaluation (not configured)")

    # Results summary
    logger.info("=" * 40)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 40)

    print_delta(base_val, fin_val, "T-Wix test")
    if mtbench_path:
        print_delta(base_mt, fin_mt, "MT-Bench-RU")

    # Save model
    logger.info("Saving LoRA adapter...")
    save_start = time.time()
    model.save_pretrained(out_dir.as_posix())
    save_time = time.time() - save_start
    logger.info(f"Model saved in {save_time:.2f} seconds")
    print(f"LoRA adapter saved -> {out_dir}")

    # Final timing summary
    total_time = time.time() - start_time
    logger.info("=" * 40)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 40)
    logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"  Model loading: {model_load_time:.2f}s")
    logger.info(f"  Data preparation: {data_prep_time:.2f}s")
    logger.info(f"  Training: {training_time:.2f}s")
    logger.info(f"  Evaluation: {baseline_eval_time + final_eval_time:.2f}s")
    if mtbench_path:
        logger.info(f"  MT-Bench: {mtbench_baseline_time + final_mtbench_time:.2f}s")
    logger.info(f"  Model saving: {save_time:.2f}s")
    logger.info("Training pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()