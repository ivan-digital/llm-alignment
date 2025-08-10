#!/usr/bin/env python
"""
sft_train_mtbench_ru.py  –  LoRA SFT on the Russian T-Wix dataset
with automatic *before & after* evaluation on

  • T-Wix test-split perplexity
  • MT-Bench-RU average score (using Skywork reward model)

Key options
-----------
--mtbench_ru  auto | /path/to/mt_bench_ru.json
    auto  → download + convert t-tech/ru-mt-bench/raw/question.jsonl

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
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
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
    """Early stopping based on MT-Bench scores to prevent catastrophic forgetting."""

    def __init__(self, mtbench_path, baseline_score, patience=3, min_delta=0.05):
        self.mtbench_path = mtbench_path
        self.baseline_score = baseline_score
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = baseline_score
        self.patience_counter = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, model, tokenizer, **kwargs):
        if self.mtbench_path and state.epoch and state.epoch > 0.1:  # Skip very early evaluations
            logger.info("Running MT-Bench evaluation for early stopping...")
            try:
                scores = run_mtbench_ru(model, tokenizer, "cuda", self.mtbench_path, max_tokens=512)
                current_score = scores.get('mtbench_ru_avg', 0)

                logger.info(f"MT-Bench score: {current_score:.4f} (best: {self.best_score:.4f}, baseline: {self.baseline_score:.4f})")

                # Check if we improved
                if current_score > self.best_score + self.min_delta:
                    self.best_score = current_score
                    self.patience_counter = 0
                    logger.info(f"✅ MT-Bench improved! New best: {self.best_score:.4f}")
                else:
                    self.patience_counter += 1
                    logger.info(f"⚠️ No improvement. Patience: {self.patience_counter}/{self.patience}")

                # Check if score dropped significantly below baseline
                if current_score < self.baseline_score - 0.3:  # 0.3 point drop threshold
                    logger.warning(f"🚨 Catastrophic forgetting detected! Score dropped to {current_score:.4f} from baseline {self.baseline_score:.4f}")
                    control.should_training_stop = True

                # Check patience
                if self.patience_counter >= self.patience:
                    logger.info(f"🛑 Early stopping triggered. No improvement for {self.patience} evaluations.")
                    control.should_training_stop = True

            except Exception as e:
                logger.error(f"MT-Bench evaluation failed: {e}")

        return control

# ---------------------------------------------------------------------------#
# MT-Bench-RU evaluation via FastChat                                         #
# ---------------------------------------------------------------------------#
try:
    from fastchat.eval import mtbench_ru as mtb        # ≥ v0.8.0-ru
except ImportError:                                    # degrade gracefully
    mtb = None

# Fallback MT-Bench evaluation implementation
class SimpleMTBenchEvaluator:
    @staticmethod
    def run_evaluation(model, tokenizer, questions, device, max_new_tokens=1024):
        """MT-Bench evaluation using Skywork reward model."""
        logger.info(f"Using custom MT-Bench evaluation implementation")
        logger.info(f"Evaluating {len(questions)} Russian MT-Bench questions with model on {device}")
        logger.info(f"Using Skywork-Reward-V2-Qwen3-0.6B for scoring")

        # Load Skywork reward model for scoring
        logger.info("Loading Skywork reward model (0.6B) for high-quality scoring...")
        reward_qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        from transformers import AutoModelForSequenceClassification
        reward_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B", trust_remote_code=True)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
            quantization_config=reward_qcfg,
            trust_remote_code=True,
            device_map="auto"
        )
        logger.info("Skywork reward model loaded successfully")

        # Generate responses for each question
        scores = {}
        for i, question in enumerate(questions):
            # Extract question text
            if isinstance(question, dict):
                question_text = question.get('question', question.get('text', str(question)))
            else:
                question_text = str(question)

            # Generate model response
            inputs = tokenizer(question_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Improved generation config for better MT-Bench responses
                generation_config = {
                    'max_new_tokens': min(max_new_tokens, 256),  # Increased from 128
                    'do_sample': True,                           # Enable sampling for diversity
                    'temperature': 0.7,                          # Moderate creativity
                    'top_p': 0.9,                               # Nucleus sampling
                    'repetition_penalty': 1.1,                  # Reduce repetition
                    'pad_token_id': tokenizer.eos_token_id,
                    'eos_token_id': tokenizer.eos_token_id,
                    'use_cache': True,
                    'return_dict_in_generate': False,
                }

                outputs = model.generate(
                    **inputs,
                    **generation_config
                )

            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)

            # Score with reward model
            score = SimpleMTBenchEvaluator._score_with_reward_model(
                question_text, response, reward_model, reward_tokenizer
            )
            scores[f"question_{i+1}"] = score

            if (i + 1) % 5 == 0 or i == len(questions) - 1:
                logger.info(f"Processed {i+1}/{len(questions)} questions")

        # Clean up reward model
        del reward_model
        del reward_tokenizer
        torch.cuda.empty_cache()
        logger.info("Reward model cleaned up from memory")

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
) -> Dict[str, float]:
    """Return per-question scores + overall average."""
    logger.info(f"Starting MT-Bench-RU evaluation")
    logger.info(f"Using benchmark file: {bench_file}")

    if mtb is None:                                    # FastChat missing
        logger.error("FastChat not available for MT-Bench evaluation")
        raise ImportError(
            "fastchat[mtbench] not installed – "
            "pip install \"fastchat[mtbench]\""
        )

    logger.info("Loading MT-Bench questions...")
    with open(bench_file, encoding="utf-8") as f:
        questions = json.load(f)
    logger.info(f"Loaded {len(questions)} questions from MT-Bench")

    logger.info("Running MT-Bench evaluation...")
    start_time = time.time()
    scores = mtb.run_evaluation(
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
    """Download question.jsonl → write JSON list file."""
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


def prepare_datasets(tok, subset: str, val_split: float, streaming: bool, sample_ratio: float = 1.0, sample_seed: int = 42):
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
        if sample_ratio < 1.0:
            logger.info(f"Sampling {sample_ratio:.1%} of the dataset...")
            original_size = len(full)

            # Use train_test_split to get the sampled portion with fixed seed
            sampled_ds, _ = full.train_test_split(
                train_size=sample_ratio,
                seed=sample_seed,
                shuffle=True
            ).values()

            full = sampled_ds
            sampled_size = len(full)
            logger.info(f"Dataset sampled: {original_size} → {sampled_size} examples ({sampled_size/original_size:.1%})")

        logger.info(f"Splitting dataset: {1-val_split:.1%} train, {val_split:.1%} validation")
        train_ds, val_ds = full.train_test_split(test_size=val_split, seed=42).values()

    logger.info("Converting examples to ChatML format...")
    train_ds = train_ds.map(lambda ex: _to_chatml(ex, tok))
    val_ds   = val_ds.map(lambda ex: _to_chatml(ex, tok))

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
    p.add_argument("--model",      default="Qwen/Qwen3-8B")
    p.add_argument("--output_dir", default="qwen3_twix_lora_sft_ru")
    p.add_argument("--subset",     choices=["general", "reasoning", "both"], default="general")
    p.add_argument("--val_split",  type=float, default=0.01)
    p.add_argument("--epochs",     type=int,   default=1)
    p.add_argument("--batch",      type=int,   default=8)
    p.add_argument("--grad_acc",   type=int,   default=16)
    p.add_argument("--streaming",  action="store_true")
    p.add_argument("--mtbench_ru", type=str,   default="auto",
                   help="'auto' or path to MT-Bench-RU JSON list")

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

    return p.parse_args()


# ---------------------------------------------------------------------------#
# Utils                                                                       #
# ---------------------------------------------------------------------------#
def compute_metrics(eval_pred):
    loss = eval_pred[0] if isinstance(eval_pred, tuple) else eval_pred
    scalar = float(loss.mean()) if hasattr(loss, "mean") else float(loss)
    ppl = math.exp(scalar) if scalar < 20 else float("inf")
    return {"eval_loss": scalar, "perplexity": ppl}


def print_delta(baseline: Dict[str, Any], final: Dict[str, Any], tag: str):
    for k in sorted(set(baseline) & set(final)):
        if not all(isinstance(x, (int, float)) for x in (baseline[k], final[k])):
            continue
        delta = final[k] - baseline[k]
        print(f"{tag:12} {k:>13}: {baseline[k]:.4f} → {final[k]:.4f}   (Δ {delta:+.4f})")


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

    # Data preparation
    logger.info("Preparing training and validation datasets...")
    data_start = time.time()
    train_ds, val_ds = prepare_datasets(tok, args.subset, args.val_split, args.streaming, args.sample_ratio, args.sample_seed)
    data_prep_time = time.time() - data_start
    logger.info(f"Dataset preparation completed in {data_prep_time:.2f} seconds")

    # MT-Bench setup (before trainer to avoid optimizer memory allocation)
    logger.info("Setting up MT-Bench-RU evaluation...")
    mtbench_path = ensure_mtbench_ru(args.mtbench_ru)

    # Baseline evaluation WITHOUT trainer (no optimizer memory allocated)
    logger.info("=" * 40)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 40)

    logger.info("Running baseline validation evaluation (without optimizer)...")
    baseline_start = time.time()

    # Manual evaluation without trainer to save memory
    model.eval()
    total_loss = 0.0
    num_samples = 0

    # Create a simple dataloader for evaluation
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(eval_dataloader):
            if i >= 100:  # Limit to 100 samples for faster baseline evaluation
                break

            # Tokenize the text
            inputs = tok(batch['text'], return_tensors="pt", truncation=True, max_length=1024, padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            total_loss += loss.item()
            num_samples += 1

            if (i + 1) % 20 == 0:
                logger.info(f"Evaluated {i + 1} samples...")

    baseline_loss = total_loss / num_samples if num_samples > 0 else float('inf')
    baseline_perplexity = math.exp(baseline_loss) if baseline_loss < 20 else float('inf')

    base_val = {
        'eval_loss': baseline_loss,
        'perplexity': baseline_perplexity
    }

    baseline_eval_time = time.time() - baseline_start
    logger.info(f"Baseline validation completed in {baseline_eval_time:.2f} seconds")
    logger.info(f"Baseline loss: {baseline_loss:.4f}")
    logger.info(f"Baseline perplexity: {baseline_perplexity:.4f}")

    base_mt = {}
    if mtbench_path:
        logger.info("Running baseline MT-Bench-RU evaluation...")
        mtbench_start = time.time()
        base_mt = run_mtbench_ru(model, tok, "cuda", mtbench_path)
        mtbench_baseline_time = time.time() - mtbench_start
        logger.info(f"Baseline MT-Bench evaluation completed in {mtbench_baseline_time:.2f} seconds")
    else:
        logger.info("Skipping MT-Bench-RU baseline (not configured)")

    # Training - NOW initialize trainer (after baseline evaluation to save memory)
    logger.info("=" * 40)
    logger.info("STARTING TRAINING")
    logger.info("=" * 40)

    logger.info("Initializing SFT trainer (with optimizer)...")

    # Create training arguments with compatible parameters
    training_args = TrainingArguments(
        output_dir=out_dir.as_posix(),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        prediction_loss_only=True,
        eval_accumulation_steps=8,
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
        learning_rate=args.learning_rate,

        # Anti-catastrophic forgetting settings
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,

        # Evaluation settings - use compatible parameter names
        do_eval=True,
        load_best_model_at_end=True if args.early_stop_patience else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Add evaluation strategy based on early stopping
    if args.early_stop_patience:
        training_args.evaluation_strategy = "steps"
        training_args.eval_steps = args.eval_steps
    else:
        training_args.evaluation_strategy = "epoch"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=1024,
        args=training_args,
    )
    logger.info("SFT trainer initialized successfully")

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
        logger.info("Running final MT-Bench-RU evaluation...")
        final_mtbench_start = time.time()
        fin_mt = run_mtbench_ru(model, tok, "cuda", mtbench_path)
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
    print(f"✅  LoRA adapter saved → {out_dir}")

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