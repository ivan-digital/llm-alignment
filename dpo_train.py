#!/usr/bin/env python
"""
dpo_train.py  – LoRA-in-LoRA Direct Preference Optimisation for Qwen-3-8B

Example:
    python dpo_train.py \
        --lora_path qwen3_twix_lora_sft \
        --pairs_file dpo_pairs.jsonl \
        --output_dir qwen3_twix_lora_sft_dpo

Arguments
---------
--model        Base model ID (default: Qwen/Qwen3-8B)
--lora_path    Path to *SFT* LoRA adapter directory
--pairs_file   JSONL with {'prompt','chosen','rejected'} lines
--output_dir   Where to store the updated DPO adapter
--beta         DPO beta (default: 0.05)
--batch        Per-device batch size (default: 2)
--grad_acc     Gradient-accumulation steps (default: 16)
--epochs       Number of epochs (default: 1)
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import DPOTrainer


def parse_args():
    p = argparse.ArgumentParser("LoRA-in-LoRA DPO fine-tuning")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora_path", required=True,
                   help="Path to SFT LoRA adapter produced by sft_train.py")
    p.add_argument("--pairs_file", default="dpo_pairs.jsonl")
    p.add_argument("--output_dir", default="qwen3_twix_lora_sft_dpo")
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit quantisation so we can fit policy + reference in VRAM
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Policy model with *trainable* SFT LoRA
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )
    policy.load_adapter(args.lora_path)   # keep these weights trainable

    # Frozen reference model for KL regularisation
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )

    # Preference-pair dataset
    dpo_ds = load_dataset("json", data_files=args.pairs_file, split="train")

    trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=TrainingArguments(
            output_dir=out_dir.as_posix(),
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_acc,
            num_train_epochs=args.epochs,
            fp16=True,
            logging_steps=25,
            save_strategy="epoch",
        ),
        beta=args.beta,
        train_dataset=dpo_ds,
        tokenizer=tok,
        max_prompt_length=2048,
        max_length=3072,
        generate_during_eval=False,
    )

    trainer.train()
    policy.save_pretrained(out_dir.as_posix())
    print(f"✅ DPO LoRA adapter saved to {out_dir}")


if __name__ == "__main__":
    main()