#!/usr/bin/env python
"""sft_train.py – LoRA Supervised Fine-Tuning (SFT) on T-Wix for **Qwen-3-8B**

• Splits data into train/validation (default 99 % / 1 %).
• Prints **pre-training** and **post-training** loss / perplexity on both
  train & validation splits.

This version drops the `evaluation_strategy` argument entirely to stay
compatible with *any* Transformers / TRL mixture you might have in your
environment.

Usage
-----
```bash
python sft_train.py --output_dir qwen3_twix_lora_sft
```
"""
import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_chatml(example, tokenizer):
    user, assistant = example["messages"][0]["content"], example["messages"][1]["content"]
    msgs = [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    example["text"] = tokenizer.apply_chat_template(msgs, tokenize=False)
    return example


def prepare_datasets(tokenizer, subset: str, val_split: float, streaming: bool):
    """Return (train_ds, val_ds) as HF Datasets."""
    if streaming:
        pct = int(val_split * 100)
        train_ds = load_dataset("t-tech/T-Wix", split=f"train[{pct}%:]", streaming=True)
        val_ds = load_dataset("t-tech/T-Wix", split=f"train[:{pct}%]")
    else:
        full = load_dataset("t-tech/T-Wix", split="train")
        if subset != "both":
            full = full.filter(lambda ex: ex["subset"] == subset)
        split = full.train_test_split(test_size=val_split, seed=42)
        train_ds, val_ds = split["train"], split["test"]

    train_ds = train_ds.map(lambda ex: _to_chatml(ex, tokenizer))
    val_ds = val_ds.map(lambda ex: _to_chatml(ex, tokenizer))
    return train_ds, val_ds


def parse_args():
    p = argparse.ArgumentParser("LoRA SFT for Qwen-3 on T-Wix")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="Base HF model ID")
    p.add_argument("--output_dir", default="qwen3_twix_lora_sft")
    p.add_argument("--subset", choices=["general", "reasoning", "both"], default="general")
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--streaming", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4-bit quantisation → fits Qwen-3-8B on a 24 GB GPU
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )

    # LoRA config
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Data
    train_ds, val_ds = prepare_datasets(tok, args.subset, args.val_split, args.streaming)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    # Training-args (no evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=out_dir.as_posix(),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        fp16=True,
        save_strategy="epoch",
        logging_steps=50,
        gradient_checkpointing=True,
        optim = "paged_adamw_32bit",
    )

    # Metric helper
    def compute_metrics(eval_pred):
        loss = eval_pred[0] if isinstance(eval_pred, tuple) else eval_pred
        try:
            scalar = float(loss.mean())
        except AttributeError:
            scalar = float(loss)
        ppl = math.exp(scalar) if scalar < 20 else float("inf")
        return {"eval_loss": scalar, "perplexity": ppl}

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    base_val = trainer.evaluate()
    base_train = trainer.evaluate(eval_dataset=train_ds,
                                  metric_key_prefix ="train")
    print("Baseline metrics →", {**base_train, **base_val})
    trainer.train()
    final_val = trainer.evaluate()
    final_train = trainer.evaluate(eval_dataset=train_ds,
                                    metric_key_prefix ="train")
    print("Final metrics →", {**final_train, **final_val})

    model.save_pretrained(out_dir.as_posix())
    print(f"✅ LoRA adapter saved → {out_dir}")


if __name__ == "__main__":
    main()