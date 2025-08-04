#!/usr/bin/env python
"""
mine_pairs.py  – Generate chosen/rejected pairs for DPO

After finishing LoRA SFT, run:

    python mine_pairs.py \
        --lora_path qwen3_twix_lora_sft \
        --output_file dpo_pairs.jsonl

Options
-------
--model        Base model ID (default: Qwen/Qwen3-8B)
--lora_path    Path to the LoRA adapter directory produced by sft_train.py
--output_file  Where to write JSONL lines (default: dpo_pairs.jsonl)
--pairs        Number of pairs to mine (default: 50000)
--cands        Candidate generations per prompt (default: 4)
--streaming    Use HF streaming dataset API to save RAM
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

RM_NAME = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"


def parse_args():
    p = argparse.ArgumentParser("Generate DPO pairs using the SFT-tuned policy")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora_path", required=True,
                   help="Path to LoRA adapter produced by sft_train.py")
    p.add_argument("--output_file", default="dpo_pairs.jsonl")
    p.add_argument("--pairs", type=int, default=50_000)
    p.add_argument("--cands", type=int, default=4,
                   help="Number of candidate generations per prompt")
    p.add_argument("--streaming", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # ----- Policy model with LoRA attached ---------------------------------
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                 bnb_4bit_compute_dtype=torch.float16)
    policy = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        device_map="auto",
    )
    policy.load_adapter(args.lora_path)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # ----- Reward model pipeline ------------------------------------------
    rm_tok = AutoTokenizer.from_pretrained(RM_NAME, trust_remote_code=True)
    rm = pipeline("text-classification",
                  model=RM_NAME,
                  tokenizer=rm_tok,
                  device_map="auto",
                  truncation=True)

    # ----- T-Wix dataset ---------------------------------------------------
    ds = load_dataset("t-tech/T-Wix", split="train", streaming=args.streaming)

    out_path = Path(args.output_file)
    written = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for ex in tqdm(ds, desc="mining"):
            if written >= args.pairs:
                break

            prompt = ex["messages"][0]["content"]
            chat_prompt = [{"role": "user", "content": prompt}]
            input_ids = tok.apply_chat_template(
                chat_prompt, tokenize=True, return_tensors="pt"
            ).to(policy.device)

            with torch.no_grad():
                gens = policy.generate(
                    input_ids=input_ids,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    max_new_tokens=256,
                    num_return_sequences=args.cands,
                )

            texts = [tok.decode(g, skip_special_tokens=True) for g in gens]
            scores = [rm(prompt + "\n" + t)[0]["score"] for t in texts]
            best = texts[scores.index(max(scores))]
            worst = texts[scores.index(min(scores))]

            fp.write(json.dumps(
                {"prompt": prompt, "chosen": best, "rejected": worst},
                ensure_ascii=False
            ) + "\n")
            written += 1

    print(f"✅ Wrote {written} preference pairs to {out_path}")


if __name__ == "__main__":
    main()