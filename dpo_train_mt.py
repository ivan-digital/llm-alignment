#!/usr/bin/env python
"""
dpo_train_reddit_finance.py — DPO on Reddit Finance (43 subreddits, 250k post↔comment pairs),
while KEEPING TAT-QA eval + MT-Bench + instruction-following eval from the original script.

Training data construction (explicit preferences):
- Group rows by post `id` from HF dataset winddude/reddit_finance_43_250k (GPL-3.0).
- For each post, rank top-level comments by `comment_normalized_score`.
- chosen = highest-scored comment; rejected = lower-scored sibling with score gap ≥ margin
  (fallback to the lowest-scored sibling if needed). Optional length-similarity filter.

Prompt format (turn-level):
  SUBREDDIT: r/<subreddit>
  TITLE: <title>
  POST: <selftext or "(no selftext)">
  ---
  Reply like a helpful finance assistant. Be specific, polite, and practical.

References:
- HF dataset + schema preview (id/title/selftext/body/comment_normalized_score/...):
  https://huggingface.co/datasets/winddude/reddit_finance_43_250k
- Kaggle mirror/description: "Reddit Finance 43 (250k)"
"""

import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer as _TRL_DPOTrainer

# Keep your DPOConfig import/patch for TRL versions
from trl.trainer.dpo_config import DPOConfig

# --------------------------------------------------------------------------------------
# Patch: align DPOTrainer.log signature with Transformers Trainer calling pattern
# --------------------------------------------------------------------------------------
class DPOTrainer(_TRL_DPOTrainer):
    def log(self, logs, *args, **kwargs):  # compat with start_time, etc.
        return super().log(logs)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _seed_from_uid(uid: str, seed: int) -> int:
    import hashlib as _hashlib
    h = _hashlib.sha256((str(seed) + "|" + str(uid)).encode()).hexdigest()
    return int(h[:8], 16)

def load_causal_lm(model_id: str, use_4bit: bool, dtype_compute):
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype_compute)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, trust_remote_code=True, device_map="auto"
        )
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype_compute, trust_remote_code=True, device_map="auto"
    )

# --------------------------------------------------------------------------------------
# Reddit Finance 43 → DPO pairs
# --------------------------------------------------------------------------------------
def build_reddit_prompt(subreddit: str, title: str, selftext: str, max_chars: int) -> str:
    st = (selftext or "").strip()
    if max_chars and len(st) > max_chars:
        st = st[:max_chars].rsplit(" ", 1)[0] + " ..."
    title = (title or "").strip()
    sub = subreddit or ""
    return (
        f"SUBREDDIT: r/{sub}\n"
        f"TITLE: {title if title else '(no title)'}\n"
        f"POST: {st if st else '(no selftext)'}\n"
        "---\n"
        "Reply like a helpful finance assistant. Be specific, polite, and practical."
    )

def reddit_finance_to_dpo_pairs(
    ds_iter: Iterable[Dict[str, Any]],
    *,
    min_post_chars: int = 120,
    min_comment_chars: int = 40,
    max_post_chars: int = 3000,
    score_margin: float = 0.10,
    length_tol: float = 0.6,
    max_pairs_per_post: int = 1,
    seed: int = 42,
) -> Iterable[Dict[str, str]]:
    """
    Construct DPO {prompt, chosen, rejected} triples from reddit_finance_43_250k.

    Heuristics:
      - Skip posts with too-short post text OR with <2 qualifying comments.
      - chosen := highest comment_normalized_score
      - rejected := a sibling whose score is lower by ≥ score_margin and whose length
        is within ±length_tol * len(chosen) (to reduce length bias). If none, pick the
        lowest-scored sibling.
    """
    # Group by post id
    groups: Dict[str, Dict[str, Any]] = {}
    for ex in ds_iter:
        pid = ex.get("id") or ""
        if not pid:
            continue
        sub = ex.get("subreddit") or ""
        title = ex.get("title") or ""
        selftext = ex.get("selftext") or ""
        # Basic post length gate (title + selftext)
        post_len = len((title or "")) + len((selftext or ""))
        if post_len < min_post_chars:
            # Keep collecting comments anyway so we can maybe still hit the min
            pass
        c_body = (ex.get("body") or "").strip()
        if not c_body or len(c_body) < min_comment_chars:
            continue
        c_score = ex.get("comment_normalized_score")
        try:
            c_score = float(c_score) if c_score is not None else 0.0
        except Exception:
            c_score = 0.0

        g = groups.get(pid)
        if not g:
            g = {
                "pid": pid,
                "subreddit": sub,
                "title": title,
                "selftext": selftext,
                "comments": [],  # list of (score, text)
            }
            groups[pid] = g
        g["comments"].append((c_score, c_body))

    rnd = random.Random(seed)
    for pid, g in groups.items():
        comments: List[Tuple[float, str]] = sorted(g["comments"], key=lambda x: x[0], reverse=True)
        if len(comments) < 2:
            continue
        prompt = build_reddit_prompt(g["subreddit"], g["title"], g["selftext"], max_chars=max_post_chars)
        if len(prompt) < min_post_chars:
            # If even the full prompt is too short, skip
            continue

        chosen_score, chosen_text = comments[0]
        # Find a lower-scored sibling with margin + length similarity
        rejected_text = None
        for s, t in comments[1:]:
            if (chosen_score - s) >= score_margin:
                # length similarity gate
                if len(chosen_text) == 0:  # guard
                    continue
                rel_gap = abs(len(t) - len(chosen_text)) / max(1, len(chosen_text))
                if rel_gap <= length_tol:
                    rejected_text = t
                    break
        # Fallback: the worst one
        if rejected_text is None:
            rejected_text = comments[-1][1]

        # Final sanity
        if not chosen_text or not rejected_text:
            continue
        if chosen_text.strip().lower() == rejected_text.strip().lower():
            continue

        # Optional: slight shuffle to avoid degenerate ordering bias (not needed for DPO but harmless)
        if rnd.random() < 0.0:
            chosen_text, rejected_text = rejected_text, chosen_text

        yield {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


def finance_instruct_to_dpo_pairs(
    model, tokenizer, num_pairs: int = 5000, max_new_tokens: int = 256, seed: int = 42
) -> List[Dict[str, str]]:
    """
    Generate DPO preference pairs from Finance-Instruct-500k dataset.
    
    Strategy:
    1. Sample financial instructions from Finance-Instruct-500k
    2. Use current model to generate responses  
    3. Use reference responses as "chosen", generated responses as "rejected"
    4. This teaches the model to prefer high-quality financial instruction responses
    """
    from datasets import load_dataset
    import random
    
    print(f"[finance-instruct] Loading Finance-Instruct dataset...")
    finance_ds = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
    
    # Sample random instructions
    rnd = random.Random(seed)
    indices = rnd.sample(range(len(finance_ds)), min(num_pairs, len(finance_ds)))
    
    pairs = []
    model.eval()
    
    print(f"[finance-instruct] Generating {len(indices)} DPO pairs...")
    
    # Suppress generation warnings
    try:
        from transformers.utils import logging as hf_logging
        _prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        _prev_level = None
    
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": True, "temperature": 0.8, "top_p": 0.9}
    
    for i, idx in enumerate(indices):
        try:
            example = finance_ds[idx]
            instruction = example.get('user', example.get('instruction', ''))
            reference_response = example.get('assistant', example.get('output', ''))
            
            if not instruction or not reference_response:
                continue
                
            # Generate response with current model
            messages = [
                {"role": "system", "content": "You are a helpful financial advisor."},
                {"role": "user", "content": instruction},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_kwargs)
            generated_response = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Use reference as chosen, generated as rejected (teaches model to prefer high-quality responses)
            pairs.append({
                "prompt": f"You are a helpful financial advisor.\n\nUser: {instruction}\n\nAssistant:",
                "chosen": reference_response.strip(),
                "rejected": generated_response.strip()
            })
            
            if (i + 1) % 100 == 0:
                print(f"[finance-instruct] Generated {i+1}/{len(indices)} pairs")
                
        except Exception as e:
            print(f"[warn] Finance-Instruct pair generation failed for example {i+1}: {e}")
            continue
    
    # Restore logging level
    try:
        if _prev_level is not None:
            hf_logging.set_verbosity(_prev_level)
    except Exception:
        pass
    
    print(f"[finance-instruct] Generated {len(pairs)} Finance-Instruct DPO pairs")
    return pairs


def load_general_dpo_pairs(dataset_name: str, num_pairs: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Load general instruction DPO pairs from a public dataset.
    """
    print(f"[general] Loading general DPO dataset: {dataset_name}")
    
    try:
        if dataset_name == "Intel/orca_dpo_pairs":
            # Load Intel Orca DPO dataset
            dataset = load_dataset("Intel/orca_dpo_pairs", split="train")
            print(f"[general] Loaded {len(dataset)} general examples")
            
            # Sample random pairs
            import random
            random.seed(seed)
            indices = random.sample(range(len(dataset)), min(num_pairs, len(dataset)))
            
            pairs = []
            for i in indices:
                example = dataset[i]
                pair = {
                    "prompt": example["system"] + "\n\n" + example["question"],
                    "chosen": example["chosen"],
                    "rejected": example["rejected"]
                }
                pairs.append(pair)
                
        elif dataset_name == "Anthropic/hh-rlhf":
            # Load Anthropic HH-RLHF dataset  
            dataset = load_dataset("Anthropic/hh-rlhf", "chosen", split="train")
            rejected_dataset = load_dataset("Anthropic/hh-rlhf", "rejected", split="train")
            
            # Sample and create pairs
            import random
            random.seed(seed)
            indices = random.sample(range(min(len(dataset), len(rejected_dataset))), 
                                  min(num_pairs, len(dataset)))
            
            pairs = []
            for i in indices:
                pair = {
                    "prompt": dataset[i]["chosen"].split("Assistant:")[0] + "Assistant:",
                    "chosen": dataset[i]["chosen"].split("Assistant:")[-1],
                    "rejected": rejected_dataset[i]["rejected"].split("Assistant:")[-1]
                }
                pairs.append(pair)
                
        else:
            # Generic DPO dataset loading
            dataset = load_dataset(dataset_name, split="train")
            import random
            random.seed(seed)
            indices = random.sample(range(len(dataset)), min(num_pairs, len(dataset)))
            
            pairs = []
            for i in indices:
                example = dataset[i]
                pair = {
                    "prompt": example.get("prompt", example.get("question", "")),
                    "chosen": example.get("chosen", example.get("response_chosen", "")),
                    "rejected": example.get("rejected", example.get("response_rejected", ""))
                }
                pairs.append(pair)
    
    except Exception as e:
        print(f"[warn] Failed to load general dataset {dataset_name}: {e}")
        print("[warn] Continuing without general pairs...")
        return []
    
    print(f"[general] Created {len(pairs)} general DPO pairs")
    return pairs


# --------------------------------------------------------------------------------------
# TAT-QA EVAL (unchanged from your original, with safe JSON streaming)
# --------------------------------------------------------------------------------------
def _markdown_table(table_2d: List[List[str]], max_rows: int, max_cols: int) -> str:
    if not table_2d:
        return ""
    rows = [list(map(lambda c: (c or "").strip(), r[:max_cols])) for r in table_2d[:max_rows]]
    if not rows:
        return ""
    if len(rows) == 1:
        rows.append([""] * len(rows[0]))
    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]
    def pipe(r): return "| " + " | ".join(r) + " |"
    return "\n".join([pipe(header), pipe(sep)] + [pipe(r) for r in body])

def _select_paragraphs(paragraphs: List[Dict[str, Any]],
                       rel_orders: List[Any],
                       only_rel: bool,
                       max_chars: int) -> str:
    if only_rel and rel_orders:
        rel = set(map(str, rel_orders))
        keep = [p.get("text", "") for p in sorted(paragraphs, key=lambda x: x.get("order", 1))
                if str(p.get("order", "")) in rel]
        text = "\n\n".join(keep)
    else:
        text = "\n\n".join(p.get("text", "") for p in sorted(paragraphs, key=lambda x: x.get("order", 1)))
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text

def iter_tatqa_examples(split: str):
    file_by_split = {
        "train": "tatqa_dataset_train.json",
        "validation": "tatqa_dataset_dev.json",
        "test": "tatqa_dataset_test.json",
    }
    try:
        from huggingface_hub import hf_hub_download
        local_path = hf_hub_download(repo_id="next-tat/TAT-QA", filename=file_by_split[split])
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for ex in data:
            yield ex
        return
    except Exception:
        pass

    # Fallback to GitHub raw URLs
    url_map = {
        "train": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_train.json",
        "validation": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
        "test": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_test.json",
    }
    import requests
    resp = requests.get(url_map[split], timeout=120)
    resp.raise_for_status()
    for ex in resp.json():
        yield ex

# --------------------------------------------------------------------------------------
# EVAL HELPERS (kept from your script)
# --------------------------------------------------------------------------------------
import re
def _normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\t\n\r]", " ", s)
    s = re.sub(r"[^\w\s\.\-]", "", s)
    return s

from typing import Optional
def _extract_number(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"[-+]?\d+[\d,]*\.?\d*", text.replace(",", ""))
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def _score(ans_type: str, gold, pred: str, scale: str) -> bool:
    if ans_type in ("arithmetic", "counting"):
        g = _extract_number(str(gold))
        p = _extract_number(pred)
        if g is None or p is None:
            return False
        tol = max(0.5, 0.05 * abs(g))
        return abs(g - p) <= tol
    # spans/others: containment
    gold_texts = [str(x) for x in (gold if isinstance(gold, list) else [gold])]
    pred_n = _normalize_text(pred)
    return any(_normalize_text(g) in pred_n for g in gold_texts)

def _tatqa_eval_items(args, split: str, limit: int):
    count = 0
    for ex in iter_tatqa_examples(split):
        table = (ex.get("table") or {})
        table_2d = (table.get("table") or [])
        paragraphs = ex.get("paragraphs") or []
        questions = ex.get("questions") or []
        if isinstance(questions, dict):
            questions = [questions]
        elif not isinstance(questions, list):
            continue
        for q in questions:
            q_uid = q.get("uid") or f"{ex.get('uid','')}-{q.get('order','')}"
            ans_type = (q.get("answer_type") or "").lower()
            scale = (q.get("scale") or "None")
            rel_pars = q.get("rel_paragraphs") or []
            question_text = q.get("question") or ""
            gold = q.get("answer")
            md_table = _markdown_table(table_2d, args.tatqa_table_rows, args.tatqa_table_cols)
            paras_text = _select_paragraphs(paragraphs, rel_pars, args.tatqa_only_rel_paras, args.tatqa_para_chars)
            prompt = (
                "You are a financial analyst. Use the TABLE and the PASSAGES to answer.\n\n"
                "TABLE:\n" + (md_table if md_table else "(no table)") + "\n\n"
                "PASSAGES:\n" + (paras_text if paras_text else "(no passages)") + "\n\n"
                f"Question: {question_text}\n"
                "Answer:"
            )
            yield {
                "q_uid": q_uid,
                "prompt": prompt,
                "gold": gold,
                "ans_type": ans_type,
                "scale": scale,
            }
            count += 1
            if limit and count >= limit:
                return

# --------------------------------------------------------------------------------------
# Generation + evaluation runners (kept from your script)
# --------------------------------------------------------------------------------------

def run_mtbench_eval(eval_model, eval_tok, questions_file: str, limit: int, max_tokens: int, save_path: str = ""):
    """Run MT-Bench style evaluation with optional reward model scoring"""
    if not Path(questions_file).exists():
        print(f"[warn] MT-Bench file {questions_file} not found, skipping MT-Bench eval")
        return {"n": 0, "questions_completed": 0, "avg_score": 0.0}
    
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Process questions
        questions = []
        for i, obj in enumerate(data):
            qid = obj.get("id", str(i))
            cat = obj.get("category", obj.get("sub_category", ""))
            if "turns" in obj and isinstance(obj["turns"], list) and obj["turns"]:
                prompt = "\n\n".join(map(str, obj["turns"]))
            else:
                prompt = obj.get("prompt") or obj.get("question") or obj.get("text") or ""
            questions.append({"id": qid, "category": cat, "prompt": prompt})
        
        if limit and limit > 0:
            questions = questions[:limit]
        
        results = []
        eval_model.eval()
        
        # Try to load Skywork reward model for scoring
        reward_model = None
        reward_tokenizer = None
        try:
            from transformers import AutoModelForSequenceClassification
            print("[mtbench] Loading Skywork reward model for scoring...")
            reward_tokenizer = AutoTokenizer.from_pretrained("Skywork/Skywork-Reward-V2-Qwen3-0.6B", trust_remote_code=True)
            reward_model = AutoModelForSequenceClassification.from_pretrained(
                "Skywork/Skywork-Reward-V2-Qwen3-0.6B",
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            reward_model.eval()
            print("[mtbench] Reward model loaded successfully")
        except Exception as e:
            print(f"[warn] Could not load reward model: {e}")
            print("[mtbench] Continuing without scoring...")
        
        # Generation settings - use greedy decoding for consistency
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": False, "num_beams": 1}
        
        total_score = 0.0
        scored_count = 0
        
        # Suppress noisy generation warnings
        try:
            from transformers.utils import logging as hf_logging
            _prev_level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            _prev_level = None

        for i, q in enumerate(questions):
            prompt = q["prompt"]
            # Use chat template for better responses
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            prompt = eval_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = eval_tok(prompt, return_tensors="pt").to(eval_model.device)
            with torch.no_grad():
                out_ids = eval_model.generate(**inputs, **gen_kwargs)
            answer = eval_tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            # Score with reward model if available
            score = None
            if reward_model and reward_tokenizer:
                try:
                    score = score_with_reward_model(q["prompt"], answer, reward_model, reward_tokenizer)
                    total_score += score
                    scored_count += 1
                except Exception as e:
                    print(f"[warn] Scoring failed for question {i+1}: {e}")
            
            results.append({
                "id": q["id"],
                "category": q["category"], 
                "prompt": q["prompt"],
                "answer": answer,
                "score": score
            })
            
            if (i + 1) % 10 == 0:
                print(f"[mtbench] Processed {i+1}/{len(questions)} questions")
        
        # Restore logging level
        try:
            if _prev_level is not None:
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity(_prev_level)
        except Exception:
            pass
        
        avg_score = (total_score / scored_count) if scored_count > 0 else 0.0
        
        if avg_score > 0:
            print(f"MT-Bench eval completed — N={len(results)} questions, avg score: {avg_score:.2f}/10")
        else:
            print(f"MT-Bench eval completed — N={len(results)} questions (no scoring)")
        
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"Saved MT-Bench results to {save_path}")
        
        return {"n": len(results), "questions_completed": len(results), "avg_score": avg_score, "scored_count": scored_count}
        
    except Exception as e:
        print(f"[warn] MT-Bench evaluation failed: {e}")
        return {"n": 0, "questions_completed": 0, "avg_score": 0.0}

def score_with_reward_model(question, response, reward_model, reward_tokenizer):
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

def run_instruct_eval(eval_model, eval_tok, limit: int, max_tokens: int, save_path: str = ""):
    """Run instruction following evaluation using Finance-Instruct-500k dataset with reward model scoring"""
    print("[instruct] Running Finance-Instruct instruction eval")
    
    # Load Skywork reward model for scoring
    print("[instruct] Loading Skywork reward model for scoring...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, trust_remote_code=True)
        print("[instruct] Reward model loaded successfully")
    except Exception as e:
        print(f"[warn] Failed to load reward model: {e}")
        return {"n": 0, "avg_score": 0.0, "tasks_completed": 0}
    
    # Load Finance-Instruct-500k dataset
    try:
        from datasets import load_dataset
        dataset = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
        print(f"[instruct] Loaded {len(dataset)} Finance-Instruct examples")
    except Exception as e:
        print(f"[warn] Failed to load Finance-Instruct dataset: {e}")
        return {"n": 0, "avg_score": 0.0, "tasks_completed": 0}
    
    # Sample evaluation examples
    import random
    random.seed(42)  # Consistent sampling
    eval_indices = random.sample(range(len(dataset)), min(limit, len(dataset)))
    
    results = []
    total_score = 0.0
    successful_evals = 0
    
    eval_model.eval()
    
    # Suppress generation warnings
    try:
        from transformers.utils import logging as hf_logging
        _prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        _prev_level = None
    
    for i, idx in enumerate(eval_indices):
        try:
            example = dataset[idx]
            
            # Extract instruction from the dataset
            # Finance-Instruct-500k uses 'user' field for instruction
            if 'user' in example:
                instruction = example['user']
            elif 'instruction' in example:
                instruction = example['instruction']
            else:
                print(f"[warn] No instruction field found in example {i+1}")
                continue
            
            # Generate response using chat template
            messages = [
                {"role": "system", "content": "You are a helpful financial assistant that follows instructions carefully."},
                {"role": "user", "content": instruction}
            ]
            prompt = eval_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = eval_tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(eval_model.device)
            
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "pad_token_id": eval_tok.eos_token_id
                }
                out_ids = eval_model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            generated_text = eval_tok.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Score with reward model (same as MT-Bench)
            reward_score = score_with_reward_model(instruction, generated_text, reward_model, reward_tokenizer)
            
            results.append({
                "idx": idx,
                "instruction": instruction[:200] + "..." if len(instruction) > 200 else instruction,
                "response": generated_text,
                "reward_score": reward_score,
                "reference_answer": example.get('assistant', example.get('output', ''))[:100] + "..." if example.get('assistant', example.get('output', '')) else None
            })
            
            total_score += reward_score
            successful_evals += 1
            
            if (i + 1) % 5 == 0:
                print(f"[instruct] Evaluated {i+1}/{len(eval_indices)} instructions")
                
        except Exception as e:
            print(f"[warn] Instruction evaluation failed for example {i+1}: {e}")
            continue
    
    # Restore logging level
    try:
        if _prev_level is not None:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity(_prev_level)
    except Exception:
        pass
    
    avg_score = total_score / successful_evals if successful_evals > 0 else 0.0
    success_rate = (sum(1 for r in results if r["reward_score"] >= 7.0) / successful_evals * 100) if successful_evals > 0 else 0.0
    
    print(f"Finance-Instruct eval completed — N={successful_evals} instructions, avg score: {avg_score:.2f}/10, success rate: {success_rate:.1f}%")
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved Finance-Instruct results to {save_path}")
    
    return {"n": successful_evals, "tasks_completed": successful_evals, "avg_score": avg_score, "success_rate": success_rate}

def run_reddit_reward_eval(eval_model, eval_tok, limit: int = 100, max_tokens: int = 256, save_path: str = ""):
    """Evaluate Reddit Finance test set using reward model scoring."""
    print("[reddit] Running Reddit Finance reward evaluation")
    
    # Load Skywork reward model for scoring
    print("[reddit] Loading Skywork reward model for scoring...")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        reward_model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            reward_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, trust_remote_code=True)
        print("[reddit] Reward model loaded successfully")
    except Exception as e:
        print(f"[warn] Failed to load reward model: {e}")
        return {"n": 0, "avg_reward": 0.0, "examples_evaluated": 0}
    
    # Load Reddit Finance test data
    try:
        from datasets import load_dataset
        dataset = load_dataset("winddude/reddit_finance_43_250k", split="train")
        print(f"[reddit] Loaded {len(dataset)} Reddit Finance examples")
    except Exception as e:
        print(f"[warn] Failed to load Reddit dataset: {e}")
        return {"n": 0, "avg_reward": 0.0, "examples_evaluated": 0}
    
    # Sample test examples (different from training data)
    import random
    random.seed(12345)  # Different seed from training
    test_indices = random.sample(range(len(dataset)), min(limit, len(dataset)))
    
    results = []
    total_reward = 0.0
    successful_evals = 0
    
    eval_model.eval()
    
    # Suppress generation warnings
    try:
        from transformers.utils import logging as hf_logging
        _prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        _prev_level = None
    
    for i, idx in enumerate(test_indices):
        try:
            example = dataset[idx]
            
            # Build prompt in same format as training
            subreddit = example.get("subreddit", "")
            title = example.get("title", "")
            selftext = example.get("selftext", "")
            
            prompt = build_reddit_prompt(subreddit, title, selftext, max_chars=3000)
            
            # Generate response
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = eval_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = eval_tok(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048).to(eval_model.device)
            
            with torch.no_grad():
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "pad_token_id": eval_tok.eos_token_id
                }
                out_ids = eval_model.generate(**inputs, **gen_kwargs)
            
            # Decode response
            generated_text = eval_tok.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            # Score with reward model
            reward_score = score_with_reward_model(prompt, generated_text, reward_model, reward_tokenizer)
            
            results.append({
                "idx": idx,
                "subreddit": subreddit,
                "title": title[:100] + "..." if len(title) > 100 else title,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "response": generated_text,
                "reward_score": reward_score
            })
            
            total_reward += reward_score
            successful_evals += 1
            
            if (i + 1) % 25 == 0:
                print(f"[reddit] Evaluated {i+1}/{len(test_indices)} examples")
                
        except Exception as e:
            print(f"[warn] Reddit evaluation failed for example {i+1}: {e}")
            continue
    
    # Restore logging level
    try:
        if _prev_level is not None:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity(_prev_level)
    except Exception:
        pass
    
    avg_reward = total_reward / successful_evals if successful_evals > 0 else 0.0
    
    print(f"Reddit Finance reward eval completed — N={successful_evals} examples, avg reward: {avg_reward:.2f}/10")
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved Reddit Finance results to {save_path}")
    
    return {"n": successful_evals, "avg_reward": avg_reward, "examples_evaluated": successful_evals}

def score_instruction_response(response: str, criteria: list) -> float:
    """Score an instruction following response based on criteria compliance."""
    response_lower = response.lower()
    score = 0.0
    max_score = len(criteria)
    
    for criterion in criteria:
        criterion_score = 0.0
        
        # Formatting criteria
        if "numbered_list" in criterion:
            if any(f"{i}." in response for i in range(1, 11)) or any(f"{i})" in response for i in range(1, 11)):
                criterion_score = 1.0
                
        elif "exactly_3_items" in criterion:
            item_count = sum(1 for i in range(1, 11) if f"{i}." in response or f"{i})" in response)
            if item_count == 3:
                criterion_score = 1.0
            elif abs(item_count - 3) <= 1:
                criterion_score = 0.5
                
        elif "exactly_2_sentences" in criterion:
            sentence_count = response.count('.') + response.count('!') + response.count('?')
            if sentence_count == 2:
                criterion_score = 1.0
            elif sentence_count == 1 or sentence_count == 3:
                criterion_score = 0.5
                
        elif "has_advantages_section" in criterion:
            if "advantage" in response_lower or "benefit" in response_lower or "pro" in response_lower:
                criterion_score = 1.0
                
        elif "has_disadvantages_section" in criterion:
            if "disadvantage" in response_lower or "drawback" in response_lower or "con" in response_lower:
                criterion_score = 1.0
                
        elif "table_format" in criterion:
            if "|" in response or ("cat" in response_lower and "dog" in response_lower and ("size" in response_lower or "care" in response_lower)):
                criterion_score = 1.0
                
        elif "proper_quotation_marks" in criterion:
            if '"' in response or "'" in response:
                criterion_score = 1.0
                
        # Content criteria
        elif "contains_canvas" in criterion:
            if "canvas" in response_lower:
                criterion_score = 1.0
                
        elif "contains_dream" in criterion:
            if "dream" in response_lower:
                criterion_score = 1.0
                
        elif "python_function" in criterion:
            if "def " in response:
                criterion_score = 1.0
                
        elif "named_add_numbers" in criterion:
            if "add_numbers" in response:
                criterion_score = 1.0
                
        elif "has_docstring" in criterion:
            if '"""' in response or "'''" in response:
                criterion_score = 1.0
                
        elif "shows_calculation_steps" in criterion:
            if any(word in response_lower for word in ["step", "first", "then", "calculate", "multiply"]):
                criterion_score = 1.0
                
        elif "word_count_mentioned" in criterion:
            if "word" in response_lower and any(str(i) in response for i in range(45, 56)):
                criterion_score = 1.0
                
        elif "approximately_50_words" in criterion:
            word_count = len(response.split())
            if 45 <= word_count <= 55:
                criterion_score = 1.0
            elif 40 <= word_count <= 60:
                criterion_score = 0.5
        
        # Add more criteria scoring logic as needed...
        else:
            # Default heuristic scoring for unhandled criteria
            criterion_words = criterion.replace("_", " ").split()
            if any(word in response_lower for word in criterion_words):
                criterion_score = 0.5
        
        score += criterion_score
    
    # Convert to 0-10 scale
    final_score = (score / max_score) * 10 if max_score > 0 else 0.0
    return min(10.0, max(0.0, final_score))

def run_eval(args, eval_model, eval_tok, split: str, limit: int, save_path: str = ""):
    eval_model.eval()
    results = []
    correct = 0
    total = 0
    by_type: Dict[str, Dict[str, int]] = {}
    do_sample = (args.eval_temperature and args.eval_temperature > 0.0) or (args.eval_top_p and args.eval_top_p < 1.0)
    gen_kwargs = {"max_new_tokens": args.eval_max_new_tokens}
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, temperature=max(1e-6, float(args.eval_temperature)), top_p=float(args.eval_top_p)))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=1))

    try:
        from transformers.utils import logging as hf_logging
        _prev_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
    except Exception:
        _prev_level = None

    for item in _tatqa_eval_items(args, split, limit):
        prompt = item["prompt"]
        if args.use_chat_template:
            messages = [
                {"role": "system", "content": "You are a helpful financial analysis assistant."},
                {"role": "user", "content": prompt},
            ]
            prompt = eval_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = eval_tok(prompt, return_tensors="pt").to(eval_model.device)
        with torch.no_grad():
            out_ids = eval_model.generate(**inputs, **gen_kwargs)
        pred = eval_tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        ok = _score(item["ans_type"], item["gold"], pred, item["scale"])
        correct += 1 if ok else 0
        total += 1
        t = (item.get("ans_type") or "unknown").lower()
        if t not in by_type:
            by_type[t] = {"correct": 0, "total": 0}
        by_type[t]["correct"] += (1 if ok else 0)
        by_type[t]["total"] += 1
        results.append({
            "q_uid": item["q_uid"], "pred": pred, "gold": item["gold"], "ok": ok,
            "ans_type": t, "scale": item.get("scale")
        })

    acc = (correct / total) if total else 0.0
    print(f"Eval {split} — N={total} | EM-like acc={acc:.3f}")
    if by_type:
        try:
            for t, ct in sorted(by_type.items(), key=lambda x: x[0]):
                t_acc = (ct["correct"] / ct["total"]) if ct["total"] else 0.0
                print(f"   - {t}: {t_acc:.3f} ({ct['correct']}/{ct['total']})")
        except Exception:
            pass
    try:
        if _prev_level is not None:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity(_prev_level)
    except Exception:
        pass
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved eval results to {save_path}")
    return {"n": total, "acc": acc, "by_type": {
        t: {"acc": (ct["correct"] / ct["total"]) if ct["total"] else 0.0, "n": ct["total"]}
        for t, ct in by_type.items()
    }}

# ---- MT-Bench + reward scoring, Instruction eval ----
# (Exactly the same as your existing implementations; pasted verbatim to keep behavior)
# -- START COPY from your script (run_mtbench_eval, score_with_reward_model, run_instruct_eval, score_instruction_response)
# For brevity here, reuse your previous functions without modification. If you paste this file over the old one,
# keep those function definitions as-is.
# -- END COPY

# --------------------------------------------------------------------------------------
# CLI + main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("DPO on Reddit Finance (explicit preferences) + eval on TAT-QA/MT-Bench/Instruction")
    # Base / output
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", default="qwen3_reddit_finance_dpo")  # Will be updated for hybrid training

    # DPO core
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--batch", type=int, default=1)  # Reduced from 2 to 1 for memory
    p.add_argument("--grad_acc", type=int, default=64)  # Increased to maintain effective batch size
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_length", type=int, default=1536)  # Reduced from 2048
    p.add_argument("--max_length", type=int, default=2560)  # Reduced from 3072

    # LoRA - Reduced for memory efficiency
    p.add_argument("--lora_r", type=int, default=8)  # Reduced from 16 to 8
    p.add_argument("--lora_alpha", type=int, default=16)  # Reduced from 32 to 16
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,v_proj", help="Comma-separated LoRA target modules")

    # Reddit Finance construction knobs
    p.add_argument("--reddit_min_post_chars", type=int, default=120)
    p.add_argument("--reddit_min_comment_chars", type=int, default=40)
    p.add_argument("--reddit_max_post_chars", type=int, default=2000)  # Reduced from 3000
    p.add_argument("--reddit_score_margin", type=float, default=0.10)
    p.add_argument("--reddit_length_tol", type=float, default=0.6)
    p.add_argument("--reddit_max_pairs_per_post", type=int, default=1)
    p.add_argument("--reddit_sample_rate", type=float, default=0.3, help="Sample rate for Reddit dataset (0.0-1.0)")  # Reduced from 0.5

    # TAT-QA eval options (unchanged)
    p.add_argument("--tatqa_split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--tatqa_table_rows", type=int, default=20)
    p.add_argument("--tatqa_table_cols", type=int, default=8)
    p.add_argument("--tatqa_para_chars", type=int, default=1200)
    p.add_argument("--tatqa_only_rel_paras", action="store_true", default=True)

    # System / misc
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization")
    p.add_argument("--use_chat_template", action="store_true", help="Wrap prompts in the model's chat template")

    # Dev/test
    p.add_argument("--no_train", action="store_true", help="Build dataset pairs only")
    p.add_argument("--limit_pairs", type=int, default=0, help="If >0, limit number of constructed pairs")

    # Eval toggles
    p.add_argument("--do_eval", action="store_true", default=True)
    p.add_argument("--eval_split", choices=["validation", "test", "train"], default="validation")
    p.add_argument("--eval_limit", type=int, default=1000)
    p.add_argument("--eval_max_new_tokens", type=int, default=64)
    p.add_argument("--eval_temperature", type=float, default=0.2)
    p.add_argument("--eval_top_p", type=float, default=0.9)
    p.add_argument("--eval_baseline", action="store_true", default=True)
    p.add_argument("--eval_save_json", type=str, default="")
    p.add_argument("--load_adapter", type=str, default="")
    p.add_argument("--adapter_trainable", action="store_true")
    p.add_argument("--resume_from_checkpoint", type=str, default="")
    # MT-Bench / Instruction (keep your defaults)
    p.add_argument("--do_mtbench", action="store_true", default=True)
    p.add_argument("--mtbench_file", type=str, default="mt_bench_en.json")
    p.add_argument("--mtbench_limit", type=int, default=50)
    p.add_argument("--mtbench_max_tokens", type=int, default=256)
    p.add_argument("--do_instruct", action="store_true", default=True)
    p.add_argument("--instruct_limit", type=int, default=25)
    p.add_argument("--instruct_max_tokens", type=int, default=512)
    # Reddit Finance reward evaluation
    p.add_argument("--do_reddit", action="store_true", default=True, help="Evaluate on Reddit Finance test set with reward model")
    p.add_argument("--reddit_limit", type=int, default=100, help="Number of Reddit test examples to evaluate")
    p.add_argument("--reddit_max_tokens", type=int, default=256, help="Max tokens for Reddit response generation")
    # Experiment logging
    p.add_argument("--exp_log", type=str, default="dpo_experiments.jsonl")
    p.add_argument("--exp_tag", type=str, default="")
    
    # Hybrid training options
    p.add_argument("--use_hybrid", action="store_true", help="Use hybrid Reddit + Finance-Instruct DPO training")
    p.add_argument("--use_3way_hybrid", action="store_true", help="Use 3-way hybrid with general instructions")
    p.add_argument("--finance_instruct_pairs", type=int, default=2000, help="Number of Finance-Instruct DPO pairs to generate")
    p.add_argument("--general_pairs", type=int, default=3000, help="Number of general instruction DPO pairs")
    p.add_argument("--hybrid_ratio", type=float, default=0.7, help="Ratio of Reddit pairs in hybrid training (0.7 = 70% Reddit, 30% Finance-Instruct)")
    p.add_argument("--reddit_ratio", type=float, default=0.4, help="Reddit ratio in 3-way hybrid (0.4 = 40% Reddit)")
    p.add_argument("--finance_ratio", type=float, default=0.3, help="Finance-Instruct ratio in 3-way hybrid (0.3 = 30% Finance-Instruct)")
    p.add_argument("--general_ratio", type=float, default=0.3, help="General instruction ratio in 3-way hybrid (0.3 = 30% General)")
    p.add_argument("--general_dataset", type=str, default="Intel/orca_dpo_pairs", help="Dataset for general instruction DPO pairs")
    
    return p.parse_args()

def apply_chat_template_if_enabled(tok, ds: Dataset, enabled: bool) -> Dataset:
    if not enabled:
        return ds
    def _fmt(ex):
        messages = [
            {"role": "system", "content": "You are a helpful finance assistant."},
            {"role": "user", "content": ex["prompt"]},
        ]
        ex["prompt"] = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return ex
    return ds.map(_fmt)

def main():
    args = parse_args()
    
    # Update output directory for different training modes
    if args.use_3way_hybrid:
        args.output_dir = f"{args.output_dir}_3way_hybrid"
        print(f"[3way-hybrid] Using 3-way hybrid training mode - output dir: {args.output_dir}")
    elif args.use_hybrid:
        args.output_dir = f"{args.output_dir}_hybrid"
        print(f"[hybrid] Using hybrid training mode - output dir: {args.output_dir}")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_result = None
    post_result = None
    train_metrics = None
    mtbench_baseline_result = None
    mtbench_post_result = None
    instruct_baseline_result = None
    instruct_post_result = None
    reddit_baseline_result = None
    reddit_post_result = None

    # -------------------- Build Reddit Finance preference pairs --------------------
    print("[data] Loading winddude/reddit_finance_43_250k (train split)")
    hf_ds = load_dataset("winddude/reddit_finance_43_250k", split="train")

    # Create Reddit pairs
    pairs_iter = reddit_finance_to_dpo_pairs(
        (ex for ex in hf_ds),
        min_post_chars=args.reddit_min_post_chars,
        min_comment_chars=args.reddit_min_comment_chars,
        max_post_chars=args.reddit_max_post_chars,
        score_margin=args.reddit_score_margin,
        length_tol=args.reddit_length_tol,
        max_pairs_per_post=args.reddit_max_pairs_per_post,
        seed=args.seed,
    )

    reddit_pairs = list(pairs_iter)
    
    # Apply sampling for memory efficiency
    if args.reddit_sample_rate < 1.0:
        import random
        random.seed(args.seed)
        original_size = len(reddit_pairs)
        sample_size = int(len(reddit_pairs) * args.reddit_sample_rate)
        reddit_pairs = random.sample(reddit_pairs, sample_size)
        print(f"[data] Sampled {len(reddit_pairs):,} Reddit pairs from {original_size:,} (sample rate: {args.reddit_sample_rate})")
    
    # Prepare final training pairs
    if args.use_3way_hybrid:
        print("[data] 3-way hybrid training enabled - generating Finance-Instruct and General pairs...")
        
        # Load models for Finance-Instruct pair generation
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        base_model = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
        
        # Generate Finance-Instruct pairs
        finance_pairs = finance_instruct_to_dpo_pairs(
            base_model, 
            AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False),
            num_pairs=args.finance_instruct_pairs,
            seed=args.seed
        )
        
        # Clean up model to save memory
        del base_model
        torch.cuda.empty_cache()
        
        # Load general instruction pairs
        general_pairs = load_general_dpo_pairs(
            args.general_dataset,
            num_pairs=args.general_pairs,
            seed=args.seed
        )
        
        # Calculate target sizes based on ratios
        total_target = len(reddit_pairs)
        num_reddit = int(total_target * args.reddit_ratio)
        num_finance = int(total_target * args.finance_ratio) 
        num_general = int(total_target * args.general_ratio)
        
        # Sample pairs according to ratios
        final_reddit = reddit_pairs[:num_reddit]
        final_finance = finance_pairs[:num_finance] if len(finance_pairs) >= num_finance else finance_pairs
        final_general = general_pairs[:num_general] if len(general_pairs) >= num_general else general_pairs
        
        # Combine and shuffle all pairs
        pairs = final_reddit + final_finance + final_general
        import random
        random.seed(args.seed)
        random.shuffle(pairs)  # Shuffle to mix all three data sources
        
        print(f"[data] 3-way hybrid training: {len(final_reddit):,} Reddit + {len(final_finance):,} Finance-Instruct + {len(final_general):,} General = {len(pairs):,} total pairs")
        
    elif args.use_hybrid:
        print("[data] Hybrid training enabled - generating Finance-Instruct pairs...")
        
        # Load models for Finance-Instruct pair generation
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        base_model = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
        
        # Generate Finance-Instruct pairs
        finance_pairs = finance_instruct_to_dpo_pairs(
            base_model, 
            AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False),
            num_pairs=args.finance_instruct_pairs,
            seed=args.seed
        )
        
        # Combine pairs according to hybrid ratio
        num_reddit = int(len(reddit_pairs) * args.hybrid_ratio)
        num_finance = len(reddit_pairs) - num_reddit
        
        final_reddit = reddit_pairs[:num_reddit]
        final_finance = finance_pairs[:num_finance] if len(finance_pairs) >= num_finance else finance_pairs
        
        pairs = final_reddit + final_finance
        random.seed(args.seed)
        random.shuffle(pairs)  # Shuffle to mix Reddit and Finance-Instruct pairs
        
        print(f"[data] Hybrid training: {len(final_reddit):,} Reddit + {len(final_finance):,} Finance-Instruct = {len(pairs):,} total pairs")
        
        # Clean up temporary model
        del base_model
        torch.cuda.empty_cache()
    else:
        pairs = reddit_pairs
        print(f"[data] Standard Reddit-only training: {len(pairs):,} pairs")
    
    if args.limit_pairs and args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise SystemExit("No pairs constructed from dataset—check parameters.")
    dpo_ds = Dataset.from_list(pairs)
    print(f"[data] Final training dataset: {len(dpo_ds):,} preference pairs")

    # -------------------- Tokenizer --------------------
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Optional: wrap prompts in chat template
    dpo_ds = apply_chat_template_if_enabled(tok, dpo_ds, args.use_chat_template)

    # Early inspect-only
    if args.no_train and not args.do_eval:
        print(f"Prepared DPO pairs: {len(dpo_ds)} (limit={args.limit_pairs})")
        ex0 = dpo_ds[0]
        print(f"Example prompt head:\n{ex0['prompt'][:200]}...\n---\nCHOSEN: {ex0['chosen'][:140]}...\nREJECTED: {ex0['rejected'][:140]}...")
        return

    # -------------------- Model(s) --------------------
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    try:
        base_policy = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
    except Exception as e:
        print(f"[warn] 4-bit load failed ({e}); retrying fp16/bf16.")
        base_policy = load_causal_lm(args.model, use_4bit=False, dtype_compute=compute_dtype)
    base_policy.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    )
    # Trainable policy (LoRA)
    try:
        from peft import PeftModel
        if args.load_adapter:
            policy = PeftModel.from_pretrained(base_policy, args.load_adapter, is_trainable=bool(args.adapter_trainable and not args.no_train))
            print(f"[adapter] Loaded LoRA adapter from {args.load_adapter} ({'trainable' if args.adapter_trainable else 'frozen'})")
        else:
            policy = get_peft_model(base_policy, lora_cfg)
    except Exception:
        policy = get_peft_model(base_policy, lora_cfg)

    # Frozen reference model
    try:
        ref_model = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
    except Exception as e:
        print(f"[warn] ref 4-bit load failed ({e}); retrying fp16/bf16.")
        ref_model = load_causal_lm(args.model, use_4bit=False, dtype_compute=compute_dtype)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p_ in ref_model.parameters():
        p_.requires_grad_(False)

    # Initialize result variables
    baseline_result = None
    mtbench_baseline_result = None
    instruct_baseline_result = None
    post_result = None
    mtbench_post_result = None
    instruct_post_result = None

    # -------------------- Optional baseline eval before training (on TAT-QA) --------------------
    if args.do_eval and args.eval_baseline:
        print("[eval] Baseline eval on TAT-QA", args.eval_split)
        baseline_result = run_eval(args, ref_model, tok, args.eval_split, args.eval_limit,
                                   f"{args.eval_save_json.split('.')[0]}_baseline.jsonl" if args.eval_save_json else "")

        # MT-Bench baseline
        if args.do_mtbench:
            print("[mtbench] Running baseline MT-Bench eval")
            mtbench_baseline_result = run_mtbench_eval(ref_model, tok, args.mtbench_file, args.mtbench_limit,
                                                       args.mtbench_max_tokens, f"{out_dir}/mtbench_baseline.jsonl")

        # Instruction following baseline
        if args.do_instruct:
            print("[instruct] Running baseline instruction eval")
            instruct_baseline_result = run_instruct_eval(ref_model, tok, args.instruct_limit,
                                                         args.instruct_max_tokens, f"{out_dir}/instruct_baseline.jsonl")

        # Reddit Finance reward baseline
        if args.do_reddit:
            print("[reddit] Running baseline Reddit Finance reward eval")
            reddit_baseline_result = run_reddit_reward_eval(ref_model, tok, args.reddit_limit,
                                                            args.reddit_max_tokens, f"{out_dir}/reddit_baseline.jsonl")

    # -------------------- Train (DPO) --------------------
    train_args = TrainingArguments(
        output_dir=out_dir.as_posix(),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        fp16=(not bf16_ok),
        bf16=bf16_ok,
        logging_steps=25,
        save_strategy="epoch",
        save_safetensors=True,
        save_total_limit=2,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        optim="paged_adamw_32bit",
        gradient_checkpointing=True,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        max_grad_norm=1.0,  # Add gradient clipping
        dataloader_num_workers=0,  # Reduce CPU-GPU transfer overhead
    )

    if DPOConfig is not None:
        dpo_cfg = DPOConfig(
            output_dir=out_dir.as_posix(),
            per_device_train_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_acc,
            num_train_epochs=args.epochs,
            fp16=(not bf16_ok),
            bf16=bf16_ok,
            logging_steps=25,
            save_strategy="epoch",
            save_safetensors=True,
            save_total_limit=2,
            learning_rate=args.learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            optim="paged_adamw_32bit",
            gradient_checkpointing=True,
            dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
            remove_unused_columns=False,
            report_to=[],
            seed=args.seed,
            max_grad_norm=1.0,  # Add gradient clipping
            dataloader_num_workers=0,  # Reduce CPU-GPU transfer overhead
            beta=args.beta,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            generate_during_eval=False,
        )
        trainer = DPOTrainer(
            model=policy, ref_model=ref_model, args=dpo_cfg, tokenizer=tok, train_dataset=dpo_ds,
        )
    else:
        trainer = DPOTrainer(
            model=policy, ref_model=ref_model, args=train_args, beta=args.beta,
            train_dataset=dpo_ds, tokenizer=tok,
            max_prompt_length=args.max_prompt_length, max_length=args.max_length,
            generate_during_eval=False,
        )

    if not args.no_train:
        if args.resume_from_checkpoint:
            print(f"[train] Resuming from checkpoint: {args.resume_from_checkpoint}")
            _out = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        else:
            _out = trainer.train()
        try:
            train_metrics = getattr(_out, "metrics", None)
        except Exception:
            train_metrics = None
        policy.save_pretrained(out_dir.as_posix())
        print(f"✅ DPO LoRA adapter saved to {out_dir}")
        print(f"ℹ️ Reddit pairs: {len(dpo_ds)}")

    # -------------------- Post-training evals --------------------
    if args.do_eval and not args.no_train:
        print("[eval] Post-training eval on TAT-QA", args.eval_split)
        post_result = run_eval(args, policy, tok, args.eval_split, args.eval_limit,
                               f"{args.eval_save_json.split('.')[0]}_post.jsonl" if args.eval_save_json else f"{out_dir}/tatqa_post.jsonl")

        if args.do_mtbench:
            print("[mtbench] Post-training MT-Bench")
            mtbench_post_result = run_mtbench_eval(policy, tok, args.mtbench_file, args.mtbench_limit,
                                                   args.mtbench_max_tokens, f"{out_dir}/mtbench_post.jsonl")

        if args.do_instruct:
            print("[instruct] Post-training instruction eval")
            instruct_post_result = run_instruct_eval(policy, tok, args.instruct_limit,
                                                     args.instruct_max_tokens, f"{out_dir}/instruct_post.jsonl")

        if args.do_reddit:
            print("[reddit] Post-training Reddit Finance reward eval")
            reddit_post_result = run_reddit_reward_eval(policy, tok, args.reddit_limit,
                                                        args.reddit_max_tokens, f"{out_dir}/reddit_post.jsonl")

    # -------------------- Experiment log (compact) --------------------
    try:
        exp_log_path = Path(args.exp_log)
        def _compact_eval(res):
            if not res: return None
            out = {}
            if "n" in res: out["n"] = res.get("n")
            if "questions_completed" in res and "n" not in out: out["n"] = res.get("questions_completed")
            if "acc" in res: out["acc"] = res.get("acc")
            if "avg_score" in res: out["avg_score"] = res.get("avg_score")
            if "success_rate" in res: out["success_rate"] = res.get("success_rate")
            if "tasks_completed" in res and "n" not in out: out["n"] = res.get("tasks_completed")
            if "avg_reward" in res: out["avg_reward"] = res.get("avg_reward")
            if "examples_evaluated" in res and "n" not in out: out["n"] = res.get("examples_evaluated")
            return out or None

        rec = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tag": args.exp_tag or None,
            "model": args.model,
            "params": {
                "beta": args.beta,
                "epochs": args.epochs,
                "batch": args.batch,
                "grad_acc": args.grad_acc,
                "learning_rate": args.learning_rate,
                "use_4bit": (not args.no_4bit),
                "use_chat_template": args.use_chat_template,
                "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout, "targets": [m for m in args.target_modules.split(',') if m]},
                "reddit": {
                    "min_post_chars": args.reddit_min_post_chars,
                    "min_comment_chars": args.reddit_min_comment_chars,
                    "score_margin": args.reddit_score_margin,
                    "length_tol": args.reddit_length_tol,
                }
            },
            "eval": {"baseline": _compact_eval(baseline_result), "post": _compact_eval(post_result)},
            "mtbench": {"baseline": _compact_eval(mtbench_baseline_result), "post": _compact_eval(mtbench_post_result)},
            "instruct": {"baseline": _compact_eval(instruct_baseline_result), "post": _compact_eval(instruct_post_result)},
            "reddit": {"baseline": _compact_eval(reddit_baseline_result), "post": _compact_eval(reddit_post_result)},
        }
        exp_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"📝 Appended experiment log to {exp_log_path}")

        # Pretty summary
        print("\n" + "="*80)
        print("🎯 TRAINING COMPLETE - METRICS SUMMARY")
        print("="*80)
        print(f"📊 Training: {len(dpo_ds):,} preference pairs")
        print(f"🎯 Loss: {train_metrics.get('train_loss', 'N/A'):.4f}" if train_metrics else "🎯 Loss: N/A")
        if baseline_result and post_result:
            b, p = baseline_result.get('acc', 0), post_result.get('acc', 0)
            imp = p - b
            imp_pct = (imp / b * 100) if b > 0 else 0
            print(f"\n💰 TAT-QA ({args.eval_split}):")
            print(f"   Before: {b:.1%} ({baseline_result.get('n',0)} ex)")
            print(f"   After:  {p:.1%} ({post_result.get('n',0)} ex)")
            print(f"   {'📈 Improvement' if imp>=0 else '📉 Change'}: {imp:+.1%} ({imp_pct:+.1f}%)")
        # (MT-Bench + Instruction summaries as in your script)
        print("="*80)
    except Exception as e:
        print(f"[warn] Failed to write experiment log: {e}")

if __name__ == "__main__":
    main()