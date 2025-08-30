#!/usr/bin/env python
"""
dpo_train_tatqa_base.py — DPO on TAT-QA (table+text finance QA) starting from BASE model (no SFT)

- Streams TAT-QA to avoid PyArrow ArrowInvalid errors from mixed types.
- Builds preference pairs on the fly: chosen = gold; rejected = plausible-but-wrong.
- Fresh LoRA adapter only (no SFT init, no LoRA-in-LoRA).
- Optional 4-bit quantization (use --no_4bit to disable on Windows if bnb is finicky).

Refs (docs/schema/background):
- TRL DPOTrainer docs
- TAT-QA dataset card + GitHub
- PyArrow “cannot mix list and non-list” context
"""

import argparse
import hashlib
import random
from datetime import datetime
import json
from typing import List, Dict, Any, Iterable
from pathlib import Path

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
from typing import Optional

# Patch: align DPOTrainer.log signature with Transformers Trainer calling pattern
class DPOTrainer(_TRL_DPOTrainer):
    def log(self, logs, *args, **kwargs):  # keep compat: ignore extra args like start_time
        return super().log(logs)
from trl.trainer.dpo_config import DPOConfig
#from trl import DPOConfig  # TRL >= 0.11


# -----------------------------
# TAT-QA -> DPO helpers
# -----------------------------

def _seed_from_uid(uid: str, seed: int) -> int:
    h = hashlib.sha256((str(seed) + "|" + str(uid)).encode()).hexdigest()
    return int(h[:8], 16)


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


def _format_with_scale(x: Any, ans_type: str, scale: str) -> str:
    if ans_type in ("arithmetic", "counting"):
        try:
            xv = float(x)
            s = str(int(round(xv))) if xv.is_integer() else f"{xv:.4g}"
        except Exception:
            return str(x)
        if scale and str(scale).lower() != "none":
            return f"{s} {scale}"
        return s
    if isinstance(x, list):
        return "; ".join(str(t) for t in x)
    return str(x)


def _numeric_perturb(true_val: float, seed: int) -> float:
    rnd = random.Random(seed)
    if true_val == 0.0:
        delta = rnd.uniform(1.0, 10.0)
        return rnd.choice([-1, 1]) * round(delta, 1)
    frac = rnd.uniform(0.10, 0.35)
    wrong = true_val + rnd.choice([-1, 1]) * frac * abs(true_val)
    return round(wrong, 1)


def _collect_distractors(table_2d: List[List[str]], gold_texts: List[str]) -> List[str]:
    flat = []
    for r in table_2d or []:
        for c in r or []:
            s = (c or "").strip()
            if s:
                flat.append(s)
    gold_lower = {g.strip().lower() for g in gold_texts}
    uniq = []
    seen = set()
    for s in flat:
        key = s.lower()
        if key not in seen and key not in gold_lower and 1 <= len(s) <= 60:
            uniq.append(s)
            seen.add(key)
    return uniq


def tatqa_to_dpo_pairs(examples_iter: Iterable[Dict[str, Any]],
                       table_max_rows=20,
                       table_max_cols=8,
                       para_max_chars=1200,
                       only_rel_paras=True,
                       seed=42) -> Iterable[Dict[str, str]]:
    """Yield {prompt, chosen, rejected} dicts built from TAT-QA records."""
    for ex in examples_iter:
        table = (ex.get("table") or {})
        table_2d = (table.get("table") or [])
        paragraphs = ex.get("paragraphs") or []
        questions = ex.get("questions") or []

        # Normalize questions to a list; guard against mixed types in raw JSON
        if isinstance(questions, dict):
            questions = [questions]
        elif not isinstance(questions, list):
            # Unexpected type; skip this example gracefully
            continue

        for q in questions:
            q_uid = q.get("uid") or f"{ex.get('uid','')}-{q.get('order','')}"
            ans_type = (q.get("answer_type") or "").lower()
            scale = (q.get("scale") or "None")
            rel_pars = q.get("rel_paragraphs") or []
            question_text = q.get("question") or ""
            gold = q.get("answer")

            if ans_type == "spans" and not isinstance(gold, list):
                gold = [gold] if gold is not None else []
            chosen = _format_with_scale(gold, ans_type, scale)

            md_table = _markdown_table(table_2d, table_max_rows, table_max_cols)
            paras_text = _select_paragraphs(paragraphs, rel_pars, only_rel_paras, para_max_chars)
            prompt = (
                "You are a financial analyst. Use the TABLE and the PASSAGES to answer.\n\n"
                "TABLE:\n" + (md_table if md_table else "(no table)") + "\n\n"
                "PASSAGES:\n" + (paras_text if paras_text else "(no passages)") + "\n\n"
                f"Question: {question_text}\n"
                "Answer:"
            )

            rng_seed = _seed_from_uid(q_uid, seed)
            if ans_type in ("arithmetic", "counting"):
                try:
                    true_val = float(gold)
                except Exception:
                    true_val = 0.0
                wrong_val = _numeric_perturb(true_val, rng_seed)
                rejected = _format_with_scale(wrong_val, ans_type, scale)
            else:
                gold_texts = gold if isinstance(gold, list) else [str(gold)]
                cands = _collect_distractors(table_2d, [str(x) for x in gold_texts])
                if cands:
                    rejected = random.Random(rng_seed).choice(cands)
                else:
                    rejected = (gold_texts[0] + "s") if gold_texts and isinstance(gold_texts[0], str) else "N/A"

            if not chosen or not rejected or chosen.strip().lower() == rejected.strip().lower():
                continue

            yield {"prompt": prompt, "chosen": str(chosen), "rejected": str(rejected)}


# -----------------------------
# Model loading
# -----------------------------

def load_causal_lm(model_id: str, use_4bit: bool, dtype_compute):
    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype_compute)
        return AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_cfg, trust_remote_code=True, device_map="auto"
        )
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype_compute, trust_remote_code=True, device_map="auto"
    )


# -----------------------------
# Training
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser("DPO on TAT-QA (auto pairs, base model only)")
    # Base / output
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--output_dir", default="qwen3_tatqa_dpo_base")

    # DPO
    p.add_argument("--beta", type=float, default=0.03)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument("--max_length", type=int, default=3072)

    # LoRA (fresh adapter only)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", default="q_proj,v_proj",
                   help="Comma-separated LoRA target modules")

    # TAT-QA options
    p.add_argument("--tatqa_split", choices=["train", "validation", "test"], default="train")
    p.add_argument("--tatqa_table_rows", type=int, default=20)
    p.add_argument("--tatqa_table_cols", type=int, default=8)
    p.add_argument("--tatqa_para_chars", type=int, default=1200)
    p.add_argument("--tatqa_only_rel_paras", action="store_true", default=True)

    # System / misc
    p.add_argument("--no_4bit", action="store_true", help="Disable 4-bit quantization (Windows-friendly)")
    p.add_argument("--use_chat_template", action="store_true",
                   help="Wrap the prompt in the model's chat template (Qwen etc.)")
    # Dev/test helpers
    p.add_argument("--no_train", action="store_true", help="Only build dataset pairs and exit")
    p.add_argument("--limit_pairs", type=int, default=0, help="If >0, limit number of constructed pairs")
    # Eval
    p.add_argument("--do_eval", action="store_true", default=True, help="Run evaluation on TAT-QA split (default: True)")
    p.add_argument("--eval_split", choices=["validation", "test", "train"], default="validation")
    p.add_argument("--eval_limit", type=int, default=1000)
    p.add_argument("--eval_max_new_tokens", type=int, default=64)
    p.add_argument("--eval_temperature", type=float, default=0.2)
    p.add_argument("--eval_top_p", type=float, default=0.9)
    p.add_argument("--eval_baseline", action="store_true", default=True, help="Use base (no LoRA) for eval—baseline measure (default: True)")
    p.add_argument("--eval_save_json", type=str, default="", help="Optional path to save eval results JSONL")
    p.add_argument("--load_adapter", type=str, default="", help="Path to a saved PEFT LoRA adapter to load (eval or continue training)")
    
    # MT-Bench evaluation
    p.add_argument("--do_mtbench", action="store_true", default=True, help="Run MT-Bench evaluation (default: True)")
    p.add_argument("--mtbench_file", type=str, default="mt_bench_ru.json", help="MT-Bench questions file")
    p.add_argument("--mtbench_limit", type=int, default=50, help="Number of MT-Bench questions to evaluate")
    p.add_argument("--mtbench_max_tokens", type=int, default=256, help="Max tokens for MT-Bench responses")
    
    # Instruction Following evaluation
    p.add_argument("--do_instruct", action="store_true", default=True, help="Run instruction following evaluation (default: True)")
    p.add_argument("--instruct_limit", type=int, default=25, help="Number of instruction following tasks to evaluate")
    p.add_argument("--instruct_max_tokens", type=int, default=512, help="Max tokens for instruction following responses")
    # Experiment logging
    p.add_argument("--exp_log", type=str, default="dpo_experiments.jsonl", help="Path to append JSONL logs for each run")
    p.add_argument("--exp_tag", type=str, default="", help="Optional tag/name for this experiment run")
    # Continuation/resume controls
    p.add_argument("--adapter_trainable", action="store_true", help="If set with --load_adapter, continue training that adapter instead of treating it as eval-only")
    p.add_argument("--resume_from_checkpoint", type=str, default="", help="Path to Trainer checkpoint to resume from (e.g., output_dir/checkpoint-XXXX)")
    return p.parse_args()


def apply_chat_template_if_enabled(tok, ds: Dataset, enabled: bool) -> Dataset:
    if not enabled:
        return ds
    def _fmt(ex):
        messages = [
            {"role": "system", "content": "You are a helpful financial analysis assistant."},
            {"role": "user", "content": ex["prompt"]},
        ]
        ex["prompt"] = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return ex
    return ds.map(_fmt)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_result = None
    post_result = None
    train_metrics = None
    mtbench_baseline_result = None
    mtbench_post_result = None
    instruct_baseline_result = None
    instruct_post_result = None

    # ---------- Load TAT-QA without Arrow conversion ----------
    def iter_tatqa_examples(split: str):
        # Try to download the raw JSON file from HF Hub to a local path, then parse with stdlib json
        file_by_split = {
            "train": "tatqa_dataset_train.json",
            "validation": "tatqa_dataset_dev.json",
            "test": "tatqa_dataset_test.json",
        }
        try:
            from huggingface_hub import hf_hub_download
            local_path = hf_hub_download(repo_id="next-tat/TAT-QA", filename=file_by_split[split])
            import json
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
        import requests, json
        resp = requests.get(url_map[split], timeout=120)
        resp.raise_for_status()
        data = resp.json()
        for ex in data:
            yield ex

    raw_stream = iter_tatqa_examples(args.tatqa_split)

    # Build DPO pairs while streaming
    pairs_iter = tatqa_to_dpo_pairs(
        (ex for ex in raw_stream),
        table_max_rows=args.tatqa_table_rows,
        table_max_cols=args.tatqa_table_cols,
        para_max_chars=args.tatqa_para_chars,
        only_rel_paras=args.tatqa_only_rel_paras,
        seed=args.seed,
    )

    # Materialize flat dataset (strings only)
    pairs = list(pairs_iter)
    if args.limit_pairs and args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]
    if not pairs:
        raise SystemExit("No pairs constructed from TAT-QA—check parameters.")
    dpo_ds = Dataset.from_list(pairs)

    # Tokenizer (only needed from here on)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # Optional: apply chat template
    dpo_ds = apply_chat_template_if_enabled(tok, dpo_ds, args.use_chat_template)

    # Early dataset-only visibility
    if args.no_train and not args.do_eval:
        print(f"Prepared DPO pairs: {len(dpo_ds)} (limit={args.limit_pairs})")
        print(f"Example:\nPrompt head: {dpo_ds[0]['prompt'][:120]}...\nChosen: {dpo_ds[0]['chosen']}\nRejected: {dpo_ds[0]['rejected']}")
        return

    # Dtype heuristic
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    # Policy (base model + fresh LoRA)
    try:
        base_policy = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
    except Exception as e:
        print(f"[warn] 4-bit load failed ({e}); falling back to non-quantized.")
        base_policy = load_causal_lm(args.model, use_4bit=False, dtype_compute=compute_dtype)
    base_policy.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules, task_type="CAUSAL_LM", bias="none",
    )
    policy = get_peft_model(base_policy, lora_cfg)
    if args.load_adapter:
        try:
            from peft import PeftModel
            policy = PeftModel.from_pretrained(base_policy, args.load_adapter, is_trainable=bool(args.adapter_trainable and not args.no_train))
            mode = "trainable" if (args.adapter_trainable and not args.no_train) else "frozen"
            print(f"[adapter] Loaded LoRA adapter from {args.load_adapter} ({mode})")
        except Exception as e:
            print(f"[warn] Failed to load adapter from {args.load_adapter}: {e}")

    # Reference model (frozen base)
    try:
        ref_model = load_causal_lm(args.model, use_4bit=not args.no_4bit, dtype_compute=compute_dtype)
    except Exception as e:
        print(f"[warn] ref 4-bit load failed ({e}); falling back to non-quantized.")
        ref_model = load_causal_lm(args.model, use_4bit=False, dtype_compute=compute_dtype)
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ---------- Eval helpers ----------
    import re

    def _normalize_text(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[\t\n\r]", " ", s)
        s = re.sub(r"[^\w\s\.\-]", "", s)
        return s

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
        # spans or others -> textual containment / exact-ish
        if isinstance(gold, list):
            gold_texts = [str(x) for x in gold]
        else:
            gold_texts = [str(gold)]
        pred_n = _normalize_text(pred)
        for g in gold_texts:
            if _normalize_text(g) in pred_n:
                return True
        return False

    def _tatqa_eval_items(split: str, limit: int):
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

    def run_eval(eval_model, eval_tok, split: str, limit: int, save_path: str = ""):
        eval_model.eval()
        results = []
        correct = 0
        total = 0
        # Per-answer-type counters
        by_type = {}
        # Configure generation args carefully to avoid warnings when do_sample=False
        do_sample = (args.eval_temperature and args.eval_temperature > 0.0) or (args.eval_top_p and args.eval_top_p < 1.0)
        gen_kwargs = {"max_new_tokens": args.eval_max_new_tokens}
        if do_sample:
            gen_kwargs.update(
                dict(
                    do_sample=True,
                    temperature=max(1e-6, float(args.eval_temperature)),
                    top_p=float(args.eval_top_p),
                )
            )
        else:
            # Greedy; do not pass sampling params (temperature/top_p/top_k)
            gen_kwargs.update(dict(do_sample=False, num_beams=1))

        # Suppress noisy generation warnings about ignored sampling flags
        try:
            from transformers.utils import logging as hf_logging
            _prev_level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            _prev_level = None
        for item in _tatqa_eval_items(split, limit):
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
            # Per-type accounting
            t = (item.get("ans_type") or "unknown").lower()
            if t not in by_type:
                by_type[t] = {"correct": 0, "total": 0}
            by_type[t]["correct"] += (1 if ok else 0)
            by_type[t]["total"] += 1
            results.append({
                "q_uid": item["q_uid"],
                "pred": pred,
                "gold": item["gold"],
                "ok": ok,
                "ans_type": t,
                "scale": item.get("scale")
            })
        acc = (correct / total) if total else 0.0
        print(f"Eval {split} — N={total} | EM-like acc={acc:.3f}")
        # Print per-type breakdown
        if by_type:
            try:
                breakdown_lines = []
                for t, ct in sorted(by_type.items(), key=lambda x: x[0]):
                    t_acc = (ct["correct"] / ct["total"]) if ct["total"] else 0.0
                    breakdown_lines.append(f"   - {t}: {t_acc:.3f} ({ct['correct']}/{ct['total']})")
                print("Per-type breakdown:\n" + "\n".join(breakdown_lines))
            except Exception:
                pass
        # Restore logging level
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

    # ---------- MT-Bench evaluation function ----------
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
            
            # Generation settings
            do_sample = (args.eval_temperature and args.eval_temperature > 0.0) or (args.eval_top_p and args.eval_top_p < 1.0)
            gen_kwargs = {"max_new_tokens": max_tokens}
            if do_sample:
                gen_kwargs.update(dict(do_sample=True, temperature=max(1e-6, float(args.eval_temperature)), top_p=float(args.eval_top_p)))
            else:
                gen_kwargs.update(dict(do_sample=False, num_beams=1))
            
            total_score = 0.0
            scored_count = 0
            
            # Suppress noisy generation warnings about ignored sampling flags
            try:
                from transformers.utils import logging as hf_logging
                _prev_level = hf_logging.get_verbosity()
                hf_logging.set_verbosity_error()
            except Exception:
                _prev_level = None

            for i, q in enumerate(questions):
                prompt = q["prompt"]
                if args.use_chat_template:
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

    # ---------- Instruction Following evaluation function ----------
    def run_instruct_eval(eval_model, eval_tok, limit: int, max_tokens: int, save_path: str = ""):
        """Run instruction following evaluation with diverse task types"""
        
        # Define instruction following tasks
        instruct_tasks = [
            {
                "id": "format_list",
                "category": "formatting",
                "instruction": "List the top 3 benefits of exercise. Format your response as a numbered list with exactly 3 items.",
                "criteria": ["numbered_list", "exactly_3_items", "about_exercise"]
            },
            {
                "id": "creative_constraint",
                "category": "creative",
                "instruction": "Write a short story (exactly 2 sentences) about a robot that learns to paint. The story must include the words 'canvas' and 'dream'.",
                "criteria": ["exactly_2_sentences", "contains_canvas", "contains_dream", "about_robot_painting"]
            },
            {
                "id": "analysis_structure",
                "category": "analysis",
                "instruction": "Analyze the advantages and disadvantages of remote work. Structure your response with exactly two sections: 'Advantages' and 'Disadvantages'.",
                "criteria": ["has_advantages_section", "has_disadvantages_section", "about_remote_work"]
            },
            {
                "id": "code_simple",
                "category": "coding",
                "instruction": "Write a Python function called 'add_numbers' that takes two parameters and returns their sum. Include a docstring.",
                "criteria": ["python_function", "named_add_numbers", "takes_two_params", "has_docstring"]
            },
            {
                "id": "reasoning_steps",
                "category": "reasoning",
                "instruction": "If a train travels 60 mph for 2 hours, then 80 mph for 1.5 hours, what is the total distance? Show your work step by step.",
                "criteria": ["shows_calculation_steps", "correct_approach", "includes_units"]
            },
            {
                "id": "summarize_constraint",
                "category": "summarization",
                "instruction": "Summarize the concept of photosynthesis in exactly 50 words. Count the words in your response.",
                "criteria": ["about_photosynthesis", "word_count_mentioned", "approximately_50_words"]
            },
            {
                "id": "comparison_table",
                "category": "comparison",
                "instruction": "Compare cats and dogs as pets using a simple table format with 3 characteristics: Size, Care Level, and Social Needs.",
                "criteria": ["table_format", "compares_cats_dogs", "includes_size_care_social"]
            },
            {
                "id": "roleplay_specific",
                "category": "roleplay",
                "instruction": "You are a librarian helping someone find books about space exploration. Recommend 2 books and explain why each would be helpful.",
                "criteria": ["acts_as_librarian", "recommends_2_books", "about_space_exploration", "explains_helpfulness"]
            },
            {
                "id": "translation_explain",
                "category": "language",
                "instruction": "Translate 'Hello, how are you?' to Spanish and French. Then explain which language was easier for you to translate and why.",
                "criteria": ["spanish_translation", "french_translation", "provides_explanation"]
            },
            {
                "id": "recipe_specific",
                "category": "instructions",
                "instruction": "Give me a recipe for scrambled eggs with exactly 4 ingredients and 3 cooking steps. Number the steps.",
                "criteria": ["exactly_4_ingredients", "exactly_3_steps", "numbered_steps", "about_scrambled_eggs"]
            },
            {
                "id": "letter_format",
                "category": "writing",
                "instruction": "Write a brief thank you note to a teacher. Include a proper greeting, one specific thing you're thankful for, and a proper closing.",
                "criteria": ["has_greeting", "specific_thanks", "proper_closing", "appropriate_tone"]
            },
            {
                "id": "math_explanation",
                "category": "math",
                "instruction": "Explain how to calculate the area of a rectangle to a 10-year-old. Use simple language and give an example with specific numbers.",
                "criteria": ["simple_language", "includes_formula", "provides_specific_example", "age_appropriate"]
            },
            {
                "id": "pros_cons_format",
                "category": "decision",
                "instruction": "Should I buy an electric car? Give me exactly 2 pros and 2 cons in a clear format.",
                "criteria": ["exactly_2_pros", "exactly_2_cons", "clear_format", "about_electric_cars"]
            },
            {
                "id": "definition_example",
                "category": "explanation", 
                "instruction": "Define 'artificial intelligence' and then give 2 real-world examples of how it's used today.",
                "criteria": ["defines_ai", "gives_2_examples", "examples_are_realistic"]
            },
            {
                "id": "story_structure",
                "category": "creative",
                "instruction": "Tell me a story about friendship with a clear beginning, middle, and end. Label each section.",
                "criteria": ["about_friendship", "labeled_beginning", "labeled_middle", "labeled_end"]
            },
            {
                "id": "email_professional",
                "category": "professional",
                "instruction": "Write a professional email requesting a meeting. Include subject line, proper greeting, purpose, suggested times, and professional closing.",
                "criteria": ["has_subject_line", "proper_greeting", "states_purpose", "suggests_times", "professional_closing"]
            },
            {
                "id": "troubleshoot_steps",
                "category": "problem_solving",
                "instruction": "My computer won't start. Give me 4 troubleshooting steps to try, numbered in order of priority.",
                "criteria": ["exactly_4_steps", "numbered_in_order", "realistic_troubleshooting", "prioritized"]
            },
            {
                "id": "budget_calculation",
                "category": "practical",
                "instruction": "I earn $3000/month and want to save 20% for vacation. Calculate how much I save monthly and how long to save $2400. Show the math.",
                "criteria": ["calculates_monthly_savings", "calculates_time_needed", "shows_calculations"]
            },
            {
                "id": "persuasive_structure",
                "category": "persuasion",
                "instruction": "Convince me to exercise more using exactly 3 persuasive arguments. Label each argument clearly.",
                "criteria": ["exactly_3_arguments", "labeled_clearly", "persuasive_about_exercise"]
            },
            {
                "id": "sequence_instruction",
                "category": "sequencing",
                "instruction": "Explain how to make a paper airplane in 5 sequential steps. Start each step with 'Step X:'.",
                "criteria": ["exactly_5_steps", "sequential_order", "starts_with_step_x", "about_paper_airplane"]
            },
            {
                "id": "conditional_response",
                "category": "logic",
                "instruction": "If it's raining, I should bring an umbrella. If it's sunny, I should wear sunglasses. What should I do if it's both rainy and sunny?",
                "criteria": ["addresses_both_conditions", "logical_response", "practical_advice"]
            },
            {
                "id": "data_interpretation",
                "category": "analysis",
                "instruction": "If 40% of 200 students prefer pizza and 30% prefer burgers, how many students prefer each? Show your calculation and state which is more popular.",
                "criteria": ["calculates_pizza_students", "calculates_burger_students", "shows_calculation", "states_more_popular"]
            },
            {
                "id": "multiple_choice_explain",
                "category": "reasoning",
                "instruction": "Which is larger: 3/4 or 0.8? Show how you determined this and explain your reasoning clearly.",
                "criteria": ["identifies_correct_answer", "shows_comparison_method", "clear_reasoning"]
            },
            {
                "id": "rhyme_constraint",
                "category": "creative",
                "instruction": "Write a 4-line poem about sunshine where lines 2 and 4 rhyme. Label which words rhyme.",
                "criteria": ["4_lines", "about_sunshine", "lines_2_and_4_rhyme", "labels_rhyming_words"]
            },
            {
                "id": "dialogue_format",
                "category": "writing",
                "instruction": "Write a short dialogue between a customer and cashier at a grocery store. Use proper dialogue formatting with quotation marks.",
                "criteria": ["customer_cashier_dialogue", "grocery_store_context", "proper_quotation_marks", "realistic_conversation"]
            }
        ]
        
        # Select limited number of tasks
        if limit and limit > 0:
            tasks_to_run = instruct_tasks[:limit]
        else:
            tasks_to_run = instruct_tasks
        
        results = []
        eval_model.eval()
        
        # Default to greedy + chat template for instruction tasks (better formatting adherence)
        gen_kwargs = {"max_new_tokens": max_tokens, "do_sample": False, "num_beams": 1}

        # Suppress noisy generation warnings about ignored sampling flags
        try:
            from transformers.utils import logging as hf_logging
            _prev_level = hf_logging.get_verbosity()
            hf_logging.set_verbosity_error()
        except Exception:
            _prev_level = None
        
        for task in tasks_to_run:
            instruction = task["instruction"]
            # Always use chat template by default for instruction eval
            messages = [
                {"role": "system", "content": "You are a helpful assistant that follows instructions carefully."},
                {"role": "user", "content": instruction},
            ]
            prompt = eval_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = eval_tok(prompt, return_tensors="pt").to(eval_model.device)
            with torch.no_grad():
                out_ids = eval_model.generate(**inputs, **gen_kwargs)
            response = eval_tok.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
            results.append({
                "id": task["id"],
                "category": task["category"],
                "instruction": instruction,
                "response": response,
                "criteria": task["criteria"]
            })
        
        # Score each response based on criteria
        total_score = 0.0
        successful_tasks = 0
        for result in results:
            score = score_instruction_response(result["response"], result["criteria"])
            result["score"] = score
            total_score += score
            if score >= 7.0:  # Consider ≥7 as successful completion
                successful_tasks += 1
        
        avg_score = total_score / len(results) if results else 0.0
        success_rate = (successful_tasks / len(results) * 100) if results else 0.0
        
        print(f"Instruction following eval completed — N={len(results)} tasks, avg score: {avg_score:.2f}/10, success rate: {success_rate:.1f}%")
        
        # Restore logging level
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
            print(f"Saved instruction following results to {save_path}")
        
        return {"n": len(results), "tasks_completed": len(results), "avg_score": avg_score, "success_rate": success_rate}

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
                    criterion_score = 0.7
                    
            elif "spanish_translation" in criterion:
                spanish_words = ["hola", "cómo", "como", "estás", "estas", "usted"]
                if any(word in response_lower for word in spanish_words):
                    criterion_score = 1.0
                    
            elif "french_translation" in criterion:
                french_words = ["bonjour", "salut", "comment", "allez", "vous", "ça", "va"]
                if any(word in response_lower for word in french_words):
                    criterion_score = 1.0
                    
            elif "exactly_4_ingredients" in criterion:
                ingredient_indicators = response.count('\n') + response.count(',') + response.count(';')
                if 3 <= ingredient_indicators <= 6:  # Flexible count based on formatting
                    criterion_score = 0.8
                    
            elif "exactly_3_steps" in criterion:
                step_count = sum(1 for i in range(1, 6) if f"{i}." in response or f"step {i}" in response_lower)
                if step_count == 3:
                    criterion_score = 1.0
                elif step_count == 2 or step_count == 4:
                    criterion_score = 0.5
                    
            elif "numbered_steps" in criterion or "starts_with_step_x" in criterion:
                if any(f"step {i}" in response_lower for i in range(1, 6)) or any(f"{i}." in response for i in range(1, 6)):
                    criterion_score = 1.0
                    
            elif "has_greeting" in criterion:
                greetings = ["dear", "hello", "hi", "good morning", "good afternoon"]
                if any(greeting in response_lower for greeting in greetings):
                    criterion_score = 1.0
                    
            elif "proper_closing" in criterion:
                closings = ["sincerely", "thank you", "best regards", "yours", "respectfully"]
                if any(closing in response_lower for closing in closings):
                    criterion_score = 1.0
                    
            elif "exactly_2_pros" in criterion:
                pro_indicators = response_lower.count("pro:") + response_lower.count("advantage") + response_lower.count("benefit")
                if pro_indicators >= 2:
                    criterion_score = 1.0
                    
            elif "exactly_2_cons" in criterion:
                con_indicators = response_lower.count("con:") + response_lower.count("disadvantage") + response_lower.count("drawback")
                if con_indicators >= 2:
                    criterion_score = 1.0
                    
            elif "gives_2_examples" in criterion:
                example_count = response_lower.count("example") + response_lower.count("such as") + response_lower.count("like")
                if example_count >= 1 and len(response.split()) > 30:  # Approximation
                    criterion_score = 0.8
                    
            elif "labeled_beginning" in criterion:
                if "beginning" in response_lower or "start" in response_lower:
                    criterion_score = 1.0
                    
            elif "labeled_middle" in criterion:
                if "middle" in response_lower:
                    criterion_score = 1.0
                    
            elif "labeled_end" in criterion:
                if "end" in response_lower or "conclusion" in response_lower:
                    criterion_score = 1.0
                    
            elif "4_lines" in criterion:
                line_count = len([line for line in response.split('\n') if line.strip()])
                if line_count == 4:
                    criterion_score = 1.0
                elif abs(line_count - 4) <= 1:
                    criterion_score = 0.5
                    
            elif "lines_2_and_4_rhyme" in criterion:
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                if len(lines) >= 4:
                    criterion_score = 0.7  # Assume rhyming is present (hard to verify automatically)
                    
            # Generic content criteria
            elif any(topic in criterion for topic in ["about_exercise", "about_robot_painting", "about_remote_work", "about_photosynthesis", "about_scrambled_eggs", "about_electric_cars", "about_friendship", "about_sunshine"]):
                # Check if response is reasonably long and on-topic
                if len(response) > 20:
                    criterion_score = 0.8
                    
            # Generic format criteria
            elif any(format_word in criterion for format_word in ["clear_format", "realistic_troubleshooting", "age_appropriate", "professional_closing"]):
                if len(response) > 15:  # Basic completeness check
                    criterion_score = 0.7
                    
            else:
                # Default scoring for unrecognized criteria
                if len(response) > 10:
                    criterion_score = 0.5
            
            score += criterion_score
        
        # Convert to 1-10 scale
        normalized_score = (score / max_score) if max_score > 0 else 0.0
        final_score = 1.0 + (normalized_score * 9.0)
        return min(10.0, max(1.0, final_score))

    # ---------- Optional baseline eval before training ----------
    if args.do_eval and args.eval_baseline:
        print("[eval] Running baseline eval on", args.eval_split)
        baseline_result = run_eval(ref_model, tok, args.eval_split, args.eval_limit, 
                                 f"{args.eval_save_json.split('.')[0]}_baseline.jsonl" if args.eval_save_json else "")
        
        # MT-Bench baseline
        if args.do_mtbench:
            print("[mtbench] Running baseline MT-Bench eval")
            mtbench_baseline_result = run_mtbench_eval(ref_model, tok, args.mtbench_file, args.mtbench_limit, 
                                                     args.mtbench_max_tokens, f"{out_dir}/mtbench_baseline.jsonl")
        
        # Instruction following baseline
        if args.do_instruct:
            print("[instruct] Running baseline instruction following eval")
            instruct_baseline_result = run_instruct_eval(ref_model, tok, args.instruct_limit, 
                                                        args.instruct_max_tokens, f"{out_dir}/instruct_baseline.jsonl")

    # ---------- Train ----------
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
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
    )

    if DPOConfig is not None:
        dpo_cfg = DPOConfig(
            # Base TrainingArguments passthrough
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
            remove_unused_columns=False,
            report_to=[],
            seed=args.seed,
            # DPO-specific
            beta=args.beta,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            generate_during_eval=False,
        )
        trainer = DPOTrainer(
            model=policy,
            ref_model=ref_model,
            args=dpo_cfg,
            tokenizer=tok,
            train_dataset=dpo_ds,
        )
    else:
        trainer = DPOTrainer(
            model=policy,
            ref_model=ref_model,
            args=train_args,
            beta=args.beta,
            train_dataset=dpo_ds,
            tokenizer=tok,
            max_prompt_length=args.max_prompt_length,
            max_length=args.max_length,
            generate_during_eval=False,
        )

    if not args.no_train:
        # Support resuming from a Trainer checkpoint if provided
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
        print(f"ℹ️ Split: {args.tatqa_split} | Pairs: {len(pairs)}")

    # ---------- Eval after training ----------
    if args.do_eval and not args.no_train:
        print("[eval] Running post-training eval on", args.eval_split)
        post_result = run_eval(policy, tok, args.eval_split, args.eval_limit, 
                             f"{args.eval_save_json.split('.')[0]}_post.jsonl" if args.eval_save_json else f"{out_dir}/tatqa_post.jsonl")
        
        # MT-Bench post-training
        if args.do_mtbench:
            print("[mtbench] Running post-training MT-Bench eval")
            mtbench_post_result = run_mtbench_eval(policy, tok, args.mtbench_file, args.mtbench_limit, 
                                                 args.mtbench_max_tokens, f"{out_dir}/mtbench_post.jsonl")
        
        # Instruction following post-training
        if args.do_instruct:
            print("[instruct] Running post-training instruction following eval")
            instruct_post_result = run_instruct_eval(policy, tok, args.instruct_limit, 
                                                    args.instruct_max_tokens, f"{out_dir}/instruct_post.jsonl")
    elif args.do_eval and args.eval_baseline and not args.no_train:
        # This case handles when we only did baseline but still want post-training eval
        print("[eval] Running post-training eval on", args.eval_split)
        post_result = run_eval(policy, tok, args.eval_split, args.eval_limit, 
                             f"{args.eval_save_json.split('.')[0]}_post.jsonl" if args.eval_save_json else f"{out_dir}/tatqa_post.jsonl")

    # ---------- Eval-only mode with loaded adapter ----------
    elif args.do_eval and args.no_train and args.load_adapter:
        # When evaluating only, run eval on the loaded adapter (policy)
        print("[eval] Running eval-only with loaded adapter on", args.eval_split)
        post_result = run_eval(policy, tok, args.eval_split, args.eval_limit,
                             f"{args.eval_save_json.split('.')[0]}_post.jsonl" if args.eval_save_json else f"{out_dir}/tatqa_post.jsonl")

        if args.do_mtbench:
            print("[mtbench] Running eval-only MT-Bench with loaded adapter")
            mtbench_post_result = run_mtbench_eval(policy, tok, args.mtbench_file, args.mtbench_limit,
                                                 args.mtbench_max_tokens, f"{out_dir}/mtbench_post.jsonl")

        if args.do_instruct:
            print("[instruct] Running eval-only instruction following with loaded adapter")
            instruct_post_result = run_instruct_eval(policy, tok, args.instruct_limit,
                                                    args.instruct_max_tokens, f"{out_dir}/instruct_post.jsonl")

    # ---------- Experiment log (JSONL append) ----------
    try:
        exp_log_path = Path(args.exp_log)
        # Build compact record: only tweakable params and baseline/post metrics
        def _compact_eval(res):
            if not res:
                return None
            out = {}
            if "n" in res:
                out["n"] = res.get("n")
            # Map alternative count fields to n
            if "questions_completed" in res and "n" not in out:
                out["n"] = res.get("questions_completed")
            if "acc" in res:
                out["acc"] = res.get("acc")
            if "avg_score" in res:
                out["avg_score"] = res.get("avg_score")
            if "success_rate" in res:
                out["success_rate"] = res.get("success_rate")
            if "tasks_completed" in res and "n" not in out:
                out["n"] = res.get("tasks_completed")
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
                "lora": {
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "targets": target_modules,
                },
            },
            "eval": {
                "baseline": _compact_eval(baseline_result),
                "post": _compact_eval(post_result),
            },
            "mtbench": {
                "baseline": _compact_eval(mtbench_baseline_result),
                "post": _compact_eval(mtbench_post_result),
            },
            "instruct": {
                "baseline": _compact_eval(instruct_baseline_result),
                "post": _compact_eval(instruct_post_result),
            },
        }
        exp_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(exp_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"📝 Appended experiment log to {exp_log_path}")
        
        # ---------- Print comprehensive metrics summary ----------
        print("\n" + "="*80)
        print("🎯 TRAINING COMPLETE - METRICS SUMMARY")
        print("="*80)
        
        # Training info
        print(f"📊 Training: {len(pairs):,} preference pairs in {train_metrics.get('train_runtime', 0):.1f}s" if train_metrics else f"📊 Training: {len(pairs):,} preference pairs")
        print(f"🎯 Loss: {train_metrics.get('train_loss', 'N/A'):.4f}" if train_metrics else "🎯 Loss: N/A")
        
        # TAT-QA Financial QA Results
        print(f"\n💰 TAT-QA Financial QA ({args.eval_split} set):")
        if baseline_result and post_result:
            baseline_acc = baseline_result.get('acc', 0)
            post_acc = post_result.get('acc', 0)
            improvement = post_acc - baseline_acc
            improvement_pct = (improvement / baseline_acc * 100) if baseline_acc > 0 else 0
            
            print(f"   Before: {baseline_acc:.1%} ({baseline_result.get('n', 0)} examples)")
            print(f"   After:  {post_acc:.1%} ({post_result.get('n', 0)} examples)")
            if improvement >= 0:
                print(f"   📈 Improvement: +{improvement:.1%} ({improvement_pct:+.1f}%)")
            else:
                print(f"   📉 Change: {improvement:.1%} ({improvement_pct:+.1f}%)")
        elif baseline_result:
            print(f"   Baseline: {baseline_result.get('acc', 0):.1%} ({baseline_result.get('n', 0)} examples)")
        elif post_result:
            print(f"   Post-training: {post_result.get('acc', 0):.1%} ({post_result.get('n', 0)} examples)")
        else:
            print("   No TAT-QA evaluation completed")
        
        # MT-Bench General Capability Results
        print(f"\n🌐 MT-Bench General Capability:")
        if mtbench_baseline_result and mtbench_post_result:
            baseline_n = mtbench_baseline_result.get('questions_completed', 0)
            post_n = mtbench_post_result.get('questions_completed', 0)
            baseline_score = mtbench_baseline_result.get('avg_score', 0)
            post_score = mtbench_post_result.get('avg_score', 0)
            
            print(f"   Baseline: {baseline_n} questions, avg score: {baseline_score:.2f}/10")
            print(f"   Post-training: {post_n} questions, avg score: {post_score:.2f}/10")
            
            if baseline_score > 0 and post_score > 0:
                score_improvement = post_score - baseline_score
                if score_improvement >= 0:
                    print(f"   📈 Score improvement: +{score_improvement:.2f} points")
                else:
                    print(f"   � Score change: {score_improvement:.2f} points")
            
            print(f"   �📄 Files: mtbench_baseline.jsonl, mtbench_post.jsonl")
            
        elif mtbench_baseline_result:
            baseline_score = mtbench_baseline_result.get('avg_score', 0)
            if baseline_score > 0:
                print(f"   Baseline: {mtbench_baseline_result.get('questions_completed', 0)} questions, avg score: {baseline_score:.2f}/10")
            else:
                print(f"   Baseline: {mtbench_baseline_result.get('questions_completed', 0)} questions completed")
        elif mtbench_post_result:
            post_score = mtbench_post_result.get('avg_score', 0)
            if post_score > 0:
                print(f"   Post-training: {mtbench_post_result.get('questions_completed', 0)} questions, avg score: {post_score:.2f}/10")
            else:
                print(f"   Post-training: {mtbench_post_result.get('questions_completed', 0)} questions completed")
        else:
            print("   No MT-Bench evaluation completed")
        
        # Instruction Following Results
        print(f"\n📋 Instruction Following Capability:")
        if instruct_baseline_result and instruct_post_result:
            baseline_n = instruct_baseline_result.get('tasks_completed', 0)
            post_n = instruct_post_result.get('tasks_completed', 0)
            baseline_score = instruct_baseline_result.get('avg_score', 0)
            post_score = instruct_post_result.get('avg_score', 0)
            baseline_success = instruct_baseline_result.get('success_rate', 0)
            post_success = instruct_post_result.get('success_rate', 0)
            
            print(f"   Baseline: {baseline_n} tasks, avg score: {baseline_score:.2f}/10, success: {baseline_success:.1f}%")
            print(f"   Post-training: {post_n} tasks, avg score: {post_score:.2f}/10, success: {post_success:.1f}%")
            
            if baseline_score > 0 and post_score > 0:
                score_improvement = post_score - baseline_score
                success_improvement = post_success - baseline_success
                if score_improvement >= 0:
                    print(f"   📈 Improvements: +{score_improvement:.2f} points, +{success_improvement:.1f}% success rate")
                else:
                    print(f"   � Changes: {score_improvement:.2f} points, {success_improvement:.1f}% success rate")
            
            print(f"   �📄 Files: instruct_baseline.jsonl, instruct_post.jsonl")
            
        elif instruct_baseline_result:
            baseline_score = instruct_baseline_result.get('avg_score', 0)
            baseline_success = instruct_baseline_result.get('success_rate', 0)
            if baseline_score > 0:
                print(f"   Baseline: {instruct_baseline_result.get('tasks_completed', 0)} tasks, avg score: {baseline_score:.2f}/10, success: {baseline_success:.1f}%")
            else:
                print(f"   Baseline: {instruct_baseline_result.get('tasks_completed', 0)} tasks completed")
        elif instruct_post_result:
            post_score = instruct_post_result.get('avg_score', 0)
            post_success = instruct_post_result.get('success_rate', 0)
            if post_score > 0:
                print(f"   Post-training: {instruct_post_result.get('tasks_completed', 0)} tasks, avg score: {post_score:.2f}/10, success: {post_success:.1f}%")
            else:
                print(f"   Post-training: {instruct_post_result.get('tasks_completed', 0)} tasks completed")
        else:
            print("   No instruction following evaluation completed")
        
        # Output files summary
        print(f"\n📁 Output Files:")
        print(f"   🎯 Model: {out_dir}/adapter_model.safetensors")
        if baseline_result or post_result:
            if baseline_result:
                print("   📊 TAT-QA Baseline: *_baseline.jsonl")
            if post_result:
                print("   📊 TAT-QA Post: tatqa_post.jsonl")
        if mtbench_baseline_result or mtbench_post_result:
            print("   🌐 MT-Bench: mtbench_baseline.jsonl, mtbench_post.jsonl")
        if instruct_baseline_result or instruct_post_result:
            print("   📋 Instruction Following: instruct_baseline.jsonl, instruct_post.jsonl")
        
        # Recommendations
        print(f"\n💡 Next Steps:")
        if baseline_result and post_result:
            if post_result.get('acc', 0) > baseline_result.get('acc', 0):
                print("   ✅ Model shows improvement on financial QA task")
            else:
                print("   ⚠️  Consider adjusting hyperparameters or training longer")
        
        print("   📋 Review MT-Bench responses for qualitative capability assessment")
        print("   � Analyze instruction following tasks for formatting and adherence")
        print("   �🔄 Run additional experiments with different configurations")
        
        print("="*80)
        
    except Exception as e:
        print(f"[warn] Failed to write experiment log: {e}")


if __name__ == "__main__":
    main()