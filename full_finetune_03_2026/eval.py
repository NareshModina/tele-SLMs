"""
tele-SLMs — Tele-Eval Benchmark Evaluation
============================================
Evaluates a trained checkpoint (or any HuggingFace model) on the
AliMaatouk/Tele-Eval benchmark (750k open-ended telecom Q&A pairs).

All defaults are read from config.py. CLI flags override config values.

Metrics:
    Token F1      word-overlap with gold answer  (primary)
    Exact Match   normalised string equality
    Perplexity    cross-entropy on gold answer tokens (lower = better)

Usage:
    # Evaluate fine-tuned checkpoint vs base model
    python eval.py --checkpoint ./checkpoints/SmolLM2-135M-telespec \
                   --model HuggingFaceTB/SmolLM2-135M

    # Evaluate base model only (as baseline)
    python eval.py --model HuggingFaceTB/SmolLM2-135M --label SmolLM2-135M-base

    # Print leaderboard from all saved results
    python eval.py --summary-only

Outputs:
    ./results/{label}_eval.json
    ./results/summary.json       (cumulative leaderboard)
"""

import argparse
import json
import math
import os
import re
import string
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import config as C

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_slug(name: str) -> str:
    return name.split("/")[-1]


def results_path(label: str) -> str:
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    return os.path.join(C.RESULTS_DIR, f"{label}_eval.json")


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tok = normalize(pred).split()
    gold_tok = normalize(gold).split()
    if not pred_tok or not gold_tok:
        return 0.0
    common = Counter(pred_tok) & Counter(gold_tok)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_tok)
    r = n / len(gold_tok)
    return 2 * p * r / (p + r)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

PROMPT = "Question: {q}\nAnswer:"


def generate_answers(model, tokenizer, questions: list[str]) -> list[str]:
    model.eval()
    answers = []
    for q in tqdm(questions, desc="  generating", leave=False):
        inputs = tokenizer(
            PROMPT.format(q=q),
            return_tensors = "pt",
            truncation     = True,
            max_length     = 512,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = C.EVAL["max_new_tokens"],
                do_sample      = False,
                eos_token_id   = tokenizer.eos_token_id,
                pad_token_id   = tokenizer.eos_token_id,
            )
        new_ids  = out[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)
        answers.append(response.split("\n")[0].strip())
    return answers


def compute_perplexity(model, tokenizer,
                       questions: list[str], answers: list[str]) -> float:
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for q, a in tqdm(zip(questions, answers),
                     total=len(questions), desc="  perplexity", leave=False):
        prompt   = PROMPT.format(q=q)
        full_seq = prompt + " " + a

        enc_full   = tokenizer(full_seq, return_tensors="pt",
                               truncation=True, max_length=1024).to(model.device)
        enc_prompt = tokenizer(prompt, return_tensors="pt",
                               add_special_tokens=False).to(model.device)

        labels    = enc_full["input_ids"].clone()
        p_len     = enc_prompt["input_ids"].shape[1]
        labels[:, :p_len] = -100

        if (labels != -100).sum() == 0:
            continue

        with torch.no_grad():
            out = model(**enc_full, labels=labels)

        n_tok = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tok
        total_tokens += n_tok

    if total_tokens == 0:
        return float("inf")
    return round(math.exp(total_loss / total_tokens), 4)


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate(load_path: str, model_name: str, label: str,
             n: int, seed: int, source_filter: str = "all") -> dict:

    print(f"\n{'='*60}")
    print(f"  LABEL     : {label}")
    print(f"  PATH      : {load_path}")
    print(f"  EVAL      : {C.TELEEVAL_DATASET}  (n={n})")
    print(f"{'='*60}")

    # ── Load Tele-Eval ───────────────────────────────────────────────────
    print("  Loading Tele-Eval...")
    ds = load_dataset(C.TELEEVAL_DATASET, split="data")

    # Apply source filter before shuffling and selecting
    if source_filter != "all":
        ds = ds.filter(lambda x: x["id"].startswith(source_filter + "_"))
        print(f"  Source filter   : {source_filter} ({len(ds):,} examples available)")

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    questions = list(ds["Statement"])
    gold      = list(ds["Answer"])
    ids       = list(ds["id"])

    src = {
        "standard": sum(1 for i in ids if "standard" in i),
        "arxiv":    sum(1 for i in ids if "arxiv"    in i),
        "wiki":     sum(1 for i in ids if "wiki"     in i),
    }
    print(f"  Source breakdown : {src}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\n  Loading model from: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        load_path, dtype=dtype, device_map="auto"
    )

    # ── Step 1: Generate + score ──────────────────────────────────────────
    print("\n  Step 1/2 — Generating answers...")
    predictions = generate_answers(model, tokenizer, questions)

    em_scores = [exact_match(p, g) for p, g in zip(predictions, gold)]
    f1_scores = [token_f1(p, g)    for p, g in zip(predictions, gold)]
    em = round(100 * sum(em_scores) / len(em_scores), 2)
    f1 = round(100 * sum(f1_scores) / len(f1_scores), 2)

    # ── Step 2: Perplexity ────────────────────────────────────────────────
    ppl_n = min(C.EVAL["ppl_subsample"], n)
    print(f"  Step 2/2 — Perplexity (n={ppl_n})...")
    ppl = compute_perplexity(
        model, tokenizer,
        questions[:ppl_n],
        gold[:ppl_n],
    )

    # ── Samples ───────────────────────────────────────────────────────────
    samples = [
        {
            "question":   questions[i],
            "gold":       gold[i],
            "prediction": predictions[i],
            "em":         em_scores[i],
            "f1":         round(f1_scores[i], 4),
        }
        for i in range(min(10, len(questions)))
    ]

    result = {
        "label":            label,
        "model_name":       model_name,
        "checkpoint_path":  load_path,
        "n_examples":       len(questions),
        "source_filter":    source_filter,
        "exact_match":      em,
        "token_f1":         f1,
        "perplexity":       ppl,
        "source_breakdown": src,
        "samples":          samples,
    }

    print(f"\n  ── Results: {label} ──")
    print(f"    Token F1    : {f1:.2f}%")
    print(f"    Exact Match : {em:.2f}%")
    print(f"    Perplexity  : {ppl:.4f}")

    out = results_path(label)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out}")

    del model
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def update_summary(result: dict):
    summary_file = os.path.join(C.RESULTS_DIR, "summary.json")
    summary = []
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)

    summary = [r for r in summary if r.get("label") != result["label"]]
    summary.append({k: result[k] for k in
                    ("label", "model_name", "source_filter",
                     "exact_match", "token_f1", "perplexity", "n_examples")})

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)


def print_summary(summary: list[dict]):
    print(f"\n{'='*77}")
    print(f"  LEADERBOARD  ({C.TELEEVAL_DATASET})")
    print(f"{'='*77}")
    print(f"  {'Model':<32} {'Filter':<12} {'F1':>8} {'EM':>8} {'PPL':>10}")
    print(f"  {'-'*75}")
    for r in sorted(summary, key=lambda x: (-x.get("token_f1", 0))):
        src = r.get("source_filter", "all")
        print(f"  {r['label']:<32} {src:<12} "
              f"{r['token_f1']:>7.2f}% "
              f"{r['exact_match']:>7.2f}% "
              f"{r['perplexity']:>10.4f}")
    print(f"{'='*77}\n")


def load_and_print_summary():
    summary_file = os.path.join(C.RESULTS_DIR, "summary.json")
    if not os.path.exists(summary_file):
        print(f"  No summary found at {summary_file}. Run eval first.")
        return
    with open(summary_file) as f:
        print_summary(json.load(f))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SLMs on Tele-Eval"
    )
    parser.add_argument("--model",
                        default=C.ACTIVE_MODEL,
                        help=f"Base model ID (config default: {C.ACTIVE_MODEL})")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to fine-tuned checkpoint. "
                             "If omitted, evaluates --model directly as baseline.")
    parser.add_argument("--label",
                        default=None,
                        help="Display name in leaderboard (auto-generated if omitted)")
    parser.add_argument("--n",
                        type=int, default=C.EVAL["n_examples"])
    parser.add_argument("--seed",
                        type=int, default=C.EVAL["seed"])
    parser.add_argument("--source-filter",
                        default=C.EVAL["source_filter"],
                        dest="source_filter",
                        choices=["all", "standard", "arxiv", "wiki"],
                        help="Filter Tele-Eval by question source (default: %(default)s)")
    parser.add_argument("--summary-only",
                        action="store_true", dest="summary_only",
                        help="Print leaderboard from saved results and exit")
    args = parser.parse_args()

    if args.summary_only:
        load_and_print_summary()
        return

    # Resolve load path and label
    load_path  = args.checkpoint if args.checkpoint else args.model
    model_name = args.model

    if args.label:
        label = args.label
    elif args.checkpoint:
        label = model_slug(args.checkpoint)
    else:
        label = model_slug(args.model) + "-base"

    result = evaluate(
        load_path     = load_path,
        model_name    = model_name,
        label         = label,
        n             = args.n,
        seed          = args.seed,
        source_filter = args.source_filter,
    )
    update_summary(result)


if __name__ == "__main__":
    main()