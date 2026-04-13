"""
tele-SLMs — Tele-Eval Benchmark
=================================
Evaluates a model on Tele-Eval using the two automated metrics from:
  Maatouk et al. (2024) "Tele-LLMs: A Series of Specialized Large Language
  Models for Telecommunications" — arXiv:2409.05314

Metrics implemented:
  1. Ans-PPL  — answer perplexity: how surprised the model is by the ground
                truth answer given the question. Lower is better.
  2. SemScore — cosine similarity between sentence embeddings of the model's
                generated output and the gold answer, using all-mpnet-base-v2.
                Higher is better (range: -1 to 1).

LLM-Eval (the third metric from the paper) is excluded as it requires a
separate Mixtral-8x7B judge model.

Answering format follows Appendix C of the paper:
  "The following is a question about telecommunications and networking.
   Question: {statement}
   Answer:"

Generation: greedy decoding, max 100 new tokens (as per paper Section VII).

Usage:
  # Standards-only eval (primary metric — matches training domain)
  python benchmark.py \\
      --model  HuggingFaceTB/SmolLM2-135M \\
      --checkpoint ./checkpoints/SmolLM2-135M-telespec-sft-merged \\
      --filter standard \\
      --n      2000

  # Full Tele-Eval
  python benchmark.py \\
      --model  HuggingFaceTB/SmolLM2-135M \\
      --checkpoint ./checkpoints/SmolLM2-135M-telespec-sft-merged \\
      --filter all \\
      --n      2000

  # Base model (no checkpoint — evaluate from HuggingFace weights directly)
  python benchmark.py \\
      --model  HuggingFaceTB/SmolLM2-135M \\
      --filter standard \\
      --n      2000

  # Print saved leaderboard only
  python benchmark.py --summary-only

Outputs:
  ./results/benchmark_{label}_{timestamp}.json   full results + samples
  ./results/benchmark_summary.json               leaderboard across all runs
"""

import argparse
import json
import os
import math
import random
from datetime import datetime

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import logging

import config as C

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TELEEVAL_DATASET   = "AliMaatouk/Tele-Eval"
SEMSCORE_MODEL     = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS     = 100
RESULTS_DIR        = "./results"
SUMMARY_PATH       = os.path.join(RESULTS_DIR, "benchmark_summary.json")
RANDOM_SEED        = 42

ANSWER_PROMPT = (
    "The following is a question about telecommunications and networking.\n"
    "Question: {statement}\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Metric 1 — Answer Perplexity
# ---------------------------------------------------------------------------

def compute_ans_ppl(model, tokenizer, questions, answers, device, batch_size=8):
    """
    Compute answer perplexity as defined in the Tele-LLMs paper (eq. 5):
    PPL of the model on the ground truth answer, conditioned on the question.
    The loss is computed only over the answer tokens, not the question tokens.
    """
    model.eval()
    total_log_prob = 0.0
    total_tokens   = 0

    for i in tqdm(range(0, len(questions), batch_size), desc="  Ans-PPL", ncols=100, dynamic_ncols=False):
        batch_q = questions[i : i + batch_size]
        batch_a = answers[i : i + batch_size]

        for q, a in zip(batch_q, batch_a):
            prompt   = ANSWER_PROMPT.format(statement=q)
            full_text = prompt + " " + a

            prompt_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=512
            ).input_ids.to(device)

            full_ids = tokenizer(
                full_text, return_tensors="pt", truncation=True, max_length=640
            ).input_ids.to(device)

            # Number of prompt tokens (ki in the paper)
            k = prompt_ids.shape[1]
            T = full_ids.shape[1]

            # Skip if answer is empty after tokenization
            if T <= k:
                continue

            with torch.no_grad():
                outputs = model(full_ids, labels=full_ids)
                # Recompute loss only over answer tokens
                logits  = outputs.logits  # (1, T, vocab)
                # Shift: predict token t from tokens 0..t-1
                shift_logits = logits[0, :-1, :]          # (T-1, vocab)
                shift_labels = full_ids[0, 1:]             # (T-1,)
                # Only keep answer positions
                shift_logits = shift_logits[k-1:]          # answer tokens
                shift_labels = shift_labels[k-1:]

                log_probs = torch.nn.functional.cross_entropy(
                    shift_logits, shift_labels, reduction="sum"
                )
                n_answer_tokens = shift_labels.shape[0]

            total_log_prob += log_probs.item()
            total_tokens   += n_answer_tokens

    if total_tokens == 0:
        return float("inf")

    avg_neg_log_prob = total_log_prob / total_tokens
    return math.exp(avg_neg_log_prob)


# ---------------------------------------------------------------------------
# Metric 2 — SemScore
# ---------------------------------------------------------------------------

def compute_semscore(generated, references, sem_model):
    """
    SemScore = mean cosine similarity between sentence embeddings of
    the generated answers and gold answers, using all-mpnet-base-v2.
    """
    print("  Computing SemScore embeddings...")
    gen_embeddings = sem_model.encode(generated,  batch_size=64,
                                      show_progress_bar=True,
                                      convert_to_numpy=True)
    ref_embeddings = sem_model.encode(references, batch_size=64,
                                      show_progress_bar=True,
                                      convert_to_numpy=True)

    scores = []
    for g, r in zip(gen_embeddings, ref_embeddings):
        score = cosine_similarity(g.reshape(1, -1), r.reshape(1, -1))[0][0]
        scores.append(float(score))

    return sum(scores) / len(scores), scores


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answers(model, tokenizer, questions, device):
    """
    Generate answers using greedy decoding, max 100 new tokens.
    Format matches Appendix C of the Tele-LLMs paper.
    """
    model.eval()
    generated = []

    for q in tqdm(questions, desc="  Generating answers", ncols=100, dynamic_ncols=False, miniters=10):
        prompt = ANSWER_PROMPT.format(statement=q)
        inputs = tokenizer(
            prompt,
            return_tensors     = "pt",
            truncation         = True,
            max_length         = 512,
            padding            = False,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens  = MAX_NEW_TOKENS,
                do_sample       = False,   # greedy
                temperature     = 1.0,
                pad_token_id    = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids  = output_ids[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        generated.append(response)

    return generated


# ---------------------------------------------------------------------------
# Main benchmark function
# ---------------------------------------------------------------------------

def benchmark(model_name: str, load_path: str, label: str,
              n: int, seed: int, source_filter: str):

    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"  BENCHMARK : {label}")
    print(f"  Model     : {model_name}")
    print(f"  Checkpoint: {load_path}")
    print(f"  Filter    : {source_filter}")
    print(f"  N         : {n}")
    print(f"{'='*60}\n")

    # ── Load Tele-Eval ────────────────────────────────────────────────────
    print("  Loading Tele-Eval...")
    ds = load_dataset(TELEEVAL_DATASET, split="data")

    if source_filter != "all":
        ds = ds.filter(lambda x: x["id"].startswith(source_filter + "_"))
        print(f"  After filter ({source_filter}): {len(ds):,} examples")

    ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    print(f"  Evaluating on {len(ds):,} examples\n")

    questions  = [ex["Statement"] for ex in ds]
    references = [ex["Answer"]    for ex in ds]

    # ── Load model ────────────────────────────────────────────────────────
    print(f"  Loading model from: {load_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype = dtype,
        device_map  = "auto",
    )
    model.eval()
    print(f"  Model loaded on {device} ({dtype})\n")

    # ── Load SemScore model ───────────────────────────────────────────────
    print(f"  Loading sentence transformer: {SEMSCORE_MODEL}")
    sem_model = SentenceTransformer(SEMSCORE_MODEL)

    # ── Generate answers ──────────────────────────────────────────────────
    print("\n  [1/3] Generating answers (greedy, max 100 tokens)...")
    generated = generate_answers(model, tokenizer, questions, device)

    # ── Ans-PPL ───────────────────────────────────────────────────────────
    print("\n  [2/3] Computing Ans-PPL...")
    ans_ppl = compute_ans_ppl(model, tokenizer, questions, references, device)
    print(f"  Ans-PPL: {ans_ppl:.4f}")

    # ── SemScore ──────────────────────────────────────────────────────────
    print("\n  [3/3] Computing SemScore...")
    mean_semscore, semscore_list = compute_semscore(generated, references, sem_model)
    print(f"  SemScore: {mean_semscore:.4f}")

    # ── Save results ──────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples = [
        {
            "question":   q,
            "gold":       r,
            "generated":  g,
            "semscore":   s,
        }
        for q, r, g, s in zip(questions[:20], references[:20],
                               generated[:20], semscore_list[:20])
    ]

    result = {
        "label":         label,
        "model_name":    model_name,
        "checkpoint":    load_path,
        "source_filter": source_filter,
        "n_examples":    len(questions),
        "seed":          seed,
        "timestamp":     timestamp,
        "ans_ppl":       ans_ppl,
        "semscore":      mean_semscore,
        "samples":       samples,
    }

    result_path = os.path.join(
        RESULTS_DIR, f"benchmark_{label}_{timestamp}.json"
    )
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved → {result_path}")

    update_summary(result)
    return result


# ---------------------------------------------------------------------------
# Summary / leaderboard
# ---------------------------------------------------------------------------

def update_summary(result: dict):
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH) as f:
            summary = json.load(f)
    else:
        summary = []

    # Replace existing entry with same label + filter, or append
    key = (result["label"], result["source_filter"])
    summary = [
        r for r in summary
        if (r["label"], r["source_filter"]) != key
    ]
    summary.append({
        k: result[k] for k in
        ("label", "model_name", "source_filter",
         "n_examples", "ans_ppl", "semscore", "timestamp")
    })

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)


def print_summary():
    if not os.path.exists(SUMMARY_PATH):
        print("  No benchmark results found.")
        return

    with open(SUMMARY_PATH) as f:
        summary = json.load(f)

    print(f"\n{'='*75}")
    print(f"  BENCHMARK LEADERBOARD  ({TELEEVAL_DATASET})")
    print(f"{'='*75}")
    print(f"  {'Model':<36} {'Filter':<12} {'Ans-PPL':>10} {'SemScore':>10}")
    print(f"  {'-'*70}")

    # Sort by SemScore descending within each filter group
    for filt in ["standard", "all"]:
        group = [r for r in summary if r.get("source_filter") == filt]
        group = sorted(group, key=lambda x: -x.get("semscore", 0))
        for r in group:
            print(
                f"  {r['label']:<36} {r['source_filter']:<12} "
                f"{r['ans_ppl']:>10.4f} {r['semscore']:>10.4f}"
            )
        if group:
            print()

    print(f"{'='*75}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tele-Eval benchmark using Ans-PPL and SemScore metrics "
                    "(Maatouk et al. 2024)"
    )
    parser.add_argument("--model",
                        default=C.ACTIVE_MODEL,
                        help="HuggingFace model ID for tokenizer")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to fine-tuned checkpoint (default: use base model)")
    parser.add_argument("--label",
                        default=None,
                        help="Display name for this run in the leaderboard")
    parser.add_argument("--n",
                        type=int, default=C.BENCHMARK["n_examples"],
                        help="Number of Tele-Eval examples to evaluate")
    parser.add_argument("--seed",
                        type=int, default=RANDOM_SEED)
    parser.add_argument("--filter",
                        default="standard",
                        choices=["all", "standard", "arxiv", "wiki"],
                        dest="source_filter",
                        help="Source filter for Tele-Eval questions "
                             "(default: standard)")
    parser.add_argument("--log",
                        default=None,
                        help="Path to log file (output is tee'd to file and stdout)")
    parser.add_argument("--summary-only",
                        action="store_true", dest="summary_only",
                        help="Print saved leaderboard and exit")
    args = parser.parse_args()

    # ── Log file setup — tee output to file and stdout ──────────────────
    if hasattr(args, 'log') and args.log:
        import sys

        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()
            def fileno(self):
                return self.streams[0].fileno()
            def isatty(self):
                return False

        os.makedirs(os.path.dirname(args.log), exist_ok=True) if os.path.dirname(args.log) else None
        log_fh = open(args.log, "w")
        sys.stdout = Tee(sys.__stdout__, log_fh)
        print(f"  Logging to: {args.log}")

    if args.summary_only:
        print_summary()
    else:
        load_path = args.checkpoint if args.checkpoint else args.model

        if args.label:
            label = args.label
        elif args.checkpoint:
            slug = os.path.basename(args.checkpoint.rstrip("/"))
            label = f"{slug}-{args.source_filter}"
        else:
            from train import model_slug
            label = f"{model_slug(args.model)}-base-{args.source_filter}"

        result = benchmark(
            model_name    = args.model,
            load_path     = load_path,
            label         = label,
            n             = args.n,
            seed          = args.seed,
            source_filter = args.source_filter,
        )
        print_summary()