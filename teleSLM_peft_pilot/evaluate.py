"""
Tele-Eval Evaluation Script
=============================
Evaluates models on the Tele-Eval benchmark (AliMaatouk/Tele-Eval).
Dataset: open-ended Q&A — fields: Statement, Answer, id
Task:    generate answer given Statement, compare to ground truth Answer

Metrics:
  - Exact match accuracy  (normalised: lowercase, strip punctuation/whitespace)
  - Token-level F1        (word overlap, standard for open QA)
  - Perplexity            (cross-entropy loss on ground truth answer tokens)

Models compared:
  - Base Qwen2.5-1.5B          (no training)
  - Stage 2 merged             (Phase 1 pretrain + Phase 2 SFT)

Usage:
    python evaluate.py \
        --stage2  ./checkpoints/stage2-merged \
        --base    Qwen/Qwen2.5-1.5B \
        --n       2000 \
        --output  ./eval_results.json \
        --seed    42

    # To evaluate only one model:
    python evaluate.py --stage2 ./checkpoints/stage2-merged --skip-base
"""

import argparse
import json
import math
import re
import string
from collections import Counter

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
MAX_NEW_TOKENS  = 200
BATCH_SIZE      = 8    # for perplexity computation
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Text normalisation (same as SQuAD official eval)
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, articles, and extra whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(pred: str, gold: str) -> int:
    return int(normalize(pred) == normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall    = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answers(model, tokenizer, questions: list[str],
                     use_template: bool = True) -> list[str]:
    """
    Generate answers for a list of questions.
    use_template=True  → Alpaca format (for SFT models)
    use_template=False → raw continuation (for base model)
    """
    answers = []
    model.eval()

    for q in tqdm(questions, desc="  generating", leave=False):
        if use_template:
            prompt = PROMPT_TEMPLATE.format(instruction=q)
        else:
            # Base model: just ask the question directly
            prompt = f"Question: {q}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=512).to(DEVICE)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens    = MAX_NEW_TOKENS,
                do_sample         = False,
                eos_token_id      = tokenizer.eos_token_id,
                pad_token_id      = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids  = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        # Trim at first newline for base model (stops at next "Question:")
        if not use_template:
            response = response.split("\n")[0].strip()

        answers.append(response.strip())

    return answers


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, questions: list[str],
                       gold_answers: list[str],
                       use_template: bool = True) -> float:
    """
    Compute average perplexity of the model on ground-truth answers.
    For SFT models: prompt = template + gold answer, loss on answer tokens only.
    For base model: prompt = Q + A, loss on all tokens.
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for i in range(0, len(questions), BATCH_SIZE):
        batch_q = questions[i : i + BATCH_SIZE]
        batch_a = gold_answers[i : i + BATCH_SIZE]

        for q, a in zip(batch_q, batch_a):
            if use_template:
                prompt   = PROMPT_TEMPLATE.format(instruction=q)
                full_seq = prompt + a
            else:
                full_seq = f"Question: {q}\nAnswer: {a}"

            enc_full   = tokenizer(full_seq,   return_tensors="pt",
                                   truncation=True, max_length=1024).to(DEVICE)
            enc_prompt = tokenizer(prompt if use_template else f"Question: {q}\nAnswer:",
                                   return_tensors="pt",
                                   truncation=True, max_length=512).to(DEVICE)

            input_ids = enc_full["input_ids"]
            labels    = input_ids.clone()

            # Mask prompt tokens from loss
            prompt_len = enc_prompt["input_ids"].shape[1]
            labels[:, :prompt_len] = -100

            if labels[:, prompt_len:].shape[1] == 0:
                continue  # answer was empty after truncation

            with torch.no_grad():
                out = model(**enc_full, labels=labels)

            n_answer_tokens = (labels != -100).sum().item()
            if n_answer_tokens > 0:
                total_loss   += out.loss.item() * n_answer_tokens
                total_tokens += n_answer_tokens

    avg_loss    = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity  = math.exp(avg_loss)
    return round(perplexity, 4)


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(model_path: str, questions: list[str],
                   gold_answers: list[str], use_template: bool,
                   label: str) -> dict:

    print(f"\n  {'='*50}")
    print(f"  Evaluating: {label}")
    print(f"  Path      : {model_path}")
    print(f"  Template  : {use_template}")
    print(f"  {'='*50}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype      = torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        device_map = "auto",
    )

    # ── Generation metrics ───────────────────────────────────────────────
    print("  Step 1/2 — Generating answers...")
    predictions = generate_answers(model, tokenizer, questions, use_template)

    em_scores = [exact_match(p, g) for p, g in zip(predictions, gold_answers)]
    f1_scores = [token_f1(p, g)    for p, g in zip(predictions, gold_answers)]

    em  = round(100 * sum(em_scores) / len(em_scores), 2)
    f1  = round(100 * sum(f1_scores) / len(f1_scores), 2)

    # ── Perplexity ───────────────────────────────────────────────────────
    print("  Step 2/2 — Computing perplexity...")
    ppl = compute_perplexity(model, tokenizer, questions, gold_answers, use_template)

    # ── Sample predictions ───────────────────────────────────────────────
    samples = []
    for i in range(min(5, len(questions))):
        samples.append({
            "question":   questions[i],
            "gold":       gold_answers[i],
            "prediction": predictions[i],
            "em":         em_scores[i],
            "f1":         round(f1_scores[i], 4),
        })

    results = {
        "model":       label,
        "model_path":  model_path,
        "n_examples":  len(questions),
        "exact_match": em,
        "token_f1":    f1,
        "perplexity":  ppl,
        "samples":     samples,
    }

    print(f"\n  Results for {label}:")
    print(f"    Exact Match  : {em:.2f}%")
    print(f"    Token F1     : {f1:.2f}%")
    print(f"    Perplexity   : {ppl:.2f}")

    # Free GPU memory before loading next model
    del model
    torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2",    default="./checkpoints/stage2-merged")
    parser.add_argument("--base",      default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--n",         type=int,  default=2000,
                        help="Number of examples to evaluate (max 750k)")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--output",    default="./eval_results.json")
    parser.add_argument("--skip-base", action="store_true",
                        help="Skip base model evaluation")
    args = parser.parse_args()

    # ── Load dataset ─────────────────────────────────────────────────────
    print(f"\n  Loading Tele-Eval (n={args.n})...")
    ds = load_dataset("AliMaatouk/Tele-Eval", split="data")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))

    questions    = ds["Statement"]
    gold_answers = ds["Answer"]
    ids          = ds["id"]

    print(f"  Loaded {len(questions)} examples")
    print(f"  Source breakdown: "
          f"standard={sum(1 for i in ids if 'standard' in i)}, "
          f"arxiv={sum(1 for i in ids if 'arxiv' in i)}, "
          f"wiki={sum(1 for i in ids if 'wiki' in i)}")

    all_results = []

    # ── Evaluate Stage 2 model ────────────────────────────────────────────
    r2 = evaluate_model(
        model_path   = args.stage2,
        questions    = questions,
        gold_answers = gold_answers,
        use_template = True,
        label        = "Stage2 (pretrain+SFT)",
    )
    all_results.append(r2)

    # ── Evaluate base model ───────────────────────────────────────────────
    if not args.skip_base:
        r_base = evaluate_model(
            model_path   = args.base,
            questions    = questions,
            gold_answers = gold_answers,
            use_template = False,
            label        = "Base Qwen2.5-1.5B",
        )
        all_results.append(r_base)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS  ({args.n} examples from Tele-Eval)")
    print(f"{'='*60}")
    print(f"  {'Model':<30} {'EM':>8} {'F1':>8} {'PPL':>10}")
    print(f"  {'-'*58}")
    for r in all_results:
        print(f"  {r['model']:<30} {r['exact_match']:>7.2f}% "
              f"{r['token_f1']:>7.2f}% {r['perplexity']:>10.2f}")
    print(f"{'='*60}\n")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "config": {
            "n_examples": args.n,
            "seed":       args.seed,
            "dataset":    "AliMaatouk/Tele-Eval",
        },
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
