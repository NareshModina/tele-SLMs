"""
tele-SLMs — Instruction Fine-Tuning (SFT)
==========================================
LoRA-based instruction fine-tuning on an Alpaca-style dataset.
Takes a pretrained checkpoint (or base model) as input and produces
a merged model ready for evaluation.

Run this after train.py to add chatbot-like behaviour on top of
domain-adapted weights. LoRA is used (not FFT) to preserve the
telecom knowledge acquired in Stage 1.

Two experimental conditions:
    A) pretrain → SFT   (domain knowledge + instruction following)
    B) base    → SFT   (instruction following only, no domain pretraining)

Comparing A vs B on Tele-Eval isolates the contribution of Stage 1.

Usage:
    # Condition A — SFT on top of Stage 1 checkpoint (default)
    python sft.py --input ./checkpoints/SmolLM2-135M-telespec

    # Condition B — SFT on top of base model
    python sft.py --input HuggingFaceTB/SmolLM2-135M --base

    # Multi-GPU
    python -m torch.distributed.run --nproc_per_node=3 --master_port=29503 \
        sft.py --input ./checkpoints/SmolLM2-135M-telespec

    # Smoke test
    python sft.py --input ./checkpoints/SmolLM2-135M-telespec --max-steps 50

Outputs:
    ./checkpoints/{slug}-sft/        LoRA adapter
    ./checkpoints/{slug}-sft-merged/ merged weights (used by eval.py)
"""

import argparse
import os

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

import config as C

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def model_slug(path: str) -> str:
    """
    ./checkpoints/SmolLM2-135M-telespec → SmolLM2-135M-telespec
    HuggingFaceTB/SmolLM2-135M         → SmolLM2-135M
    """
    return os.path.basename(path.rstrip("/")) if os.path.exists(path) \
        else path.split("/")[-1]


def adapter_dir(input_path: str, is_base: bool) -> str:
    slug   = model_slug(input_path)
    suffix = "-sft" if not is_base else "-sft"
    return os.path.join(C.CHECKPOINTS_DIR, slug + suffix)


def merged_dir(input_path: str) -> str:
    slug = model_slug(input_path)
    return os.path.join(C.CHECKPOINTS_DIR, slug + "-sft-merged")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_sft_dataset(dataset_id: str, eval_ratio: float,
                     max_length: int, tokenizer) -> tuple[Dataset, Dataset]:
    print(f"\n  Loading instruction dataset: {dataset_id}")

    # ── Load ─────────────────────────────────────────────────────────────
    if dataset_id == "tatsu-lab/alpaca":
        ds = load_dataset(dataset_id, split="train")
        print(f"  Loaded {len(ds):,} examples")

        def fmt(example):
            instruction = example["instruction"]
            if example.get("input", "").strip():
                instruction = f"{instruction}\n\n{example['input'].strip()}"
            return {
                "prompt":     PROMPT_TEMPLATE.format(instruction=instruction),
                "completion": example.get("output", "").strip(),
            }

        ds = ds.map(fmt, remove_columns=ds.column_names, num_proc=4)

    elif dataset_id == "allenai/open-instruct-v1":
        ds = load_dataset(dataset_id, split="train")
        print(f"  Loaded {len(ds):,} examples")

        def fmt(example):
            return {
                "prompt":     PROMPT_TEMPLATE.format(
                                  instruction=example.get("instruction", "").strip()),
                "completion": example.get("output", "").strip(),
            }

        ds = ds.map(fmt, remove_columns=ds.column_names, num_proc=4)

    elif dataset_id == "Muennighoff/flan":
        # FLAN has a shard mismatch issue — load with verification disabled
        ds = load_dataset(dataset_id, split="train",
                          verification_mode="no_checks").select(range(200_000))
        print(f"  Loaded {len(ds):,} examples (capped at 200k)")

        def fmt(example):
            return {
                "prompt":     PROMPT_TEMPLATE.format(
                                  instruction=example.get("inputs", "").strip()),
                "completion": example.get("targets", "").strip(),
            }

        ds = ds.map(fmt, remove_columns=ds.column_names, num_proc=4)

    else:
        # Generic fallback — assumes instruction/output schema
        print(f"  [INFO] Unknown dataset schema, attempting generic load...")
        ds = load_dataset(dataset_id, split="train")
        print(f"  Loaded {len(ds):,} examples")
        cols = ds.column_names

        def fmt(example):
            instruction = (example.get("instruction")
                           or example.get("prompt")
                           or example.get("inputs", "")).strip()
            completion  = (example.get("output")
                           or example.get("completion")
                           or example.get("targets", "")).strip()
            return {"prompt": PROMPT_TEMPLATE.format(instruction=instruction),
                    "completion": completion}

        ds = ds.map(fmt, remove_columns=cols, num_proc=4)

    # ── Filter empty completions ──────────────────────────────────────────
    ds = ds.filter(lambda x: len(x["completion"].strip()) > 10)

    # ── Length filter ─────────────────────────────────────────────────────
    print(f"  Filtering by length (max {max_length} tokens)...")

    def within_length(example):
        ids = tokenizer(
            example["prompt"] + example["completion"],
            truncation=False, add_special_tokens=False,
        )["input_ids"]
        return len(ids) <= max_length

    ds = ds.filter(within_length, num_proc=4)
    ds = ds.shuffle(seed=C.TRAIN["seed"])

    # ── Split ─────────────────────────────────────────────────────────────
    split    = ds.train_test_split(test_size=eval_ratio, seed=C.TRAIN["seed"])
    train_ds = split["train"]
    eval_ds  = split["test"]

    print(f"  Train : {len(train_ds):,}  |  Eval : {len(eval_ds):,}")
    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# SFT
# ---------------------------------------------------------------------------

def sft(args):
    out_adapter = adapter_dir(args.input, args.base)
    out_merged  = merged_dir(args.input)
    slug        = model_slug(args.input)

    print(f"\n{'='*60}")
    print(f"  SFT : {slug}")
    print(f"  Input     : {args.input}")
    print(f"  Dataset   : {args.dataset}")
    print(f"  Adapter → : {out_adapter}")
    print(f"  Merged  → : {out_merged}")
    print(f"{'='*60}")

    # ── Device ───────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device_type = "cuda"
        n_gpus      = torch.cuda.device_count()
        dtype       = torch.bfloat16
        attn_impl   = "sdpa"
    else:
        device_type = "cpu"
        n_gpus      = 1
        dtype       = torch.float32
        attn_impl   = "eager"
    print(f"  Device : {device_type.upper()} ({n_gpus} GPU(s))")

    # ── Tokenizer ────────────────────────────────────────────────────────
    print(f"  Loading tokenizer from: {args.input}")
    tokenizer = AutoTokenizer.from_pretrained(args.input)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ──────────────────────────────────────────────────────────
    train_ds, eval_ds = load_sft_dataset(
        dataset_id = args.dataset,
        eval_ratio = C.SFT["eval_ratio"],
        max_length = args.max_length,
        tokenizer  = tokenizer,
    )

    # ── Model ────────────────────────────────────────────────────────────
    print(f"  Loading model from: {args.input}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = AutoModelForCausalLM.from_pretrained(
        args.input,
        dtype               = dtype,
        attn_implementation = attn_impl,
        device_map          = None,
    )
    model.config.use_cache = False
    if device_type == "cuda":
        model = model.to(f"cuda:{local_rank}")

    # ── LoRA ─────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = C.SFT["lora_r"],
        lora_alpha     = C.SFT["lora_alpha"],
        lora_dropout   = C.SFT["lora_dropout"],
        target_modules = C.SFT["target_modules"],
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Batch size ────────────────────────────────────────────────────────
    per_device_bs = args.per_device_bs
    grad_accum    = max(1, C.TARGET_EFFECTIVE_BS // (per_device_bs * max(n_gpus, 1)))
    eff_bs        = per_device_bs * grad_accum * max(n_gpus, 1)
    print(f"  Effective batch size : {eff_bs} "
          f"({per_device_bs} × {grad_accum} accum × {max(n_gpus,1)} GPU)")

    # Adjust eval/save steps for short smoke test runs
    eval_steps = (min(C.SFT["eval_steps"], max(10, args.max_steps // 2))
                  if args.max_steps > 0 else C.SFT["eval_steps"])
    save_steps = eval_steps

    # ── SFTConfig ─────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir                  = out_adapter,
        num_train_epochs            = args.epochs,
        max_steps                   = args.max_steps,
        per_device_train_batch_size = per_device_bs,
        per_device_eval_batch_size  = per_device_bs,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = C.SFT["lr_scheduler"],
        warmup_ratio                = C.SFT["warmup_ratio"],
        weight_decay                = C.SFT["weight_decay"],
        bf16                        = (dtype == torch.bfloat16),
        fp16                        = False,
        logging_steps               = C.SFT["logging_steps"],
        eval_strategy               = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = save_steps,
        save_total_limit            = C.SFT["save_total_limit"],
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = C.REPORT_TO,
        dataloader_num_workers      = C.TRAIN["dataloader_workers"],
        remove_unused_columns       = False,
        ddp_find_unused_parameters  = False,
        # Completion-only loss — trains on response tokens only
        completion_only_loss        = True,
        max_length                  = args.max_length,
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model         = model,
        args          = sft_config,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n  Starting SFT — {args.epochs} epoch(s), LR={args.lr}")
    trainer.train()
    trainer.save_model(out_adapter)
    tokenizer.save_pretrained(out_adapter)
    print(f"  Adapter saved → {out_adapter}")

    # ── Merge adapter into base weights ───────────────────────────────────
    print(f"\n  Merging adapter into base weights...")
    merged = model.merge_and_unload()
    merged.save_pretrained(out_merged)
    tokenizer.save_pretrained(out_merged)
    print(f"  Merged model saved → {out_merged}")

    print(f"\n  SFT complete.")
    print(f"  Evaluate with:")
    print(f"    python eval.py --checkpoint {out_merged} --model {args.input}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA instruction fine-tuning (SFT) for tele-SLMs"
    )
    parser.add_argument("--input",     required=True,
                        help="Path to Stage 1 checkpoint or base model HF ID")
    parser.add_argument("--base",      action="store_true",
                        help="Flag that --input is a base model (affects output naming)")
    parser.add_argument("--dataset",   default=C.SFT["dataset"],
                        help=f"Instruction dataset (config default: {C.SFT['dataset']})")
    parser.add_argument("--epochs",    type=int,   default=C.SFT["epochs"])
    parser.add_argument("--lr",        type=float, default=C.SFT["lr"])
    parser.add_argument("--per-device-bs", type=int, default=C.SFT["per_device_bs"],
                        dest="per_device_bs")
    parser.add_argument("--max-length",    type=int, default=C.SFT["max_length"],
                        dest="max_length")
    parser.add_argument("--max-steps",     type=int, default=C.SFT["max_steps"],
                        dest="max_steps",
                        help="Hard stop after N steps (-1 = full training)")
    args = parser.parse_args()
    sft(args)


if __name__ == "__main__":
    main()
