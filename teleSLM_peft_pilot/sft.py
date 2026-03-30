"""
Phase 2 — Instruction Fine-Tuning (SFT)
=========================================
Fine-tunes the Phase 1 merged model on Alpaca + Open-Instruct data
using LoRA + SFTTrainer. Loss is computed on completion tokens only
via SFTConfig(completion_only_loss=True) — the TRL 0.29 replacement
for the removed DataCollatorForCompletionOnlyLM.

Usage:
    # Single GPU
    python sft.py

    # Multi-GPU (3x L40S)
    python -m torch.distributed.run --nproc_per_node=3 --master_port=29501 sft.py

Key differences from Phase 1 (pretrain.py):
    - Input:  Phase 1 merged model (not base Qwen)
    - Data:   Alpaca + Open-Instruct — {"prompt": ..., "completion": ...} format
    - Loss:   Completion tokens only (completion_only_loss=True in SFTConfig)
    - LR:     5e-6 (lower than Phase 1)
    - Epochs: 2
"""

import os
import torch
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH  = "./checkpoints/stage1-merged"
OUTPUT_DIR  = "./checkpoints/stage2-lora"
MAX_LENGTH  = 1024
EVAL_RATIO  = 0.02
RANDOM_SEED = 42

# Alpaca-style prompt template
# Dataset uses "prompt" + "completion" fields
# SFTConfig(completion_only_loss=True) masks prompt tokens from the loss
PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

# LoRA — lighter than Phase 1
LORA_CONFIG = LoraConfig(
    task_type      = TaskType.CAUSAL_LM,
    r              = 32,
    lora_alpha     = 64,
    lora_dropout   = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias = "none",
)

# ---------------------------------------------------------------------------
# Data loading and formatting
# ---------------------------------------------------------------------------

def load_alpaca() -> Dataset:
    print("  Loading Alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  Alpaca: {len(ds):,} examples")
    return ds


def load_open_instruct() -> Dataset:
    print("  Loading Open-Instruct...")
    try:
        ds = load_dataset("allenai/open-instruct-v1", split="train")
        print(f"  Open-Instruct: {len(ds):,} examples")
        return ds
    except Exception:
        print("  [WARN] open-instruct-v1 unavailable, trying FLAN subset...")
        try:
            ds = load_dataset("Muennighoff/flan", split="train").select(range(200_000))
            print(f"  FLAN subset: {len(ds):,} examples")
            return ds
        except Exception as e:
            print(f"  [WARN] Could not load Open-Instruct/FLAN: {e}")
            return None


def format_alpaca(example: dict) -> dict:
    instruction = example["instruction"]
    if example.get("input", "").strip():
        instruction = f"{instruction}\n\n{example['input'].strip()}"
    return {
        "prompt":     PROMPT_TEMPLATE.format(instruction=instruction),
        "completion": example.get("output", "").strip(),
    }


def format_open_instruct(example: dict) -> dict | None:
    if "instruction" in example and "output" in example:
        instruction = example["instruction"].strip()
        response    = example["output"].strip()
    elif "inputs" in example and "targets" in example:
        instruction = example["inputs"].strip()
        response    = example["targets"].strip()
    elif "prompt" in example and "completion" in example:
        instruction = example["prompt"].strip()
        response    = example["completion"].strip()
    else:
        return None
    if not instruction or not response:
        return None
    return {
        "prompt":     PROMPT_TEMPLATE.format(instruction=instruction),
        "completion": response,
    }


def build_dataset(tokenizer) -> tuple[Dataset, Dataset]:
    alpaca_raw = load_alpaca()
    alpaca = alpaca_raw.map(
        format_alpaca,
        remove_columns=alpaca_raw.column_names,
        num_proc=4,
    )
    datasets = [alpaca]

    oi_raw = load_open_instruct()
    if oi_raw is not None:
        def _fmt(ex):
            result = format_open_instruct(ex)
            return result if result else {"prompt": "", "completion": ""}
        oi = oi_raw.map(_fmt, remove_columns=oi_raw.column_names, num_proc=4)
        oi = oi.filter(lambda x: len(x["completion"]) > 10)
        datasets.append(oi)

    combined = concatenate_datasets(datasets).shuffle(seed=RANDOM_SEED)

    print(f"  Filtering by length (max {MAX_LENGTH} tokens)...")
    def within_length(example):
        ids = tokenizer(
            example["prompt"] + example["completion"],
            truncation=False,
            add_special_tokens=False,
        )["input_ids"]
        return len(ids) <= MAX_LENGTH

    combined = combined.filter(within_length, num_proc=4)
    print(f"  Total after filtering: {len(combined):,} examples")

    split = combined.train_test_split(test_size=EVAL_RATIO, seed=RANDOM_SEED)
    return split["train"], split["test"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  Phase 2 — Instruction Fine-Tuning (SFT)")
    print("="*60)

    if torch.cuda.is_available():
        device_type = "cuda"
        n_gpus      = torch.cuda.device_count()
        dtype       = torch.bfloat16
        attn_impl   = "sdpa"
        print(f"  Device : CUDA ({n_gpus} GPU(s))")
    elif torch.backends.mps.is_available():
        device_type = "mps"
        n_gpus      = 1
        dtype       = torch.float16
        attn_impl   = "eager"
        print(f"  Device : MPS")
    else:
        device_type = "cpu"
        n_gpus      = 1
        dtype       = torch.float32
        attn_impl   = "eager"
        print(f"  Device : CPU")

    print(f"\n  Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("\n  Building dataset...")
    train_ds, eval_ds = build_dataset(tokenizer)
    print(f"  Train : {len(train_ds):,}")
    print(f"  Eval  : {len(eval_ds):,}")

    print(f"\n  Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype               = dtype,
        attn_implementation = attn_impl,
        device_map          = None,
    )
    model.config.use_cache = False
    # In DDP mode, Accelerate handles device placement — do NOT use device_map
    # Move model to local rank GPU manually
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if device_type == "cuda":
        model = model.to(f"cuda:{local_rank}")

    print("  Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    per_device_bs = 4
    grad_accum    = 4

    training_args = SFTConfig(
        output_dir              = OUTPUT_DIR,
        num_train_epochs        = 2,
        per_device_train_batch_size = per_device_bs,
        per_device_eval_batch_size  = per_device_bs,
        gradient_accumulation_steps = grad_accum,
        learning_rate           = 5e-6,
        lr_scheduler_type       = "cosine",
        warmup_steps            = 50,
        weight_decay            = 0.01,
        bf16                    = (dtype == torch.bfloat16),
        fp16                    = (dtype == torch.float16),
        logging_steps           = 10,
        eval_strategy           = "steps",
        eval_steps              = 200,
        save_strategy           = "steps",
        save_steps              = 200,
        save_total_limit        = 3,
        load_best_model_at_end  = True,
        metric_for_best_model   = "eval_loss",
        greater_is_better       = False,
        report_to               = "none",
        dataloader_num_workers  = 4,
        remove_unused_columns   = False,
        ddp_find_unused_parameters = False,
        # TRL 0.29: completion_only_loss replaces DataCollatorForCompletionOnlyLM
        # trains only on "completion" field, masks "prompt" tokens from loss
        completion_only_loss    = True,
        max_length              = MAX_LENGTH,
    )

    trainer = SFTTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = eval_ds,
    )

    print("\n  Starting Phase 2 SFT...")
    print(f"  Effective batch size : {per_device_bs * grad_accum * max(n_gpus, 1)}")
    print(f"  Max sequence length  : {MAX_LENGTH}")
    print(f"  Output dir           : {OUTPUT_DIR}\n")

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n  Phase 2 complete. Adapter saved -> {OUTPUT_DIR}")
    print("  Next: python merge_adapter.py --base-model ./checkpoints/stage1-merged "
          "--adapter ./checkpoints/stage2-lora --output ./checkpoints/stage2-merged\n")


if __name__ == "__main__":
    main()