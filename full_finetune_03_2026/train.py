"""
tele-SLMs — Full Fine-Tuning on TeleSpec-Data
===============================================
Full fine-tuning (no LoRA/PEFT) of small language models on the
nareshmodina/TeleSpec-Data dataset (38k 3GPP + ETSI documents).

All defaults are read from config.py. CLI flags override config values.

Usage:
    # Use defaults from config.py
    python train.py

    # Override model and learning rate
    python train.py --model HuggingFaceTB/SmolLM2-360M --lr 5e-5

    # Multi-GPU (3x L40S)
    python -m torch.distributed.run --nproc_per_node=3 --master_port=29502 \
        train.py

Outputs:
    ./checkpoints/{model-slug}-telespec/   weights + tokenizer
    ./checkpoints/{model-slug}-telespec/trainer_state.json
"""

import argparse
import os

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

import config as C

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def checkpoint_dir(model_name: str) -> str:
    return os.path.join(C.CHECKPOINTS_DIR, model_slug(model_name) + "-telespec")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_dataset(tokenized_path: str):
    """
    Load the pre-tokenized and packed dataset from disk.
    Run tokenize_dataset.py first to produce this dataset.
    """
    from datasets import load_from_disk

    train_path = os.path.join(tokenized_path, "train")
    eval_path  = os.path.join(tokenized_path, "eval")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Pre-tokenized dataset not found at {train_path}\n"
            f"Run first: python tokenize_dataset.py --model <model_name>"
        )

    print(f"\n  Loading pre-tokenized dataset from {tokenized_path}...")
    train_ds = load_from_disk(train_path)
    eval_ds  = load_from_disk(eval_path)

    print(f"  Train blocks : {len(train_ds):,}")
    print(f"  Eval blocks  : {len(eval_ds):,}")
    print(f"  Block length : {len(train_ds[0]['input_ids'])} tokens")

    return train_ds, eval_ds



def train(args):
    out_dir = checkpoint_dir(args.model)
    slug    = model_slug(args.model)

    print(f"\n{'='*60}")
    print(f"  TRAINING  : {slug}")
    print(f"  Dataset   : {args.tokenized_path}")
    print(f"  Epochs    : {args.epochs}  |  LR: {args.lr}")
    print(f"  Output    : {out_dir}")
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
    print(f"  Device    : {device_type.upper()} ({n_gpus} GPU(s))")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Dataset ──────────────────────────────────────────────────────────
    train_ds, eval_ds = build_dataset(args.tokenized_path)

    # ── Model ────────────────────────────────────────────────────────────
    print(f"  Loading model: {args.model}")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype               = dtype,
        attn_implementation = attn_impl,
        device_map          = None,
    )
    model.config.use_cache = False
    if device_type == "cuda":
        model = model.to(f"cuda:{local_rank}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params/1e6:.1f}M (all trainable — full fine-tuning)")

    # ── Effective batch size ──────────────────────────────────────────────
    grad_accum = max(1, C.TARGET_EFFECTIVE_BS // (args.per_device_bs * max(n_gpus, 1)))
    eff_bs     = args.per_device_bs * grad_accum * max(n_gpus, 1)
    print(f"  Effective batch size : {eff_bs} "
          f"({args.per_device_bs} × {grad_accum} accum × {max(n_gpus,1)} GPU)")

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = out_dir,
        num_train_epochs            = args.epochs,
        max_steps                   = args.max_steps,
        per_device_train_batch_size = args.per_device_bs,
        per_device_eval_batch_size  = args.per_device_bs,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = C.TRAIN["lr_scheduler"],
        warmup_ratio                = C.TRAIN["warmup_ratio"],
        weight_decay                = C.TRAIN["weight_decay"],
        bf16                        = (dtype == torch.bfloat16),
        fp16                        = False,
        logging_steps               = C.TRAIN["logging_steps"],
        eval_strategy               = "steps",
        eval_steps                  = min(C.TRAIN["eval_steps"], max(10, args.max_steps // 2)) if args.max_steps > 0 else C.TRAIN["eval_steps"],
        save_strategy               = "steps",
        save_steps                  = min(C.TRAIN["save_steps"], max(10, args.max_steps // 2)) if args.max_steps > 0 else C.TRAIN["save_steps"],
        save_total_limit            = C.TRAIN["save_total_limit"],
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        report_to                   = C.REPORT_TO,
        dataloader_num_workers      = C.TRAIN["dataloader_workers"],
        remove_unused_columns       = False,
        ddp_find_unused_parameters  = False,
        gradient_checkpointing      = args.gradient_checkpointing,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    print(f"\n  Starting training...")
    trainer.train()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    print(f"\n  Training complete. Checkpoint → {out_dir}")
    print(f"  Next step — instruction fine-tuning:")
    print(f"    python sft.py --input {out_dir}\n")
    print(f"  Then evaluate with:")
    print(f"    python eval.py --checkpoint {out_dir}-sft-merged --model {args.model}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full fine-tuning of SLMs on TeleSpec-Data"
    )
    # All defaults come from config.py
    parser.add_argument("--model",
                        default=C.ACTIVE_MODEL,
                        help=f"HuggingFace model ID (config default: {C.ACTIVE_MODEL})")
    parser.add_argument("--tokenized-path",
                        default=C.TOKENIZED_DATASET_DIR,
                        dest="tokenized_path",
                        help=f"Path to pre-tokenized dataset (config default: {C.TOKENIZED_DATASET_DIR})")
    parser.add_argument("--epochs",
                        type=int, default=C.TRAIN["epochs"])
    parser.add_argument("--lr",
                        type=float, default=None,
                        help="Learning rate (default: per-model value from config.LR_OVERRIDES)")
    parser.add_argument("--per-device-bs",
                        type=int, default=C.TRAIN["per_device_bs"],
                        dest="per_device_bs")
    parser.add_argument("--max-steps",
                        type=int, default=-1,
                        dest="max_steps",
                        help="Hard stop after N steps (-1 = disabled, trains full epochs)")
    parser.add_argument("--gradient-checkpointing",
                        action="store_true",
                        default=C.TRAIN["gradient_checkpointing"],
                        dest="gradient_checkpointing")

    args = parser.parse_args()

    # Resolve LR: CLI > config LR_OVERRIDES > LR_DEFAULT
    if args.lr is None:
        args.lr = C.get_lr(args.model)

    train(args)


if __name__ == "__main__":
    main()