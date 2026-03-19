"""
Stage 1 - Continual Pretraining with LoRA
==========================================
Compatible with TRL 0.29+

Default model : Qwen/Qwen2.5-1.5B (no access request needed)
For real run  : swap to meta-llama/Llama-3.2-3B once access granted

Requirements:
    pip install torch transformers datasets peft trl accelerate

Smoke test (Mac / single GPU):
    python pretrain.py --smoke-test

Full run on 3x GPU:
    torchrun --nproc_per_node=3 pretrain.py
"""

import argparse
import json
import os
from dataclasses import dataclass, field

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PretrainConfig:

    # Model — swap to meta-llama/Llama-3.2-3B for real training
    model_name: str = "Qwen/Qwen2.5-1.5B"

    # Data
    dataset_path: str = "./tele-tokenized"
    max_length: int = 2048

    # Output
    output_dir: str = "./checkpoints/stage1-lora"
    run_name: str = "tele-pretrain-lora"

    # LoRA
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    num_epochs: int = 2
    per_device_train_batch_size: int = 4   # L40S 45GB can handle 4 with LoRA
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 100        # ~3% of a typical epoch
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Logging & Saving
    logging_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200  # must be multiple of eval_steps when load_best_model_at_end=True
    save_total_limit: int = 3

    # Performance
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device_config():
    if torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            attn_impl = "sdpa"  # built into PyTorch 2.0+, no extra install needed
        return torch.bfloat16, attn_impl, "CUDA"
    elif torch.backends.mps.is_available():
        return torch.float16, "eager", "MPS (Apple Silicon)"
    else:
        return torch.float32, "eager", "CPU"


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogCallback(TrainerCallback):
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not logs:
            return
        step  = state.global_step
        total = state.max_steps or 1
        pct   = 100 * step / total

        parts = [f"step {step}/{total} ({pct:.1f}%)"]
        for k in ["loss", "eval_loss", "learning_rate", "grad_norm"]:
            if k in logs:
                v = logs[k]
                parts.append(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}")
        print("  " + "  |  ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(smoke_test: bool = False):
    cfg = PretrainConfig()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main    = local_rank == 0

    dtype, attn_impl, device_label = get_device_config()
    use_bf16 = torch.cuda.is_available()
    use_tf32 = torch.cuda.is_available()

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Stage 1 - Continual Pretraining (LoRA)")
        print(f"{'='*60}")
        print(f"  Model       : {cfg.model_name}")
        print(f"  Dataset     : {cfg.dataset_path}")
        print(f"  Output      : {cfg.output_dir}")
        print(f"  Device      : {device_label}")
        print(f"  dtype       : {dtype}")
        print(f"  Attention   : {attn_impl}")
        print(f"  LoRA rank   : {cfg.lora_r}")
        print(f"  Max length  : {cfg.max_length}")
        print(f"  Epochs      : {cfg.num_epochs}")
        print(f"  Smoke test  : {smoke_test}")
        print()

    # Tokenizer
    if is_main:
        print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Dataset
    # Load on ALL ranks but only print on rank 0
    # dataset is raw text — SFTTrainer tokenizes+packs with packing=True
    if is_main:
        print("  Loading dataset...")
    train_ds = load_from_disk(os.path.join(cfg.dataset_path, "train"))
    eval_ds  = load_from_disk(os.path.join(cfg.dataset_path, "eval"))

    if smoke_test:
        train_ds = train_ds.select(range(200))
        eval_ds  = eval_ds.select(range(50))
        if is_main:
            print(f"  [SMOKE TEST] {len(train_ds)} train / {len(eval_ds)} eval")

    if is_main:
        print(f"  Train : {len(train_ds):,}")
        print(f"  Eval  : {len(eval_ds):,}")

    # Model
    if is_main:
        print(f"\n  Loading model: {cfg.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        dtype=dtype,
        use_cache=False,
        attn_implementation=attn_impl,
    )

    # LoRA
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if is_main:
        trainable, total = model.get_nb_trainable_parameters()
        print(f"  Trainable : {trainable:,}  ({100*trainable/total:.2f}% of {total:,})")

    # SFTConfig — only args valid in TRL 0.29
    training_args = SFTConfig(
        # Output
        output_dir=cfg.output_dir,
        run_name=cfg.run_name,

        # Duration
        num_train_epochs=cfg.num_epochs if not smoke_test else 1,
        max_steps=10 if smoke_test else -1,

        # Batch — use smaller batch on non-CUDA to avoid OOM
        per_device_train_batch_size=1 if not torch.cuda.is_available() else cfg.per_device_train_batch_size,
        per_device_eval_batch_size=1 if not torch.cuda.is_available() else cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        # Sequence - dataset is pre-tokenized and pre-packed into exact max_length blocks
        # no padding, no packing needed at training time
        max_length=cfg.max_length,
        dataset_text_field=None,
        packing=False,

        # Optimiser
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",

        # Precision
        bf16=use_bf16,
        tf32=use_tf32,

        # Memory
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        # Logging
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        report_to="none",

        # Eval
        eval_strategy="steps",
        eval_steps=cfg.eval_steps if not smoke_test else 5,

        # Saving
        save_strategy="steps",
        save_steps=cfg.save_steps if not smoke_test else 5,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Dataloader
        dataloader_num_workers=cfg.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[LogCallback()],
    )

    # Train
    if is_main:
        print(f"\n  Starting training...\n")
    train_result = trainer.train()

    # Save
    if is_main:
        print(f"\n  Saving adapter to {cfg.output_dir}...")
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    if is_main:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        with open(os.path.join(cfg.output_dir, "train_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Train loss  : {metrics.get('train_loss', 0):.4f}")
        print(f"  Runtime     : {metrics.get('train_runtime', 0)/3600:.2f} hours")
        print(f"  Samples/sec : {metrics.get('train_samples_per_second', 0):.1f}")
        print(f"  Saved to    : {cfg.output_dir}")
        print(f"\n  Next step:")
        print(f"    python merge_adapter.py \\")
        print(f"      --base-model {cfg.model_name} \\")
        print(f"      --adapter    {cfg.output_dir} \\")
        print(f"      --output     ./checkpoints/stage1-merged")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Continual pretraining with LoRA")
    parser.add_argument("--smoke-test", action="store_true",
                        help="10 steps on 200 examples to validate pipeline")
    parser.add_argument("--model",   type=str, default=None, help="Override model name")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path")
    parser.add_argument("--output",  type=str, default=None, help="Override output dir")
    parser.add_argument("--epochs",  type=int, default=None, help="Override num epochs")
    args = parser.parse_args()

    cfg = PretrainConfig()
    if args.model:   cfg.model_name   = args.model
    if args.dataset: cfg.dataset_path = args.dataset
    if args.output:  cfg.output_dir   = args.output
    if args.epochs:  cfg.num_epochs   = args.epochs

    main(smoke_test=args.smoke_test)