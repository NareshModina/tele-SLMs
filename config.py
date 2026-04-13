"""
tele-SLMs Configuration
========================
Central config file for all training and evaluation settings.
Edit this file before running train.py or eval.py.

Usage:
    # train.py and eval.py import this automatically — no flags needed
    python train.py
    python eval.py --checkpoint ./checkpoints/SmolLM2-135M-telespec
"""

# ---------------------------------------------------------------------------
# Model ladder
# Evaluated in order — start small, scale up
# ---------------------------------------------------------------------------

MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]

# Model to train in the current run
# Override with: python train.py --model HuggingFaceTB/SmolLM2-360M
ACTIVE_MODEL = MODELS[0]

# ---------------------------------------------------------------------------
# SmolLM-TS naming — maps HF model ID to final release name
# Used for checkpoints, logs, results and HuggingFace upload
# ---------------------------------------------------------------------------

MODEL_NAMES = {
    "HuggingFaceTB/SmolLM2-135M": "SmolLM-TS-135M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM-TS-360M",
    "Qwen/Qwen2.5-0.5B":          "SmolLM-TS-500M",
    "Qwen/Qwen2.5-1.5B":          "SmolLM-TS-1.5B",
}

def get_model_name(model_id: str) -> str:
    """Return the SmolLM-TS release name for a given HF model ID."""
    return MODEL_NAMES.get(model_id, model_id.split("/")[-1])

# ---------------------------------------------------------------------------
# Instruct baselines — evaluated once via benchmark.py, no training needed
# Each entry: (hf_model_id, benchmark_label)
# These are the official instruct models corresponding to each base model
# ---------------------------------------------------------------------------

INSTRUCT_BASELINES = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M-Instruct"),
    ("Qwen/Qwen2.5-0.5B-Instruct",          "Qwen2.5-500M-Instruct"),
    ("Qwen/Qwen2.5-1.5B-Instruct",          "Qwen2.5-1.5B-Instruct"),
]

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

TELESPEC_DATASET       = "nareshmodina/TeleSpec-Data"
# Tokenized dataset dirs — separate per tokenizer family and context length
# SmolLM2 family (135M, 360M) share a tokenizer — one dir
# Qwen2.5 family (0.5B, 1.5B) share a tokenizer — separate dir
TOKENIZED_DATASET_DIR       = "./tele-tokenized-4096"        # SmolLM2 family
TOKENIZED_DATASET_DIR_QWEN  = "./tele-tokenized-4096-qwen"   # Qwen2.5 family
TELEEVAL_DATASET  = "AliMaatouk/Tele-Eval"

# Which subset of TeleSpec-Data to train on
# Options: None (all), "3gpp-standard", "etsi-standard"
TELESPEC_SUBSET   = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Base directories — pipeline creates model-specific subdirs automatically
# e.g. checkpoints/SmolLM-TS-135M/, logs/SmolLM-TS-135M/, results/SmolLM-TS-135M/
CHECKPOINTS_DIR   = "./checkpoints"
RESULTS_DIR       = "./results"
LOGS_DIR          = "./logs"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

TRAIN = {
    "epochs":          2,
    "per_device_bs":   4,
    "max_length":      4096,      # 4096 for all models — consistent context length
    "eval_ratio":      0.01,       # fraction of TeleSpec-Data held out for val loss
    "weight_decay":    0.01,
    "warmup_ratio":    0.03,
    "lr_scheduler":    "cosine",
    "logging_steps":   20,
    "eval_steps":      500,
    "save_steps":      500,
    "save_total_limit": 2,
    "dataloader_workers": 4,
    "gradient_checkpointing": False,  # set True for Qwen2.5-1.5B at 4096 context
    "max_steps":            -1,        # -1 = disabled; set e.g. 50 for smoke tests
    "seed":            42,
}

# Per-model learning rate overrides
# Models not listed here use LR_DEFAULT
LR_DEFAULT = 1e-4

LR_OVERRIDES = {
    "SmolLM2-135M":  1e-4,
    "SmolLM2-360M":  5e-5,
    "Qwen2.5-0.5B":  5e-5,
    "Qwen2.5-1.5B":  1e-5,
}

def get_lr(model_name: str) -> float:
    """Return the learning rate for a given model slug."""
    slug = model_name.split("/")[-1]
    return LR_OVERRIDES.get(slug, LR_DEFAULT)

# Per-model batch size overrides at 4096 context length
# Qwen2.5-1.5B needs smaller batch + gradient checkpointing to fit on 3×L40S
PER_DEVICE_BS_OVERRIDES = {
    "SmolLM2-135M":  4,
    "SmolLM2-360M":  4,
    "Qwen2.5-0.5B":  4,
    "Qwen2.5-1.5B":  2,   # tight at ~44GB — grad checkpointing required
}

GRAD_CKPT_OVERRIDES = {
    "Qwen2.5-1.5B":  True,
}

def get_per_device_bs(model_name: str) -> int:
    """Return per-device batch size for a given model slug."""
    slug = model_name.split("/")[-1]
    return PER_DEVICE_BS_OVERRIDES.get(slug, TRAIN["per_device_bs"])

def get_grad_ckpt(model_name: str) -> bool:
    """Return whether gradient checkpointing should be enabled."""
    slug = model_name.split("/")[-1]
    return GRAD_CKPT_OVERRIDES.get(slug, TRAIN["gradient_checkpointing"])

def get_tokenized_dir(model_name: str) -> str:
    """Return the correct tokenized dataset dir for a given model family."""
    if "Qwen" in model_name:
        return TOKENIZED_DATASET_DIR_QWEN
    return TOKENIZED_DATASET_DIR

# ---------------------------------------------------------------------------
# Benchmark hyperparameters (benchmark.py)
# Generic eval (eval.py) is retired — use benchmark.py for all evaluations
# ---------------------------------------------------------------------------

BENCHMARK = {
    "n_examples":      10000,      # examples per benchmark run (use --n to override)
    "max_new_tokens":  100,        # max tokens to generate per answer
    "seed":            42,

    # Primary filter — standards-only questions (3GPP/ETSI derived)
    # Options: "standard", "arxiv", "wiki", "all"
    "source_filter":   "standard",
}

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Number of GPUs to use for training and SFT stages
# Set to 1 for single-GPU, 3 for 3x L40S, etc.
# Eval and plot stages always run on a single GPU regardless
NPROC = 2

# Target effective batch size (grad_accum is computed automatically)
TARGET_EFFECTIVE_BS = 128

# DDP port — change if you hit "address already in use"
# pipeline.py auto-increments this between pretrain and SFT to avoid conflicts
DDP_MASTER_PORT = 29502

# ---------------------------------------------------------------------------
# Logging / reporting
# ---------------------------------------------------------------------------

# "none", "wandb", "tensorboard"
REPORT_TO = "none"

# Set your wandb project name if REPORT_TO = "wandb"
WANDB_PROJECT = "tele-slms"

# ---------------------------------------------------------------------------
# SFT (Instruction Fine-Tuning) — LoRA on instruction dataset
# ---------------------------------------------------------------------------

SFT = {
    # Instruction dataset — HuggingFace dataset ID
    # Current: Alpaca for all models (V1 release)
    # V2 will use smol-smoltalk for better chat capabilities
    # Options:
    #   "tatsu-lab/alpaca"               — 52k single-turn (current, comparable to Tele-LLMs paper)
    #   "HuggingFaceTB/smol-smoltalk"    — 460k multi-turn, sub-1B SmolLM2 models
    #   "HuggingFaceTB/smoltalk"         — 1M multi-turn, Qwen2.5-0.5B and 1.5B
    "dataset":          "tatsu-lab/alpaca",

    # Fraction of dataset to use (1.0 = full, 0.05 = 5%)
    # Goal: teach ChatML format only — LoRA protects domain weights from being overwritten
    # smol-smoltalk: 5% = ~23k examples (sufficient for format acquisition)
    # smoltalk:      5% = ~50k examples
    "subset_ratio":     0.05,

    # LoRA config — conservative rank to minimise drift from domain-adapted weights
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,
    "target_modules":   ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],

    # Training — 1 epoch only: ChatML format is learnable in a single pass
    # 2 epochs risks drifting the LoRA adapter too far from domain distribution
    "epochs":           1,
    "per_device_bs":    4,
    "max_length":       4096,       # match pretraining context length
    "lr":               1e-5,       # conservative — SmolLM2 instruct used 3e-4 but from scratch
    "lr_scheduler":     "cosine",
    "warmup_ratio":     0.03,
    "weight_decay":     0.01,
    "logging_steps":    10,
    "eval_steps":       200,
    "save_steps":       200,
    "save_total_limit": 2,
    "eval_ratio":       0.02,
    "max_steps":        -1,         # -1 = disabled; set e.g. 50 for smoke tests
}