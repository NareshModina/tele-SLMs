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
    # "HuggingFaceTB/SmolLM2-1.7B",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]

# Model to train in the current run
# Override with: python train.py --model HuggingFaceTB/SmolLM2-360M
ACTIVE_MODEL = MODELS[0]

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

TELESPEC_DATASET       = "nareshmodina/TeleSpec-Data"
TOKENIZED_DATASET_DIR  = "./tele-tokenized"   # output of tokenize_dataset.py
TELEEVAL_DATASET  = "AliMaatouk/Tele-Eval"

# Which subset of TeleSpec-Data to train on
# Options: None (all), "3gpp-standard", "etsi-standard"
TELESPEC_SUBSET   = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CHECKPOINTS_DIR   = "./checkpoints"
RESULTS_DIR       = "./results"
LOGS_DIR          = "./logs"

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

TRAIN = {
    "epochs":          2,
    "per_device_bs":   4,
    "max_length":      2048,
    "eval_ratio":      0.01,       # fraction of TeleSpec-Data held out for val loss
    "weight_decay":    0.01,
    "warmup_ratio":    0.03,
    "lr_scheduler":    "cosine",
    "logging_steps":   20,
    "eval_steps":      500,
    "save_steps":      500,
    "save_total_limit": 2,
    "dataloader_workers": 4,
    "gradient_checkpointing": False,  # set True for 1.7B+ if OOM
    "max_steps":            -1,        # -1 = disabled; set e.g. 50 for smoke tests
    "seed":            42,
}

# Per-model learning rate overrides
# Models not listed here use LR_DEFAULT
LR_DEFAULT = 1e-4

LR_OVERRIDES = {
    "SmolLM2-135M":  1e-4,
    "SmolLM2-360M":  5e-5,
    "SmolLM2-1.7B":  2e-5,
    "Qwen2.5-0.5B":  5e-5,
    "Qwen2.5-1.5B":  1e-5,
}

def get_lr(model_name: str) -> float:
    """Return the learning rate for a given model slug."""
    slug = model_name.split("/")[-1]
    return LR_OVERRIDES.get(slug, LR_DEFAULT)

# ---------------------------------------------------------------------------
# Evaluation hyperparameters
# ---------------------------------------------------------------------------

EVAL = {
    "n_examples":      2000,       # examples per eval run
    "max_new_tokens":  150,        # max tokens to generate per answer
    "ppl_subsample":   500,        # examples used for perplexity (speed vs accuracy)
    "seed":            42,

    # Source filters — controls which subset of Tele-Eval to evaluate on
    # "all"       → full Tele-Eval (standard + arxiv + wiki)
    # "standard"  → standards-only (3GPP/ETSI derived questions)
    # "arxiv"     → arXiv-derived questions
    # "wiki"      → Wikipedia-derived questions
    "source_filter": "all",
}

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

# Number of GPUs to use for training and SFT stages
# Set to 1 for single-GPU, 3 for 3x L40S, etc.
# Eval and plot stages always run on a single GPU regardless
NPROC = 3

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
    # Alternatives:
    #   "allenai/open-instruct-v1"
    #   "Muennighoff/flan"
    #   "HuggingFaceH4/ultrachat_200k"
    "dataset":          "tatsu-lab/alpaca",

    # LoRA config
    "lora_r":           16,
    "lora_alpha":       32,
    "lora_dropout":     0.05,
    "target_modules":   ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],

    # Training
    "epochs":           2,
    "per_device_bs":    4,
    "max_length":       1024,
    "lr":               5e-5,
    "lr_scheduler":     "cosine",
    "warmup_ratio":     0.03,
    "weight_decay":     0.01,
    "logging_steps":    10,
    "eval_steps":       200,
    "save_steps":       200,
    "save_total_limit": 2,
    "eval_ratio":       0.02,
    "max_steps":        -1,        # -1 = disabled; set e.g. 50 for smoke tests
}