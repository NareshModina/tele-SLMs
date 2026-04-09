# tele-SLMs

Fine-tuning small language models (SLMs) on telecommunications standards
for domain-specific knowledge acquisition and benchmarking.

---

## Overview

This project investigates whether sub-2B parameter language models can acquire
accurate knowledge of 3GPP and ETSI telecommunications standards through
full fine-tuning on a curated corpus of standards documents.

The pipeline covers:

1. **Continual pretraining** on raw standards text (domain adaptation)
2. **Benchmark evaluation** on Tele-Eval (open-ended telecom Q&A)
3. **Model size ladder** — SmolLM2-135M → 360M → 1.7B — to study the
   relationship between model size and domain knowledge retention

---

## Background — LoRA Pilot

Before committing to the full pipeline, we ran a pilot study using
**Qwen2.5-1.5B** with LoRA (r=64) to test the overall approach and validate
the data pipeline. The pilot consisted of two phases:

- **Phase 1 — Continual pretraining** on TeleSpec-Data (~1.26B tokens,
  3× L40S, 2 epochs). Loss dropped from 1.76 → 1.48, token accuracy
  improved from 62% → 66.5%. Qualitative inspection confirmed strong domain
  adaptation — the model produced correct 3GPP terminology and
  standards-style procedural language.

- **Phase 2 — Instruction fine-tuning (SFT)** on Stanford Alpaca (52k
  examples), using `completion_only_loss=True` in TRL 0.29. Loss dropped
  from 1.59 → 1.33. The model successfully learned to follow instructions
  and answer questions in a structured format.

Tele-Eval benchmark results from the pilot (2,000 examples, seed 42):

| Model | Token F1 | Perplexity |
|---|---|---|
| Qwen2.5-1.5B base | 31.20% | 8.90 |
| Qwen2.5-1.5B LoRA (pretrain + SFT) | 30.08% | 8.35 |

The perplexity improvement (+6%) confirmed that Phase 1 pretraining
successfully shifted the model distribution toward telecom standards. The
slight F1 regression was traced to a format mismatch — Alpaca-style SFT
produces verbose responses that overlap poorly with the terse gold answers
in Tele-Eval — and not to a loss of domain knowledge.

**What the pilot told us:** LoRA is sufficient for a quick proof of concept
but has two fundamental limitations for this task. First, the 38k-document
TeleSpec-Data corpus is substantially larger than what LoRA can fully absorb
with a low-rank adapter — full fine-tuning is needed to update all weights
with the domain signal. Second, Alpaca alone is too small and too generic
for instruction fine-tuning on a technical domain. The current pipeline
addresses both by moving to full fine-tuning across a model size ladder,
with no separate SFT phase — domain knowledge and task format are learned
together from the standards corpus directly.

---

## Repository Structure

```
tele-SLMs/
├── config.py           # ← Edit this before running anything
├── pipeline.py         # ← End-to-end runner (pretrain → SFT → eval)
├── tokenize_dataset.py # Stage 0 — pre-tokenize and pack TeleSpec-Data (run once)
├── train.py            # Stage 1 — full fine-tuning on packed dataset
├── sft.py              # Stage 2 — LoRA instruction fine-tuning on Alpaca/FLAN
├── eval.py             # Tele-Eval benchmark evaluation
├── plot_training.py    # Plot loss / accuracy / LR curves from trainer_state.json
├── checkpoints/        # Saved model weights (gitignored)
│   ├── {model-slug}-telespec/        # Stage 1 output
│   ├── {model-slug}-telespec-sft/    # Stage 2 LoRA adapter
│   └── {model-slug}-telespec-sft-merged/  # Stage 2 merged (used by eval.py)
├── results/            # Evaluation outputs
│   ├── {model-slug}_eval.json
│   ├── summary.json
│   └── plots/
│       ├── {model-slug}-pretrain_curves.png
│       ├── {model-slug}-sft-A_curves.png
│       └── {model-slug}-sft-B_curves.png
└── logs/               # Training logs (gitignored)
```

---

## Dataset

**Training — [nareshmodina/TeleSpec-Data](https://huggingface.co/datasets/nareshmodina/TeleSpec-Data)**

| Subset | Source | Documents |
|---|---|---|
| `3gpp-standard` | TSpec-LLM (Rel-8 → Rel-19) | 15,054 |
| `etsi-standard` | NetSpec-LLM (15 working groups, 2000–2024) | 23,248 |
| **Total** | | **38,302** |

Schema: `id`, `category`, `content`, `metadata`

**Evaluation — [AliMaatouk/Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval)**

750,000 open-ended telecom Q&A pairs generated from standards, arxiv papers,
and Wikipedia using Mixtral-8x7B. Fields: `Statement`, `Answer`, `id`.

---

## Installation

```bash
pip install torch transformers datasets trl peft accelerate tqdm
```

Requires Python 3.10+ and CUDA-capable GPUs for multi-GPU training.

---

## Configuration

**All settings live in `config.py` — edit it before running any script.**

The key sections:

```python
# Which model to train/eval in the current run
ACTIVE_MODEL = "HuggingFaceTB/SmolLM2-135M"

# Full model ladder for the benchmark sweep
MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-1.7B",
]

# Dataset subset: None (all), "3gpp-standard", or "etsi-standard"
TELESPEC_SUBSET = None

# Output directory for pre-tokenized packed dataset (produced by tokenize_dataset.py)
TOKENIZED_DATASET_DIR = "./tele-tokenized"

# Training hyperparameters
TRAIN = {
    "epochs":          2,
    "per_device_bs":   4,
    "max_length":      2048,
    ...
}

# Per-model learning rates (auto-applied, no CLI flag needed)
LR_OVERRIDES = {
    "SmolLM2-135M":  1e-4,
    "SmolLM2-360M":  5e-5,
    "SmolLM2-1.7B":  2e-5,
}

# Evaluation settings
EVAL = {
    "n_examples":     2000,
    "max_new_tokens": 150,
    "ppl_subsample":  500,
}

# Logging: "none", "wandb", "tensorboard"
REPORT_TO = "none"

# SFT (instruction fine-tuning)
SFT = {
    "dataset":    "tatsu-lab/alpaca",   # or "allenai/open-instruct-v1" / "Muennighoff/flan"
    "lora_r":     16,
    "lora_alpha": 32,
    "epochs":     2,
    "lr":         5e-5,
    "max_length": 1024,
    "max_steps":  -1,    # -1 = full training; set e.g. 50 for smoke tests
    ...
}
```

CLI flags always override config values when provided.

---

## Usage

### Tokenization (run once before training)

Pre-tokenizes and packs TeleSpec-Data into fixed 2048-token blocks.
Run this once per model family — the tokenizer differs between SmolLM2 and Qwen.

```bash
# Tokenize for SmolLM2 family (covers 135M, 360M, 1.7B — same tokenizer)
python tokenize_dataset.py --model HuggingFaceTB/SmolLM2-135M

# Tokenize for Qwen family
python tokenize_dataset.py --model Qwen/Qwen2.5-1.5B

# Custom output dir or subset
python tokenize_dataset.py --model HuggingFaceTB/SmolLM2-135M --subset 3gpp-standard
```

**Why packing matters:** TeleSpec-Data has median ~500 tokens per document but
`max_length=2048`. Without packing, ~75% of each forward pass is wasted on padding.
Packing concatenates all tokens into one stream and cuts into exact 2048-token blocks —
zero waste, every token trained on, ~3-4x faster training.

Output saved to `./tele-tokenized/` (configurable via `TOKENIZED_DATASET_DIR` in config.py).

### Training

```bash
# Requires tokenize_dataset.py to have been run first
# Use defaults from config.py
python train.py

# Multi-GPU (3× L40S)
python -m torch.distributed.run --nproc_per_node=3 --master_port=29502 \
    train.py

# Override model for a single run (without editing config)
python train.py --model HuggingFaceTB/SmolLM2-360M

# Enable gradient checkpointing for large models
python train.py --model HuggingFaceTB/SmolLM2-1.7B --gradient-checkpointing

# Train on 3GPP subset only
python train.py --subset 3gpp-standard
```

**CLI arguments** (all default to values in `config.py`):

| Argument | Config key | Description |
|---|---|---|
| `--model` | `ACTIVE_MODEL` | HuggingFace model ID |
| `--subset` | `TELESPEC_SUBSET` | `None`, `3gpp-standard`, `etsi-standard` |
| `--epochs` | `TRAIN["epochs"]` | Training epochs |
| `--lr` | `LR_OVERRIDES` | Learning rate (auto-resolved from config if omitted) |
| `--per-device-bs` | `TRAIN["per_device_bs"]` | Per-GPU batch size |
| `--max-length` | `TRAIN["max_length"]` | Max token sequence length |
| `--eval-ratio` | `TRAIN["eval_ratio"]` | Fraction of data held out for val loss |
| `--gradient-checkpointing` | `TRAIN["gradient_checkpointing"]` | Reduce GPU memory usage |

Checkpoint saved to `./checkpoints/{model-slug}-telespec/`.


### Instruction Fine-Tuning (SFT)

LoRA SFT on top of a Stage 1 checkpoint or base model:

```bash
# Condition A — pretrain → SFT (full pipeline)
python sft.py --input ./checkpoints/SmolLM2-135M-telespec

# Condition B — base → SFT (no domain pretraining)
python sft.py --input HuggingFaceTB/SmolLM2-135M --base

# Multi-GPU
python -m torch.distributed.run --nproc_per_node=3 --master_port=29503 \
    sft.py --input ./checkpoints/SmolLM2-135M-telespec

# Use FLAN instead of Alpaca
python sft.py --input ./checkpoints/SmolLM2-135M-telespec --dataset Muennighoff/flan

# Smoke test
python sft.py --input ./checkpoints/SmolLM2-135M-telespec --max-steps 50
```

**CLI arguments** (all default to values in `config.py`):

| Argument | Config key | Description |
|---|---|---|
| `--input` | — | Stage 1 checkpoint path or base model HF ID |
| `--base` | — | Flag that --input is a base model (affects output naming) |
| `--dataset` | `SFT["dataset"]` | Instruction dataset HF ID |
| `--epochs` | `SFT["epochs"]` | Training epochs |
| `--lr` | `SFT["lr"]` | Learning rate |
| `--per-device-bs` | `SFT["per_device_bs"]` | Per-GPU batch size |
| `--max-length` | `SFT["max_length"]` | Max token sequence length |
| `--max-steps` | `SFT["max_steps"]` | Hard stop for smoke tests (-1 = disabled) |

Saves LoRA adapter to `./checkpoints/{slug}-sft/` and merged weights to
`./checkpoints/{slug}-sft-merged/`. Pass the merged path to `eval.py`.

### Evaluation

```bash
# Evaluate Condition A (pretrain + SFT)
python eval.py --checkpoint ./checkpoints/SmolLM2-135M-telespec-sft-merged

# Evaluate with explicit model name and label
python eval.py \
    --model      HuggingFaceTB/SmolLM2-135M \
    --checkpoint ./checkpoints/SmolLM2-135M-telespec-sft-merged \
    --label      SmolLM2-135M-pretrain-sft

# Evaluate base model as baseline (no checkpoint)
python eval.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --label SmolLM2-135M-base

# Print leaderboard from all saved results
python eval.py --summary-only
```

**CLI arguments** (all default to values in `config.py`):

| Argument | Config key | Description |
|---|---|---|
| `--model` | `ACTIVE_MODEL` | Base model ID |
| `--checkpoint` | — | Path to fine-tuned checkpoint (omit for base model eval) |
| `--label` | — | Display name in leaderboard (auto-generated if omitted) |
| `--n` | `EVAL["n_examples"]` | Number of Tele-Eval examples |
| `--seed` | `EVAL["seed"]` | Random seed for dataset shuffle |
| `--summary-only` | — | Print leaderboard and exit |

Results saved to `./results/{label}_eval.json` and aggregated in
`./results/summary.json`.

### Plot Training Curves

After training completes, `trainer_state.json` is saved alongside the
checkpoint. Use `plot_training.py` to visualise loss, token accuracy,
and learning rate schedule:

```bash
python plot_training.py \
    --json   ./checkpoints/SmolLM2-135M-telespec/trainer_state.json \
    --output ./results/SmolLM2-135M_curves.png
```

---

## Pipeline (End-to-End)

`pipeline.py` runs all stages in strict sequence — tokenize, pretrain, plot, SFT, plot,
eval — with resume support and a full summary at the end.

```bash
# No flags → interactive prompts for model selection and run mode
# (1) Select model from config.MODELS, or run all
# (2) Full run or smoke test (50 steps / 50 eval examples)
python pipeline.py

# Run a specific model directly
python pipeline.py --model HuggingFaceTB/SmolLM2-135M

# Run full model ladder sequentially
python pipeline.py --all-models

# Skip pretrain (checkpoint already exists)
python pipeline.py --skip-pretrain

# Skip pretrain and SFT, run eval only
python pipeline.py --skip-pretrain --skip-sft

# Force re-run all stages
python pipeline.py --force

# Smoke test — 50 steps, 50 eval examples
python pipeline.py --max-steps 50 --eval-n 50
```

**CLI arguments:**

| Argument | Config key | Description |
|---|---|---|
| `--model` | — | Model to run (omit to get interactive selection prompt) |
| `--all-models` | `MODELS` | Run full ladder from config sequentially |
| `--nproc` | `NPROC` | GPUs for training stages |
| `--force` | — | Re-run all stages even if checkpoints exist |
| `--skip-pretrain` | — | Skip Stage 1 (checkpoint must exist) |
| `--skip-sft` | — | Skip SFT stages, go straight to eval |
| `--max-steps` | — | Hard stop per training stage (smoke test) |
| `--eval-n` | `EVAL["n_examples"]` | Tele-Eval examples per eval run |

**Resume behaviour:** if a checkpoint already exists for a stage it is
skipped automatically. Use `--force` to override. If any stage fails,
the pipeline stops immediately and prints the path to the relevant log file.

**Outputs per run:**
- `logs/{slug}-pretrain.log`, `logs/{slug}-sft-A.log`, etc.
- `results/plots/{slug}-pretrain_curves.png`, `-sft-A_curves.png`, `-sft-B_curves.png`
- `results/{slug}-pretrain-sft_eval.json`, `results/{slug}-sft-only_eval.json`
- `results/summary.json` — cumulative leaderboard
- `logs/pipeline_{slug}_{timestamp}.log` — stage-by-stage timing summary

---

## Smoke Test

Run this first to validate the full pipeline end-to-end before committing
to a full training run. Uses 50 steps for training and 50 examples for eval
— completes in a few minutes.

```bash
mkdir -p logs results

# Run the full pipeline with 50 steps per training stage and 50 eval examples
python pipeline.py \
    --model     HuggingFaceTB/SmolLM2-135M \
    --max-steps 50 \
    --eval-n    50
```

Or run stages individually if you prefer:

```bash
mkdir -p logs results

# Step 1 — Pretrain (50 steps)
python train.py --model HuggingFaceTB/SmolLM2-135M --max-steps 50 --max-length 512 2>&1 | tee logs/smoke_train.log

# Step 2 — Plot pretrain curves
python plot_training.py --json ./checkpoints/SmolLM2-135M-telespec/trainer_state.json --output ./results/plots/smoke-pretrain_curves.png --title "Smoke — Pretrain"

# Step 3 — SFT on pretrained checkpoint (Condition A)
python sft.py --input ./checkpoints/SmolLM2-135M-telespec --max-steps 50 2>&1 | tee logs/smoke_sft_A.log

# Step 4 — SFT on base model (Condition B)
python sft.py --input HuggingFaceTB/SmolLM2-135M --base --max-steps 50 2>&1 | tee logs/smoke_sft_B.log

# Step 5 — Plot SFT curves
python plot_training.py --json ./checkpoints/SmolLM2-135M-telespec-sft/trainer_state.json --output ./results/plots/smoke-sft-A_curves.png --title "Smoke — SFT-A"
python plot_training.py --json ./checkpoints/SmolLM2-135M-sft/trainer_state.json --output ./results/plots/smoke-sft-B_curves.png --title "Smoke — SFT-B"

# Step 6 — Eval Condition A
python eval.py --model HuggingFaceTB/SmolLM2-135M --checkpoint ./checkpoints/SmolLM2-135M-telespec-sft-merged --label smoke-pretrain-sft --n 50 2>&1 | tee logs/smoke_eval_A.log

# Step 7 — Eval Condition B
python eval.py --model HuggingFaceTB/SmolLM2-135M --checkpoint ./checkpoints/SmolLM2-135M-sft-merged --label smoke-sft-only --n 50 2>&1 | tee logs/smoke_eval_B.log

# Step 8 — Print leaderboard
python eval.py --summary-only
```

What to check after each step:

| Step | What to verify |
|---|---|
| After train.py | Loss decreasing, checkpoint saved to `./checkpoints/SmolLM2-135M-telespec/` |
| After sft.py (A) | Loss decreasing, merged model saved to `./checkpoints/SmolLM2-135M-telespec-sft-merged/` |
| After eval.py (A) | `./results/smoke-pretrain-sft_eval.json` exists, F1 and PPL values present |
| After sft.py (B) | Merged model saved to `./checkpoints/SmolLM2-135M-sft-merged/` |
| After eval.py (B) | `./results/smoke-sft-only_eval.json` exists |
| After summary | Two rows in leaderboard table |

---

## Benchmark Sweep

To run the full model size ladder, edit `config.py` to set each model as
`ACTIVE_MODEL` in turn, or use CLI overrides in a loop:

```bash
mkdir -p logs results

for MODEL in \
    HuggingFaceTB/SmolLM2-135M \
    HuggingFaceTB/SmolLM2-360M \
    HuggingFaceTB/SmolLM2-1.7B; do

    SLUG=$(echo $MODEL | cut -d'/' -f2)

    # Train
    python -m torch.distributed.run --nproc_per_node=3 --master_port=29502 \
        train.py --model $MODEL \
        2>&1 | tee logs/${SLUG}-train.log

    # Stage 2 — SFT on top of pretrained model (Condition A)
    python sft.py --input ./checkpoints/${SLUG}-telespec

    # Stage 2 — SFT on top of base model (Condition B)
    python sft.py --input $MODEL --base

    # Eval Condition A (pretrain + SFT)
    python eval.py \
        --model      $MODEL \
        --checkpoint ./checkpoints/${SLUG}-telespec-sft-merged \
        --label      ${SLUG}-pretrain-sft

    # Eval Condition B (SFT only)
    python eval.py \
        --model      $MODEL \
        --checkpoint ./checkpoints/${SLUG}-sft-merged \
        --label      ${SLUG}-sft-only

    # Eval base model (no training)
    python eval.py \
        --model  $MODEL \
        --label  ${SLUG}-base

done

# Print full leaderboard
python eval.py --summary-only
```

---

## Metrics

| Metric | Description |
|---|---|
| **Token F1** | Word-overlap between prediction and gold answer — primary metric |
| **Exact Match** | Normalised string equality — expected to be low (~0–2%) for open QA |
| **Perplexity** | Cross-entropy on gold answer tokens — lower = better |

Token F1 follows the SQuAD evaluation protocol (lowercase, strip punctuation
and articles, compute word overlap). Perplexity is computed on the first
`EVAL["ppl_subsample"]` examples for speed.

---

## Results

Full fine-tuning results across the model size ladder will be populated here
as experiments complete.

| Model | Token F1 | Perplexity |
|---|---|---|
| SmolLM2-135M base | — | — |
| SmolLM2-135M SFT only | — | — |
| SmolLM2-135M pretrain + SFT | — | — |
| SmolLM2-360M base | — | — |
| SmolLM2-360M SFT only | — | — |
| SmolLM2-360M pretrain + SFT | — | — |
| SmolLM2-1.7B base | — | — |
| SmolLM2-1.7B SFT only | — | — |
| SmolLM2-1.7B pretrain + SFT | — | — |

See the Background section above for pilot results (LoRA, Qwen2.5-1.5B).

---

## Citation

```bibtex
@dataset{modina2025telespecdata,
  author    = {Naresh Modina},
  title     = {TeleSpec-Data: A Telecommunications Standards Dataset for Language Model Pretraining},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/NareshModina/TeleSpec-Data}
}

@misc{nikbakht2024tspecllm,
  title         = {TSpec-LLM: An Open-source Dataset for LLM Understanding of 3GPP Specifications},
  author        = {Rasoul Nikbakht and Mohamed Benzaghta and Giovanni Geraci},
  year          = {2024},
  eprint        = {2406.01768},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NI}
}

@misc{maatouk2024telellms,
  title         = {Tele-LLMs: A Series of Specialized Large Language Models for Telecommunications},
  author        = {Ali Maatouk and Kenny Chirino Ampudia and Rex Ying and Leandros Tassiulas},
  year          = {2024},
  eprint        = {2409.05314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}
```

---

## License

Code: MIT  
TeleSpec-Data: CC BY-NC 4.0