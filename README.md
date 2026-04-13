# tele-SLMs

Fine-tuning small language models (SLMs) on telecommunications standards
for domain-specific knowledge acquisition and benchmarking.

---

## Overview

This project investigates whether sub-2B parameter language models can acquire
accurate knowledge of 3GPP and ETSI telecommunications standards through
full fine-tuning on a curated corpus of standards documents.

The pipeline covers:

1. **Domain-adaptive pretraining** — full fine-tuning on raw standards text
2. **Instruction fine-tuning** — LoRA SFT on Alpaca (for benchmarking) and smol-smoltalk (for release)
3. **Benchmark evaluation** — Ans-PPL and SemScore on Tele-Eval standards-only subset
4. **Model size ladder** — SmolLM2-135M → 360M → Qwen2.5-0.5B → 1.5B

---

## Background — LoRA Pilot

Before committing to the full pipeline, we ran a pilot study using
**Qwen2.5-1.5B** with LoRA (r=64) to test the overall approach and validate
the data pipeline. The pilot consisted of two phases:

- **Phase 1 — Continual pretraining** on TeleSpec-Data (~1.26B tokens,
  3× L40S, 2 epochs). Loss dropped from 1.76 → 1.48, token accuracy
  improved from 62% → 66.5%.

- **Phase 2 — Instruction fine-tuning (SFT)** on Stanford Alpaca (52k
  examples). Loss dropped from 1.59 → 1.33.

Tele-Eval benchmark results from the pilot (2,000 examples, seed 42):

| Model | Token F1 | Perplexity |
|---|---|---|
| Qwen2.5-1.5B base | 31.20% | 8.90 |
| Qwen2.5-1.5B LoRA (pretrain + SFT) | 30.08% | 8.35 |

The perplexity improvement (+6%) confirmed domain adaptation. The F1 regression
was traced to Alpaca-style SFT producing verbose responses that overlap poorly
with terse gold answers — not a loss of domain knowledge.

**What the pilot told us:** LoRA is sufficient for proof of concept but full
fine-tuning is needed to update all weights with the domain signal across the
full corpus. The current pipeline addresses this with full fine-tuning across
a model size ladder, with separate Alpaca SFT for benchmarking and
smol-smoltalk SFT for the HuggingFace release.

---

## Results

### SmolLM-TS-135M (4096 context, 10,000 standards examples)

| Model | Ans-PPL ↓ | SemScore ↑ | Notes |
|---|---|---|---|
| SmolLM2-135M-alpaca (base + Alpaca) | 13.42 | 0.6183 | Baseline |
| **SmolLM-TS-135M-alpaca (pretrain + Alpaca)** | **9.19** | **0.6504** | Ours |
| Improvement | **31.5%** ↓ | **+0.0321** ↑ | |

**Pretraining:** 2 epochs, 7,054 steps, eval loss 1.326 → 0.980

**HuggingFace releases:**
- 🤗 [nareshmodina/SmolLM-TS-135M](https://huggingface.co/nareshmodina/SmolLM-TS-135M) — pretrained base
- 🤗 [nareshmodina/SmolLM-TS-135M-it](https://huggingface.co/nareshmodina/SmolLM-TS-135M-it) — Alpaca instruction tuned

### Remaining models (in progress)

| Model | Ans-PPL ↓ | SemScore ↑ | Status |
|---|---|---|---|
| SmolLM2-360M-alpaca (baseline) | — | — | 🔄 training |
| SmolLM-TS-360M-alpaca (ours) | — | — | 🔄 training |
| SmolLM-TS-500M-alpaca (ours) | — | — | ⏳ pending |
| SmolLM-TS-1.5B-alpaca (ours) | — | — | ⏳ pending |

---

## Pipeline

```
Stage 0 — Tokenize    tokenize_dataset.py   pack TeleSpec-Data into 4096-token blocks (once per model family)
Stage 1 — Pretrain    train.py              full fine-tuning on packed dataset (+ plot curves)
Stage 2 — SFT-Alpaca  sft.py ×2             Alpaca SFT on pretrained + base model (for benchmark)
Stage 3 — Benchmark   benchmark.py ×2       Ans-PPL + SemScore on standards subset
```

> **V2 (planned):** smol-smoltalk SFT for better chat capabilities — pending evaluation of V1 results across all model sizes.

Run end-to-end:
```bash
python pipeline.py --model HuggingFaceTB/SmolLM2-135M
```

Smoke test (50 steps):
```bash
python pipeline.py --model HuggingFaceTB/SmolLM2-135M --max-steps 50
```

---

## Repository Structure

```
tele-SLMs/
├── config.py              # Central config — edit before running anything
├── pipeline.py            # End-to-end runner
├── tokenize_dataset.py    # Stage 0 — pre-tokenize and pack TeleSpec-Data
├── train.py               # Stage 1 — full fine-tuning
├── sft.py                 # Stage 2 — LoRA instruction fine-tuning
├── benchmark.py           # Stage 3 — Ans-PPL + SemScore evaluation
├── plot_training.py       # Plot loss / LR curves from trainer_state.json
├── upload_to_hf.py        # Upload models to HuggingFace Hub
├── results/               # Benchmark results and training plots
│   ├── benchmark_*.json
│   ├── benchmark_summary.json
│   └── plots/
│       ├── SmolLM-TS-135M-pretrain_curves.png
│       └── ...
├── checkpoints/           # Model weights (gitignored)
└── logs/                  # Training logs (gitignored)
```

---

## Dataset

**Training — [nareshmodina/TeleSpec-Data](https://huggingface.co/datasets/nareshmodina/TeleSpec-Data)**

| Subset | Source | Documents |
|---|---|---|
| `3gpp-standard` | TSpec-LLM (Rel-8 → Rel-19) | 15,054 |
| `etsi-standard` | NetSpec-LLM (15 working groups, 2000–2024) | 23,248 |
| **Total** | | **38,302** |

Packed at 4096 tokens → 457,160 blocks → 1.87B tokens

**Evaluation — [AliMaatouk/Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval)**

750,000 open-ended telecom Q&A pairs. Primary evaluation uses the
`standard_*` subset (83,056 examples, 10,000 sampled per run).

**SFT dataset (V1):** `tatsu-lab/alpaca` (52k examples, 1 epoch) — consistent with Tele-LLMs paper for fair comparison

---

## Installation

```bash
pip install torch transformers datasets trl peft accelerate sentence-transformers tqdm
```

Requires Python 3.10+ and CUDA-capable GPUs.

---

## Configuration

All settings in `config.py`. Key sections:

```python
MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-360M",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
]

MODEL_NAMES = {
    "HuggingFaceTB/SmolLM2-135M": "SmolLM-TS-135M",
    "HuggingFaceTB/SmolLM2-360M": "SmolLM-TS-360M",
    "Qwen/Qwen2.5-0.5B":          "SmolLM-TS-500M",
    "Qwen/Qwen2.5-1.5B":          "SmolLM-TS-1.5B",
}

TRAIN = {"epochs": 2, "max_length": 4096, ...}

SFT = {
    "dataset":      "HuggingFaceTB/smol-smoltalk",
    "subset_ratio": 0.05,
    "epochs":       1,
    "lr":           1e-5,
}

BENCHMARK = {"n_examples": 10000, "source_filter": "standard"}
```

---

## Checkpoint Naming

| Checkpoint | Description |
|---|---|
| `checkpoints/SmolLM-TS-135M/` | Pretrained base weights |
| `checkpoints/SmolLM-TS-135M-alpaca/` | Pretrain + Alpaca SFT — benchmark |
| `checkpoints/SmolLM2-135M-alpaca/` | Base + Alpaca SFT — baseline |
| `checkpoints/SmolLM-TS-135M-it/` | Pretrain + Alpaca SFT — HF release (V1) |

---

## Upload to HuggingFace

```bash
python upload_to_hf.py           # upload all available models
python upload_to_hf.py --dry-run # preview without uploading
python upload_to_hf.py --model SmolLM-TS-135M-alpaca
```

---

## Metrics

| Metric | Description |
|---|---|
| **Ans-PPL** | Answer perplexity conditioned on question (eq. 5, Maatouk et al. 2024) |
| **SemScore** | Cosine similarity via `all-mpnet-base-v2` (eq. 6, Maatouk et al. 2024) |

---

## HuggingFace Collection

🗂️ [nareshmodina/SmolLM-TS](https://huggingface.co/collections/nareshmodina/smollm-ts)

---

## Citation

```bibtex
@misc{modina2025smollmts,
  author    = {Naresh Modina},
  title     = {SmolLM-TS: Small Language Models for Telecommunications Standards},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/collections/nareshmodina/smollm-ts}
}

@dataset{modina2025telespecdata,
  author    = {Naresh Modina},
  title     = {TeleSpec-Data: A Telecommunications Standards Dataset for Language Model Pretraining},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/nareshmodina/TeleSpec-Data}
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