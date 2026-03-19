# tele-SLMs

*Exploring whether small language models can be specialized for 3GPP telecommunications standards — accurately, and with low retraining overhead.*

---

## Motivation

3GPP standards evolve with every major release cycle. Large general-purpose LLMs are expensive to retrain and require significant infrastructure. This project explores whether sub-2B parameter SLMs — small enough to retrain on modest hardware — can be made accurate on 3GPP standards content through domain-specific fine-tuning. The practical outcome is understanding whether this direction is viable at all.

---

## Research Questions

1. Can a sub-2B SLM produce accurate, standards-compliant responses to 3GPP questions?
2. What is the minimum viable model size for acceptable standards recall?
3. Does continual pretraining on standards data meaningfully improve over a general-purpose baseline of the same size?

---

## Project Phases

**Phase 1 — Continual Pretraining (current)**
LoRA fine-tuning of `Qwen/Qwen2.5-1.5B` on a curated telecom corpus (3GPP standards + arXiv papers). This is a trial run to validate the pipeline and establish a baseline before broader exploration.

**Phase 2 — Instruction Fine-Tuning (current)**
SFT on general instruction-following datasets (Alpaca, Open-Instruct) to make the Phase 1 model respond conversationally. Tele-Eval is intentionally excluded from training to preserve benchmark integrity.

**Phase 3 — SLM Benchmark Study (planned)**
Apply the same pipeline to a range of mini, micro, and small SLMs to find the minimum viable model size. Candidate models:

| Category | Models |
|---|---|
| Mini / Micro SLM | SmolLM2-135M, SmolLM2-360M |
| Small SLM | SmolLM2-1.7B, Qwen2.5-0.5B, Qwen2.5-1.5B |

Each candidate will be evaluated on [Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval) — used strictly as a held-out benchmark, never seen during training — using the three metrics from the Tele-LLMs paper: Answer Perplexity, SemScore, and LLM-Eval.

---

## Design Philosophy

- **Single purpose** — standards QA only, not a general assistant
- **Single modal** — text only
- **Updatable** — low infrastructure barrier to retrain when new releases drop
- **Accurate over broad** — precision on standards content is the only metric that matters

---

## Dataset

Training data comes from [Tele-Data (Maatouk et al., 2024)](https://huggingface.co/datasets/AliMaatouk/Tele-Data). Two of the four sources are used:

| Source | Documents | Approx. Tokens | Used |
|---|---|---|---|
| 3GPP Standards | 2,801 | 86M | ✅ |
| arXiv papers | 90,310 | 1.08B | ✅ |
| Wikipedia | 19,543 | 26M | ❌ |
| Web | 740,024 | 1.55B | ❌ |

Standards documents are split on clause boundaries (`X.X.X\tTitle`) and upsampled ×8 to counterbalance the larger arXiv corpus. After pre-tokenization and sequence packing into 2048-token blocks, the training set contains approximately **1.26B tokens per epoch**.

---

## Pipeline

```
audit_dataset.py      Audit Tele-Data structure, lengths, metadata
preprocess.py         Clause-aware splitting, upsampling, formatting
tokenize_dataset.py   Pre-tokenize and pack into fixed 2048-token blocks
pretrain.py           Stage 1 continual pretraining (LoRA or full fine-tuning)
merge_adapter.py      Merge LoRA adapter into base model weights
sft.py                Stage 2 instruction fine-tuning — coming soon
```

---

## Hardware

Current experiments run on **3× NVIDIA L40S (45GB VRAM each)**. A design goal of Phase 3 is that candidate models should be trainable on a **single consumer GPU**, validating the low-infrastructure premise of the project.

---

## Current Status

Phase 1 smoke test (10 steps, Qwen2.5-1.5B + LoRA):

| Metric | Value |
|---|---|
| Train loss | 1.73 |
| Eval loss | 1.88 |
| Token accuracy | 62.5% |

Full Phase 1 and Phase 2 results will be added here as experiments complete.

---

## Roadmap

- [x] Dataset auditing pipeline
- [x] Clause-aware preprocessing for 3GPP standards
- [x] Pre-tokenization and sequence packing
- [x] Phase 1 trial run — Qwen2.5-1.5B LoRA pretraining
- [x] Multi-GPU training with PyTorch DDP (3× L40S)
- [ ] Phase 2 — Instruction fine-tuning on Tele-Eval
- [ ] Phase 3 — SLM benchmark study (SmolLM2, Qwen2.5 sub-2B)
- [ ] Evaluation on full Tele-Eval benchmark
- [ ] ETSI documentation corpus integration
- [ ] Explore beyond Q&A — specialist assistant for standards referencing, clause lookup, and cross-release change tracking
- [ ] Model release on HuggingFace Hub

---

## Installation

```bash
git clone https://github.com/NareshModina/tele-SLMs
cd tele-SLMs
pip install torch transformers datasets peft trl accelerate
```

---

## Citation

This project builds on Tele-LLMs and uses Tele-Data and Tele-Eval:

```bibtex
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

MIT
