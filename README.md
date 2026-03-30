# tele-SLMs

A research project investigating whether sub-2B parameter language models can acquire accurate knowledge of 3GPP and ETSI telecommunications standards through domain-adaptive pretraining and instruction fine-tuning.

The repository is organised in two phases, each in its own sub-directory:

```
tele-SLMs/
├── teleSLM_peft_pilot/    LoRA proof-of-concept on Qwen2.5-1.5B
└── teleSLMs-fft/      Full fine-tuning benchmark across a model size ladder
```

---

## Research Question

Can a small language model (sub-2B parameters) trained exclusively on telecommunications standards documents produce accurate, standards-grounded answers to technical questions — and how does performance scale with model size?

---

## Overview

### Phase 1 — LoRA Pilot (`teleSLM_peft_pilot/`)

Before committing to full fine-tuning, we ran a two-stage LoRA proof-of-concept on `Qwen2.5-1.5B` to validate the data pipeline and confirm that continual pretraining on telecom standards meaningfully shifts the model distribution.

**Stage 1 — Continual pretraining** on Tele-Data (Maatouk et al., 2024): 3GPP standards (upsampled 8×) and arXiv telecom papers, ~1.26B tokens/epoch, 2 epochs, 33,442 steps.

**Stage 2 — Instruction fine-tuning** on Alpaca (52k examples), LoRA SFT, ~34 minutes.

**Tele-Eval results** (2,000 examples, seed 42):

| Model | Token F1 | Perplexity |
|---|---|---|
| Qwen2.5-1.5B base | 31.20% | 8.90 |
| Qwen2.5-1.5B LoRA pretrain + SFT | 30.08% | 8.35 |

The 6% perplexity improvement confirmed that pretraining shifted the model toward the telecom distribution. The slight F1 regression was traced to a format mismatch between verbose Alpaca-style responses and the terse gold answers in Tele-Eval — not a loss of domain knowledge.

**What the pilot revealed:**

- LoRA is insufficient for fully absorbing a large standards corpus — the low-rank adapter lacks the capacity to update all weights with the domain signal
- The training data (mostly arXiv, only 2,801 standards documents) was dominated by general telecom research rather than actual standards text
- Tele-Eval scores are heavily influenced by source type — ~79% of questions come from arXiv, not standards alone.

See [`teleSLMs_peft_pilot/README.md`](teleSLM_peft_pilot/README.md) for full details.

---

### Phase 2 — Full Fine-Tuning (`teleSLMs-fft/`)

The full pipeline addresses every limitation identified in the pilot:

**Dataset — [TeleSpec-Data](https://huggingface.co/datasets/nareshmodina/TeleSpec-Data)**

A new standards-only corpus built specifically for this project:

| Subset | Source | Documents | Notes |
|---|---|---|---|
| `3gpp-standard` | TSpec-LLM (Rel-8 → Rel-19) | 15,054 | No arXiv |
| `etsi-standard` | NetSpec-LLM (15 working groups, 2000–2024) | 23,248 | No arXiv |
| **Total** | | **38,302** | **1.87B packed tokens** |

Unlike Tele-Data, TeleSpec-Data contains zero arXiv content — every token is from actual 3GPP or ETSI standards documents. This is a fundamental improvement: the pilot trained on ~13% standards, ~83% arXiv; the full pipeline trains on 100% standards.

**Training — Full fine-tuning (no LoRA)**

All model weights are updated during pretraining. LoRA is used only for the subsequent instruction fine-tuning step, preserving domain knowledge while adding chatbot-style behaviour.

**Model size ladder**

| Model | Parameters |
|---|---|
| `HuggingFaceTB/SmolLM2-135M` | 135M |
| `HuggingFaceTB/SmolLM2-360M` | 360M |
| `Qwen/Qwen2.5-1.5B` | 0.5B |
| `Qwen/Qwen2.5-1.5B` | 1.5B |

**Evaluation — standards-specific**

Tele-Eval results are reported separately for standards-derived questions (`standard_*` IDs) and the full benchmark. The standards-only F1 is the primary metric — it measures what the model actually learned from the training corpus.

---

## Key Design Decisions

| Decision | Pilot | Full pipeline | Rationale |
|---|---|---|---|
| Fine-tuning method | LoRA (pretrain + SFT) | Full FT (pretrain) + LoRA (SFT) | Full FT needed to absorb large corpus; LoRA preserves domain knowledge during SFT |
| Training data | Tele-Data (arXiv + standards) | TeleSpec-Data (standards only) | Eliminate arXiv contamination |
| Dataset size | ~1.26B tokens | 1.87B packed tokens | more tokens, most importantly all are from telecom standars |
| SFT dataset | Alpaca 52k | Alpaca 52k (→ UltraChat 200k) | Alpaca as baseline; UltraChat for ablation |
| Evaluation | Full Tele-Eval F1 | Standards-only + full F1 | Isolate standards comprehension from general telecom knowledge |
| Models | Qwen2.5-1.5B only | SmolLM2-135M → 360M → Qwen2.5-1.5B | Size ladder to study scaling behaviour |

---

## Progress

### ✅ Completed

**LoRA Pilot (`teleSLMs-pilot/`)**
- [x] Qwen2.5-1.5B — continual pretraining on Tele-Data (LoRA r=64, 33,442 steps, 1.26B tokens/epoch)
- [x] Qwen2.5-1.5B — instruction fine-tuning on Alpaca (LoRA SFT, 52k examples)
- [x] Qwen2.5-1.5B — Tele-Eval evaluation (Token F1: 30.08%, Perplexity: 8.35)
- [x] Pilot analysis — identified LoRA capacity limits and arXiv data contamination

**Full Fine-Tuning Pipeline (`teleSLMs-fft/`)**
- [x] TeleSpec-Data — 38,302 standards documents (3GPP + ETSI), published on HuggingFace
- [x] Token packing pipeline — `tokenize_dataset.py`, 914k packed blocks, 1.87B tokens
- [x] Full fine-tuning pipeline — `train.py`, `sft.py`, `eval.py`, `pipeline.py`, `config.py`
- [x] SmolLM2-135M tokenization — packed dataset ready at `./tele-tokenized`

---

### 🔄 In Progress

- **SmolLM2-135M — pretraining** (full fine-tuning on TeleSpec-Data, currently running)
- **SmolLM2-135M — SFT** (LoRA on Alpaca, pending pretrain completion)
- **SmolLM2-135M — evaluation** (Tele-Eval standards-only + full, pending SFT)
- **SmolLM2-135M — UltraChat ablation** (100k subset SFT, compare vs Alpaca)

---

### 📋 Pending

**SmolLM2-360M**
- [ ] Tokenize dataset (shared SmolLM2 tokenizer — no re-tokenization needed)
- [ ] Pretrain — full fine-tuning on TeleSpec-Data
- [ ] SFT — LoRA on Alpaca
- [ ] Evaluate — Tele-Eval standards-only + full

**Qwen/Qwen2.5-0.5B**
- [ ] Tokenize dataset (Qwen tokenizer — separate tokenization run required)
- [ ] Pretrain — full fine-tuning on TeleSpec-Data
- [ ] SFT — LoRA on Alpaca
- [ ] Evaluate — Tele-Eval standards-only + full

**Qwen/Qwen2.5-1.5B**
- [ ] Tokenize dataset (shared with Qwen2.5-0.5B tokenizer)
- [ ] Pretrain — full fine-tuning on TeleSpec-Data
- [ ] SFT — LoRA on Alpaca
- [ ] Evaluate — Tele-Eval standards-only + full

**Results & write-up**
- [ ] Populate benchmark results table across all models
- [ ] UltraChat 200k SFT ablation — compare standards F1 vs Alpaca across models
- [ ] Scaling analysis — standards F1 vs model size plot
- [ ] GitHub release with model checkpoints

---

## Dataset

**Pretraining:** [nareshmodina/TeleSpec-Data](https://huggingface.co/datasets/nareshmodina/TeleSpec-Data) — standards only, CC BY-NC 4.0

**Evaluation:** [AliMaatouk/Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval) — 750k telecom Q&A pairs, held out entirely from training

---

## Citation

If you use TeleSpec-Data, please cite:

```bibtex
@dataset{modina2025telespecdata,
  author    = {Naresh Modina},
  title     = {TeleSpec-Data: A Telecommunications Standards Dataset for Language Model Pretraining},
  year      = {2025},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/NareshModina/TeleSpec-Data}
}
```

Please also cite the upstream sources used for evaluation and comparison:

```bibtex
@misc{maatouk2024telellms,
  title         = {Tele-LLMs: A Series of Specialized Large Language Models for Telecommunications},
  author        = {Ali Maatouk and Kenny Chirino Ampudia and Rex Ying and Leandros Tassiulas},
  year          = {2024},
  eprint        = {2409.05314},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IT}
}

@misc{nikbakht2024tspecllm,
  title         = {TSpec-LLM: An Open-source Dataset for LLM Understanding of 3GPP Specifications},
  author        = {Rasoul Nikbakht and Mohamed Benzaghta and Giovanni Geraci},
  year          = {2024},
  eprint        = {2406.01768},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NI}
}
```

---

## License

Code: MIT
TeleSpec-Data: CC BY-NC 4.0
