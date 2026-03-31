"""
tele-SLMs — Pre-Tokenize and Pack Dataset
==========================================
Run ONCE before training. Loads nareshmodina/TeleSpec-Data from HuggingFace,
tokenises all documents, then packs token sequences into fixed-length blocks
using the standard group_texts approach.

Why packing matters:
    TeleSpec-Data has median ~500 tokens/document but max_length=2048.
    Without packing, ~75% of each forward pass is wasted on padding tokens.
    Packing concatenates all tokens into one stream and cuts into exact
    max_length blocks — zero waste, every token trained on.

Output is saved to disk and loaded directly by train.py at training time,
so tokenisation overhead is paid once, not every run.

Usage:
    # Tokenize for a specific model (tokenizer differs per model family)
    python tokenize_dataset.py --model HuggingFaceTB/SmolLM2-135M

    # Different output dir or max length
    python tokenize_dataset.py \
        --model      HuggingFaceTB/SmolLM2-135M \
        --output     ./tele-tokenized-smollm2 \
        --max-length 2048 \
        --num-proc   8

    # Subset only
    python tokenize_dataset.py \
        --model  HuggingFaceTB/SmolLM2-135M \
        --subset 3gpp-standard

Output:
    {output}/train/    HuggingFace Arrow dataset — packed 2048-token blocks
    {output}/eval/     HuggingFace Arrow dataset — packed 2048-token blocks
    {output}/info.json metadata about this tokenization run
"""

import argparse
import json
import os
from itertools import chain
from datetime import datetime

from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

import config as C


def tokenize_and_pack(model_name: str, output_path: str,
                      subset: str | None, max_length: int,
                      num_proc: int, eval_ratio: float):

    print(f"\n{'='*60}")
    print(f"  Pre-Tokenize and Pack — TeleSpec-Data")
    print(f"{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Dataset    : {C.TELESPEC_DATASET} (subset={subset or 'all'})")
    print(f"  Output     : {output_path}")
    print(f"  Max length : {max_length}")
    print(f"  CPU cores  : {num_proc}\n")

    # ── Load dataset ──────────────────────────────────────────────────────
    print("  Loading TeleSpec-Data from HuggingFace...")
    if subset:
        ds = load_dataset(C.TELESPEC_DATASET, name=subset, split="train")
    else:
        ds_3gpp = load_dataset(C.TELESPEC_DATASET, name="3gpp-standard", split="train")
        ds_etsi = load_dataset(C.TELESPEC_DATASET, name="etsi-standard",  split="train")
        ds = concatenate_datasets([ds_3gpp, ds_etsi])

    print(f"  Total documents : {len(ds):,}")

    # Keep only the content field — drop id, category, metadata
    ds = ds.select_columns(["content"]).rename_column("content", "text")

    # Shuffle before splitting so eval is representative
    ds = ds.shuffle(seed=C.TRAIN["seed"])
    split      = ds.train_test_split(test_size=eval_ratio, seed=C.TRAIN["seed"])
    train_docs = split["train"]
    eval_docs  = split["test"]
    print(f"  Train docs : {len(train_docs):,}  |  Eval docs : {len(eval_docs):,}")

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Tokenize and pack functions ───────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation = False,   # don't truncate — pack instead
            padding    = False,
        )

    def pack_fn(examples):
        """
        Standard group_texts approach:
        Concatenate all token sequences into one stream, then cut into
        exact max_length blocks. The last incomplete block is dropped.
        Labels = input_ids (standard causal LM, shifted internally by model).
        """
        concatenated  = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length   = len(concatenated["input_ids"])
        # Drop the last incomplete block
        total_length   = (total_length // max_length) * max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(output_path, exist_ok=True)

    stats = {}
    for split_name, split_ds in [("train", train_docs), ("eval", eval_docs)]:
        print(f"\n  [{split_name}] Tokenising {len(split_ds):,} documents...")
        tokenised = split_ds.map(
            tokenize_fn,
            batched       = True,
            batch_size    = 1000,
            num_proc      = num_proc,
            remove_columns = ["text"],
            desc          = f"  tokenising {split_name}",
        )

        print(f"  [{split_name}] Packing into {max_length}-token blocks...")
        packed = tokenised.map(
            pack_fn,
            batched    = True,
            batch_size = 1000,
            num_proc   = num_proc,
            desc       = f"  packing {split_name}",
        )

        out = os.path.join(output_path, split_name)
        packed.save_to_disk(out)

        total_tokens = len(packed) * max_length
        stats[split_name] = {
            "documents":    len(split_ds),
            "packed_blocks": len(packed),
            "total_tokens": total_tokens,
        }
        print(f"  [{split_name}] {len(packed):,} blocks × {max_length} tokens "
              f"= {total_tokens:,} tokens → {out}")

    # ── Save metadata ─────────────────────────────────────────────────────
    info = {
        "model_name":   model_name,
        "dataset":      C.TELESPEC_DATASET,
        "subset":       subset or "all",
        "max_length":   max_length,
        "eval_ratio":   eval_ratio,
        "created_at":   datetime.now().isoformat(),
        "splits":       stats,
    }
    info_path = os.path.join(output_path, "info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    total_blocks  = sum(s["packed_blocks"] for s in stats.values())
    total_tokens  = sum(s["total_tokens"]  for s in stats.values())
    print(f"\n  {'='*60}")
    print(f"  Total packed blocks : {total_blocks:,}")
    print(f"  Total tokens        : {total_tokens:,}")
    print(f"  Info saved          : {info_path}")
    print(f"\n  Now run training:")
    print(f"    python train.py --model {model_name}")
    print(f"  {'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pre-tokenize and pack TeleSpec-Data for training"
    )
    parser.add_argument("--model",       default=C.ACTIVE_MODEL,
                        help=f"HuggingFace model ID (config default: {C.ACTIVE_MODEL})")
    parser.add_argument("--subset",      default=C.TELESPEC_SUBSET,
                        choices=["3gpp-standard", "etsi-standard"],
                        help="Subset to tokenize (default: all)")
    parser.add_argument("--output",      default=C.TOKENIZED_DATASET_DIR,
                        help=f"Output directory (config default: {C.TOKENIZED_DATASET_DIR})")
    parser.add_argument("--max-length",  type=int, default=C.TRAIN["max_length"],
                        dest="max_length")
    parser.add_argument("--num-proc",    type=int, default=8,
                        dest="num_proc")
    parser.add_argument("--eval-ratio",  type=float, default=C.TRAIN["eval_ratio"],
                        dest="eval_ratio")
    args = parser.parse_args()

    tokenize_and_pack(
        model_name  = args.model,
        output_path = args.output,
        subset      = args.subset,
        max_length  = args.max_length,
        num_proc    = args.num_proc,
        eval_ratio  = args.eval_ratio,
    )
