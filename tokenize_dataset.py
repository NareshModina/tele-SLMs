"""
Pre-tokenize and Pre-pack Dataset
===================================
Run ONCE before training. Tokenizes and packs sequences into fixed-length
2048-token blocks so training loads instantly without any tokenization overhead.

Packing eliminates padding waste — our dataset has median 477 tokens/chunk,
so without packing ~75% of each forward pass is wasted on padding tokens.
Pre-packing fixes this without needing flash_attention_2.

Run:
    python tokenize_dataset.py \
        --model      Qwen/Qwen2.5-1.5B \
        --dataset    ./tele-preprocessed \
        --output     ./tele-tokenized \
        --max-length 2048 \
        --num-proc   8
"""

import argparse
import os
from itertools import chain
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer


def tokenize_and_pack(model_name, dataset_path, output_path, max_length, num_proc):
    print(f"\n{'='*60}")
    print(f"  Pre-tokenize and Pre-pack")
    print(f"{'='*60}")
    print(f"  Model      : {model_name}")
    print(f"  Input      : {dataset_path}")
    print(f"  Output     : {output_path}")
    print(f"  Max length : {max_length}")
    print(f"  CPU cores  : {num_proc}\n")

    # Load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=False,   # don't truncate — we pack instead
            padding=False,
        )

    def pack_fn(examples):
        """
        Concatenate all token sequences then split into max_length blocks.
        This is the standard 'group_texts' approach from HuggingFace examples.
        No padding, no waste — every token is used.
        """
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the last incomplete block
        total_length = (total_length // max_length) * max_length

        result = {
            k: [t[i: i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated.items()
        }
        # Labels = input_ids shifted by 1 (standard causal LM)
        result["labels"] = result["input_ids"].copy()
        return result

    os.makedirs(output_path, exist_ok=True)

    for split in ["train", "eval"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            print(f"  Skipping {split} — not found")
            continue

        print(f"  [{split}] Tokenizing...")
        ds = load_from_disk(split_path)

        ds = ds.map(
            tokenize_fn,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            remove_columns=["text"],
            desc=f"  tokenizing {split}",
        )

        print(f"  [{split}] Packing into {max_length}-token blocks...")
        ds = ds.map(
            pack_fn,
            batched=True,
            batch_size=1000,
            num_proc=num_proc,
            desc=f"  packing {split}",
        )

        out = os.path.join(output_path, split)
        ds.save_to_disk(out)

        avg_tokens = max_length  # all blocks are exactly max_length after packing
        print(f"  [{split}] {len(ds):,} packed blocks of {max_length} tokens → {out}")

    print(f"\n  Done. Use --dataset {output_path} for training.\n")
    print(f"  Expected speedup: ~3-4x vs unpacked (no padding waste)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--dataset",    default="./tele-preprocessed")
    parser.add_argument("--output",     default="./tele-tokenized")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--num-proc",   type=int, default=8)
    args = parser.parse_args()

    tokenize_and_pack(
        args.model,
        args.dataset,
        args.output,
        args.max_length,
        args.num_proc,
    )