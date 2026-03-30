"""
Merge LoRA Adapter → Full Model
================================
Run this between Stage 1 and Stage 2.
Takes the LoRA adapter from Stage 1 and merges it into the base model weights,
producing a standalone model that Stage 2 can fine-tune as a normal model.

Run:
    python merge_adapter.py \
        --base-model  meta-llama/Llama-3.2-3B \
        --adapter     ./checkpoints/stage1-lora \
        --output      ./checkpoints/stage1-merged
"""

import argparse
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

def merge(base_model: str, adapter_path: str, output_path: str):
    print(f"\n  Loading base model : {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",            # merge on CPU to avoid VRAM limits
    )

    print(f"  Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print(f"\n  Done. Merged model saved to: {output_path}")
    print(f"  Use this path as --model in sft.py\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter",    required=True)
    parser.add_argument("--output",     required=True)
    args = parser.parse_args()
    merge(args.base_model, args.adapter, args.output)
