"""
SmolLM-TS — Upload models to HuggingFace
=========================================
Uploads pretrained and instruction-tuned checkpoints to HuggingFace Hub.
Model cards (README.md) are copied into each checkpoint directory before upload.

Usage:
    # Upload all models
    python upload_to_hf.py

    # Upload a specific model only
    python upload_to_hf.py --model SmolLM-TS-135M
    python upload_to_hf.py --model SmolLM-TS-135M-it

    # Dry run — show what would be uploaded without actually uploading
    python upload_to_hf.py --dry-run
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, login

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_USERNAME  = "nareshmodina"
CHECKPOINTS  = "./checkpoints"
MODEL_CARDS  = "."          # directory where model card .md files live

# Maps: local checkpoint dir → (HF repo name, model card filename)
UPLOAD_CONFIG = {
    "SmolLM-TS-135M": (
        "SmolLM-TS-135M",
        "SmolLM-TS-135M-modelcard.md",
    ),
    "SmolLM-TS-135M-alpaca": (
        "SmolLM-TS-135M-it",
        "SmolLM-TS-135M-it-modelcard.md",
    ),
    "SmolLM-TS-360M": (
        "SmolLM-TS-360M",
        "SmolLM-TS-360M-modelcard.md",
    ),
    "SmolLM-TS-360M-alpaca": (
        "SmolLM-TS-360M-it",
        "SmolLM-TS-360M-it-modelcard.md",
    ),
    "SmolLM-TS-500M": (
        "SmolLM-TS-500M",
        "SmolLM-TS-500M-modelcard.md",
    ),
    "SmolLM-TS-500M-alpaca": (
        "SmolLM-TS-500M-it",
        "SmolLM-TS-500M-it-modelcard.md",
    ),
    "SmolLM-TS-1.5B": (
        "SmolLM-TS-1.5B",
        "SmolLM-TS-1.5B-modelcard.md",
    ),
    "SmolLM-TS-1.5B-alpaca": (
        "SmolLM-TS-1.5B-it",
        "SmolLM-TS-1.5B-it-modelcard.md",
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_repo_exists(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        return True
    except Exception:
        return False


def upload_model(api: HfApi, ckpt_name: str, repo_name: str,
                 card_filename: str, dry_run: bool = False):

    ckpt_path = Path(CHECKPOINTS) / ckpt_name
    card_path = Path(MODEL_CARDS) / card_filename
    repo_id   = f"{HF_USERNAME}/{repo_name}"

    print(f"\n{'='*60}")
    print(f"  Model     : {ckpt_name}")
    print(f"  Repo      : {repo_id}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Model card: {card_path}")
    print(f"{'='*60}")

    # ── Validate ──────────────────────────────────────────────────────────
    if not ckpt_path.exists():
        print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
        return False

    if not card_path.exists():
        print(f"  [WARN] Model card not found: {card_path}")
        print(f"  Upload will proceed without README.md")
    else:
        # Copy model card as README.md into checkpoint dir
        readme_path = ckpt_path / "README.md"
        if not dry_run:
            shutil.copy(card_path, readme_path)
            print(f"  Copied model card → {readme_path}")
        else:
            print(f"  [DRY RUN] Would copy {card_path} → {readme_path}")

    if dry_run:
        print(f"  [DRY RUN] Would upload {ckpt_path} → {repo_id}")
        return True

    # ── Check repo exists ─────────────────────────────────────────────────
    if not check_repo_exists(api, repo_id):
        print(f"  [ERROR] Repo {repo_id} does not exist on HuggingFace.")
        print(f"  Create it at https://huggingface.co/new-model first.")
        return False

    # ── Upload ────────────────────────────────────────────────────────────
    print(f"  Uploading...")
    try:
        api.upload_folder(
            folder_path     = str(ckpt_path),
            repo_id         = repo_id,
            repo_type       = "model",
            ignore_patterns = ["checkpoint-*/", "*.pt", "rng_state_*.pth"],
        )
        print(f"  ✓ Uploaded → https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"  [ERROR] Upload failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Upload SmolLM-TS models to HuggingFace"
    )
    parser.add_argument("--model",   default=None,
                        help="Upload a specific checkpoint (e.g. SmolLM-TS-135M). "
                             "Omit to upload all available models.")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Show what would be uploaded without actually uploading")
    args = parser.parse_args()

    # ── Login ────────────────────────────────────────────────────────────
    if not args.dry_run:
        token = os.environ.get("HF_TOKEN")
        if token:
            login(token=token)
            print("  Logged in via HF_TOKEN environment variable")
        else:
            print("  No HF_TOKEN found — using cached credentials")
            print("  If this fails, run: huggingface-cli login\n")

    api = HfApi()

    # ── Select models to upload ───────────────────────────────────────────
    if args.model:
        if args.model not in UPLOAD_CONFIG:
            print(f"  [ERROR] Unknown model: {args.model}")
            print(f"  Available: {', '.join(UPLOAD_CONFIG.keys())}")
            sys.exit(1)
        models_to_upload = {args.model: UPLOAD_CONFIG[args.model]}
    else:
        # Upload all checkpoints that exist locally
        models_to_upload = {
            k: v for k, v in UPLOAD_CONFIG.items()
            if (Path(CHECKPOINTS) / k).exists()
        }

    if not models_to_upload:
        print("  No checkpoints found to upload.")
        print(f"  Looking in: {Path(CHECKPOINTS).resolve()}")
        sys.exit(0)

    print(f"\n  Models to upload: {len(models_to_upload)}")
    for ckpt, (repo, _) in models_to_upload.items():
        print(f"    {ckpt} → {HF_USERNAME}/{repo}")

    if args.dry_run:
        print(f"\n  [DRY RUN MODE] No files will be uploaded.\n")

    # ── Upload ────────────────────────────────────────────────────────────
    results = {}
    for ckpt_name, (repo_name, card_filename) in models_to_upload.items():
        ok = upload_model(api, ckpt_name, repo_name, card_filename, args.dry_run)
        results[ckpt_name] = ok

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  UPLOAD SUMMARY")
    print(f"{'='*60}")
    for ckpt, ok in results.items():
        repo = UPLOAD_CONFIG[ckpt][0]
        status = "✓ done" if ok else "✗ failed/skipped"
        print(f"  {status:<12} {ckpt} → {HF_USERNAME}/{repo}")
    print()


if __name__ == "__main__":
    main()