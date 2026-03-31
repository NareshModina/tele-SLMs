"""
tele-SLMs — End-to-End Pipeline
=================================
Runs the full training and evaluation pipeline sequentially for one
or more models. Stages execute in strict order — each must complete
before the next begins.

Pipeline order (per model):
    1. Pretrain   — train.py         full fine-tuning on TeleSpec-Data
    2. Plot A     — plot_training.py pretrain training curves
    3. SFT-A      — sft.py           LoRA SFT on pretrained checkpoint
    4. SFT-B      — sft.py --base    LoRA SFT on base model
    5. Plot B     — plot_training.py SFT-A training curves
    6. Plot C     — plot_training.py SFT-B training curves
    7. Eval-A     — eval.py          evaluate pretrain + SFT
    8. Eval-B     — eval.py          evaluate SFT only
    9. Summary    — eval.py          print leaderboard

Usage:
    # Run active model from config.py
    python pipeline.py

    # Run a specific model
    python pipeline.py --model HuggingFaceTB/SmolLM2-135M

    # Run full model ladder from config.MODELS
    python pipeline.py --all-models

    # Skip pretrain (checkpoint already exists), go straight to SFT
    python pipeline.py --skip-pretrain

    # Skip pretrain and SFT, go straight to eval
    python pipeline.py --skip-pretrain --skip-sft

    # Force re-run all stages even if checkpoints exist
    python pipeline.py --force

    # Smoke test — 50 steps per training stage, 50 eval examples
    python pipeline.py --max-steps 50 --eval-n 50
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import config as C

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PLOTS_DIR = os.path.join(C.RESULTS_DIR, "plots")


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def tokenized_dataset_exists(path: str) -> bool:
    """Check if pre-tokenized dataset exists with both train and eval splits."""
    return (os.path.isdir(os.path.join(path, "train")) and
            os.path.isdir(os.path.join(path, "eval")))


def checkpoint_exists(path: str) -> bool:
    """Check if a checkpoint directory exists and contains model weights."""
    return (os.path.isdir(path) and
            any(f.endswith((".bin", ".safetensors"))
                for f in os.listdir(path)))


def pretrain_dir(model_name: str) -> str:
    return os.path.join(C.CHECKPOINTS_DIR, model_slug(model_name) + "-telespec")


def sft_merged_dir(input_path: str) -> str:
    slug = (os.path.basename(input_path.rstrip("/"))
            if os.path.exists(input_path)
            else input_path.split("/")[-1])
    return os.path.join(C.CHECKPOINTS_DIR, slug + "-sft-merged")


def sft_adapter_dir(input_path: str) -> str:
    slug = (os.path.basename(input_path.rstrip("/"))
            if os.path.exists(input_path)
            else input_path.split("/")[-1])
    return os.path.join(C.CHECKPOINTS_DIR, slug + "-sft")


def trainer_state(checkpoint_path: str) -> str:
    """
    Find trainer_state.json inside the latest checkpoint-N subdirectory.
    HuggingFace Trainer saves it under checkpoint-{step}/, not the root dir.
    Falls back to root dir for older/custom setups.
    """
    # Find all checkpoint-N subdirs and pick the one with the highest step
    subdirs = []
    if os.path.isdir(checkpoint_path):
        for entry in os.listdir(checkpoint_path):
            full = os.path.join(checkpoint_path, entry)
            if os.path.isdir(full) and entry.startswith("checkpoint-"):
                try:
                    step = int(entry.split("-")[-1])
                    subdirs.append((step, full))
                except ValueError:
                    pass

    if subdirs:
        latest = max(subdirs, key=lambda x: x[0])[1]
        return os.path.join(latest, "trainer_state.json")

    # Fallback: root dir
    return os.path.join(checkpoint_path, "trainer_state.json")


def plot_path(filename: str) -> str:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return os.path.join(PLOTS_DIR, filename)


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def run_stage(label: str, cmd: list[str], log_path: str,
              use_torchrun: bool = False, nproc: int = 1,
              port: int = C.DDP_MASTER_PORT) -> bool:
    """
    Run a single pipeline stage as a subprocess.
    Returns True on success, False on failure.
    Streams output to both terminal and log file.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Wrap training stages with torchrun for multi-GPU
    if use_torchrun and nproc > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            f"--master_port={port}",
        ] + cmd
    else:
        cmd = [sys.executable] + cmd

    print(f"\n{'─'*60}")
    print(f"  STAGE : {label}")
    print(f"  CMD   : {' '.join(cmd)}")
    print(f"  LOG   : {log_path}")
    print(f"{'─'*60}")

    t_start = time.time()

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="")          # stream to terminal
            log_file.write(line)         # write to log
            log_file.flush()
        process.wait()

    elapsed = time.time() - t_start
    status  = "✓ DONE" if process.returncode == 0 else "✗ FAILED"
    print(f"\n  {status}  ({elapsed:.0f}s)  →  {log_path}")

    return process.returncode == 0


# ---------------------------------------------------------------------------
# Pipeline per model
# ---------------------------------------------------------------------------

def run_pipeline(model_name: str, args: argparse.Namespace,
                 pipeline_log: list[dict]):
    slug  = model_slug(model_name)
    nproc = args.nproc

    # Ports — increment between pretrain and SFT to avoid conflicts
    port_pretrain = C.DDP_MASTER_PORT
    port_sft      = C.DDP_MASTER_PORT + 1

    print(f"\n{'='*60}")
    print(f"  PIPELINE : {slug}")
    print(f"  GPUs     : {nproc}")
    print(f"{'='*60}")

    def log_stage(stage, status, elapsed=0, note=""):
        pipeline_log.append({
            "model": slug, "stage": stage, "status": status,
            "elapsed_s": round(elapsed), "note": note,
        })

    # ── Stage 0: Tokenize dataset ───────────────────────────────────────
    tok_dir = C.TOKENIZED_DATASET_DIR

    if tokenized_dataset_exists(tok_dir) and not args.force:
        print(f"\n  [SKIP] Tokenize — pre-tokenized dataset exists at {tok_dir}")
        log_stage("tokenize", "skipped (exists)", note=tok_dir)
    else:
        cmd = ["tokenize_dataset.py", "--model", model_name]
        t = time.time()
        ok = run_stage(
            label    = f"Tokenize dataset — {slug}",
            cmd      = cmd,
            log_path = os.path.join(C.LOGS_DIR, f"{slug}-tokenize.log"),
        )
        elapsed = time.time() - t
        if not ok:
            log_stage("tokenize", "FAILED", elapsed)
            print(f"\n  Pipeline stopped. Check logs/{slug}-tokenize.log")
            return False
        log_stage("tokenize", "done", elapsed, note=tok_dir)

    # ── Stage 1: Pretrain ─────────────────────────────────────────────────
    pt_dir = pretrain_dir(model_name)

    if args.skip_pretrain:
        print(f"\n  [SKIP] Pretrain — --skip-pretrain set")
        if not checkpoint_exists(pt_dir):
            print(f"  [WARN] No checkpoint found at {pt_dir} — SFT may fail")
        log_stage("pretrain", "skipped", note=pt_dir)

    elif checkpoint_exists(pt_dir) and not args.force:
        print(f"\n  [SKIP] Pretrain — checkpoint exists at {pt_dir}")
        log_stage("pretrain", "skipped (exists)", note=pt_dir)

    else:
        cmd = ["train.py", "--model", model_name,
               "--tokenized-path", C.TOKENIZED_DATASET_DIR]
        if args.max_steps > 0:
            cmd += ["--max-steps", str(args.max_steps)]

        t = time.time()
        ok = run_stage(
            label       = f"Pretrain — {slug}",
            cmd         = cmd,
            log_path    = os.path.join(C.LOGS_DIR, f"{slug}-pretrain.log"),
            use_torchrun = True,
            nproc       = nproc,
            port        = port_pretrain,
        )
        elapsed = time.time() - t
        if not ok:
            log_stage("pretrain", "FAILED", elapsed)
            print(f"\n  Pipeline stopped. Check logs/{slug}-pretrain.log")
            return False
        log_stage("pretrain", "done", elapsed, note=pt_dir)

    # ── Plot 1: Pretrain curves ───────────────────────────────────────────
    ts_path = trainer_state(pt_dir)
    if os.path.exists(ts_path):
        t = time.time()
        ok = run_stage(
            label    = f"Plot pretrain curves — {slug}",
            cmd      = [
                "plot_training.py",
                "--json",   ts_path,
                "--output", plot_path(f"{slug}-pretrain_curves.png"),
                "--title",  f"{slug} — Stage 1 Pretrain",
            ],
            log_path = os.path.join(C.LOGS_DIR, f"{slug}-plot-pretrain.log"),
        )
        log_stage("plot-pretrain", "done" if ok else "FAILED", time.time() - t)
    else:
        print(f"\n  [SKIP] Plot pretrain — trainer_state.json not found at {ts_path}")
        log_stage("plot-pretrain", "skipped (no trainer_state.json)")

    # ── Stage 2A: SFT on pretrained checkpoint ────────────────────────────
    sft_a_merged = sft_merged_dir(pt_dir)
    sft_a_adapter = sft_adapter_dir(pt_dir)

    if args.skip_sft:
        print(f"\n  [SKIP] SFT-A — --skip-sft set")
        log_stage("sft-A", "skipped")

    elif checkpoint_exists(sft_a_merged) and not args.force:
        print(f"\n  [SKIP] SFT-A — merged checkpoint exists at {sft_a_merged}")
        log_stage("sft-A", "skipped (exists)", note=sft_a_merged)

    else:
        cmd = ["sft.py", "--input", pt_dir]
        if args.max_steps > 0:
            cmd += ["--max-steps", str(args.max_steps)]

        t = time.time()
        ok = run_stage(
            label        = f"SFT-A (pretrain→SFT) — {slug}",
            cmd          = cmd,
            log_path     = os.path.join(C.LOGS_DIR, f"{slug}-sft-A.log"),
            use_torchrun = True,
            nproc        = nproc,
            port         = port_sft,
        )
        elapsed = time.time() - t
        if not ok:
            log_stage("sft-A", "FAILED", elapsed)
            print(f"\n  Pipeline stopped. Check logs/{slug}-sft-A.log")
            return False
        log_stage("sft-A", "done", elapsed, note=sft_a_merged)

    # ── Stage 2B: SFT on base model ───────────────────────────────────────
    sft_b_merged  = sft_merged_dir(model_name)
    sft_b_adapter = sft_adapter_dir(model_name)

    if args.skip_sft:
        print(f"\n  [SKIP] SFT-B — --skip-sft set")
        log_stage("sft-B", "skipped")

    elif checkpoint_exists(sft_b_merged) and not args.force:
        print(f"\n  [SKIP] SFT-B — merged checkpoint exists at {sft_b_merged}")
        log_stage("sft-B", "skipped (exists)", note=sft_b_merged)

    else:
        cmd = ["sft.py", "--input", model_name, "--base"]
        if args.max_steps > 0:
            cmd += ["--max-steps", str(args.max_steps)]

        t = time.time()
        ok = run_stage(
            label        = f"SFT-B (base→SFT) — {slug}",
            cmd          = cmd,
            log_path     = os.path.join(C.LOGS_DIR, f"{slug}-sft-B.log"),
            use_torchrun = True,
            nproc        = nproc,
            port         = port_sft,
        )
        elapsed = time.time() - t
        if not ok:
            log_stage("sft-B", "FAILED", elapsed)
            print(f"\n  Pipeline stopped. Check logs/{slug}-sft-B.log")
            return False
        log_stage("sft-B", "done", elapsed, note=sft_b_merged)

    # ── Plot 2A: SFT-A curves ─────────────────────────────────────────────
    ts_sft_a = trainer_state(sft_a_adapter)
    if os.path.exists(ts_sft_a):
        t = time.time()
        ok = run_stage(
            label    = f"Plot SFT-A curves — {slug}",
            cmd      = [
                "plot_training.py",
                "--json",   ts_sft_a,
                "--output", plot_path(f"{slug}-sft-A_curves.png"),
                "--title",  f"{slug} — Stage 2A SFT (pretrain→SFT)",
            ],
            log_path = os.path.join(C.LOGS_DIR, f"{slug}-plot-sft-A.log"),
        )
        log_stage("plot-sft-A", "done" if ok else "FAILED", time.time() - t)
    else:
        print(f"\n  [SKIP] Plot SFT-A — trainer_state.json not found")
        log_stage("plot-sft-A", "skipped (no trainer_state.json)")

    # ── Plot 2B: SFT-B curves ─────────────────────────────────────────────
    ts_sft_b = trainer_state(sft_b_adapter)
    if os.path.exists(ts_sft_b):
        t = time.time()
        ok = run_stage(
            label    = f"Plot SFT-B curves — {slug}",
            cmd      = [
                "plot_training.py",
                "--json",   ts_sft_b,
                "--output", plot_path(f"{slug}-sft-B_curves.png"),
                "--title",  f"{slug} — Stage 2B SFT (base→SFT)",
            ],
            log_path = os.path.join(C.LOGS_DIR, f"{slug}-plot-sft-B.log"),
        )
        log_stage("plot-sft-B", "done" if ok else "FAILED", time.time() - t)
    else:
        print(f"\n  [SKIP] Plot SFT-B — trainer_state.json not found")
        log_stage("plot-sft-B", "skipped (no trainer_state.json)")

    # ── Eval: two passes per model — standards-only + full ───────────────
    # Eval runs if merged checkpoints exist, regardless of --skip-sft
    if not args.skip_eval and checkpoint_exists(sft_a_merged) and checkpoint_exists(sft_b_merged):
        eval_runs = [
            # (label_suffix, checkpoint, log_suffix, source_filter)
            (f"{slug}-pretrain-sft-standards", sft_a_merged, "eval-A-standards", "standard"),
            (f"{slug}-pretrain-sft-full",      sft_a_merged, "eval-A-full",      "all"),
            (f"{slug}-sft-only-standards",     sft_b_merged, "eval-B-standards", "standard"),
            (f"{slug}-sft-only-full",          sft_b_merged, "eval-B-full",      "all"),
        ]
        for label, ckpt, log_suffix, source_filter in eval_runs:
            t = time.time()
            ok = run_stage(
                label    = f"Eval ({source_filter}) — {label}",
                cmd      = [
                    "eval.py",
                    "--model",         model_name,
                    "--checkpoint",    ckpt,
                    "--label",         label,
                    "--n",             str(args.eval_n),
                    "--source-filter", source_filter,
                ],
                log_path = os.path.join(C.LOGS_DIR, f"{slug}-{log_suffix}.log"),
            )
            elapsed = time.time() - t
            if not ok:
                log_stage(log_suffix, "FAILED", elapsed)
                print(f"\n  Pipeline stopped. Check logs/{slug}-{log_suffix}.log")
                return False
            log_stage(log_suffix, "done", elapsed)

    return True


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_pipeline_summary(pipeline_log: list[dict]):
    print(f"\n{'='*65}")
    print(f"  PIPELINE SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Model':<20} {'Stage':<18} {'Status':<18} {'Time':>8}")
    print(f"  {'-'*63}")
    for entry in pipeline_log:
        time_str = f"{entry['elapsed_s']}s" if entry['elapsed_s'] else "—"
        print(f"  {entry['model']:<20} {entry['stage']:<18} "
              f"{entry['status']:<18} {time_str:>8}")
    print(f"{'='*65}\n")


def save_pipeline_log(pipeline_log: list[dict], model_names: list[str],
                      total_elapsed: float):
    os.makedirs(C.LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slugs     = "_".join(model_slug(m) for m in model_names)
    path      = os.path.join(C.LOGS_DIR, f"pipeline_{slugs}_{timestamp}.log")

    # Capture leaderboard output from eval.py --summary-only
    leaderboard = ""
    try:
        result = subprocess.run(
            [sys.executable, "eval.py", "--summary-only"],
            capture_output=True, text=True,
        )
        leaderboard = result.stdout
    except Exception as e:
        leaderboard = f"  [Could not capture leaderboard: {e}]\n"

    with open(path, "w") as f:
        f.write(f"Pipeline run : {timestamp}\n")
        f.write(f"Models       : {', '.join(model_names)}\n")
        f.write(f"Total time   : {total_elapsed/60:.1f} min\n\n")

        # Pipeline summary table
        f.write("=" * 65 + "\n")
        f.write("PIPELINE SUMMARY\n")
        f.write("=" * 65 + "\n")
        f.write(f"{'Model':<20} {'Stage':<18} {'Status':<18} {'Time':>8}\n")
        f.write("-" * 65 + "\n")
        for entry in pipeline_log:
            time_str = f"{entry['elapsed_s']}s" if entry['elapsed_s'] else "—"
            f.write(f"{entry['model']:<20} {entry['stage']:<18} "
                    f"{entry['status']:<18} {time_str:>8}\n")
            if entry.get("note"):
                f.write(f"  → {entry['note']}\n")

        # Leaderboard
        f.write("\n" + "=" * 65 + "\n")
        f.write("FINAL LEADERBOARD\n")
        f.write("=" * 65 + "\n")
        f.write(leaderboard)

    print(f"  Pipeline log saved → {path}")


# ---------------------------------------------------------------------------
# Interactive model selection
# ---------------------------------------------------------------------------

def prompt_model_selection() -> str:
    """
    Display numbered list of models from config.MODELS and prompt
    the user to select one by number.
    """
    print(f"\n{'='*60}")
    print(f"  tele-SLMs — Model Selection")
    print(f"{'='*60}")
    print(f"  Available models:\n")

    for i, model in enumerate(C.MODELS, start=1):
        slug = model.split("/")[-1]
        print(f"    [{i}] {slug:<30}  ({model})")

    print(f"\n    [0] Run all models sequentially")
    print()

    while True:
        try:
            choice = input("  Enter number: ").strip()
            idx = int(choice)
            if idx == 0:
                # Signal to run all models — handled in main()
                return "__all__"
            if 1 <= idx <= len(C.MODELS):
                selected = C.MODELS[idx - 1]
                print(f"\n  Selected: {selected}\n")
                return selected
            print(f"  Please enter a number between 0 and {len(C.MODELS)}")
        except (ValueError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)



def prompt_smoke_test() -> bool:
    """Ask the user whether this is a smoke test or a full run."""
    print(f"  Run mode:\n")
    print(f"    [1] Full run      (config defaults — full epochs, {C.EVAL['n_examples']} eval examples)")
    print(f"    [2] Smoke test    (50 training steps, 50 eval examples)")
    print(f"                      note: tokenization always runs in full)\n")

    while True:
        try:
            choice = input("  Enter number: ").strip()
            if choice == "1":
                print(f"\n  Mode: Full run\n")
                return False
            elif choice == "2":
                print(f"\n  Mode: Smoke test (50 steps / 50 eval examples)\n")
                return True
            print("  Please enter 1 or 2")
        except (ValueError, KeyboardInterrupt):
            print("\n  Aborted.")
            sys.exit(0)



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="tele-SLMs end-to-end pipeline"
    )
    parser.add_argument("--model",       default=None,
                        help="Model to run (omit to get interactive selection)")
    parser.add_argument("--all-models",  action="store_true", dest="all_models",
                        help="Run full model ladder from config.MODELS sequentially")
    parser.add_argument("--nproc",       type=int, default=C.NPROC,
                        help=f"GPUs for training stages (config default: {C.NPROC})")
    parser.add_argument("--force",       action="store_true",
                        help="Re-run all stages even if checkpoints exist")
    parser.add_argument("--skip-pretrain", action="store_true", dest="skip_pretrain",
                        help="Skip Stage 1 pretrain (checkpoint must already exist)")
    parser.add_argument("--skip-sft",    action="store_true", dest="skip_sft",
                        help="Skip SFT stages")
    parser.add_argument("--skip-eval",   action="store_true", dest="skip_eval",
                        help="Skip all eval stages")
    parser.add_argument("--max-steps",   type=int, default=-1, dest="max_steps",
                        help="Hard stop for training stages — use 50 for smoke test")
    parser.add_argument("--eval-n",      type=int, default=C.EVAL["n_examples"],
                        dest="eval_n",
                        help=f"Tele-Eval examples (config default: {C.EVAL['n_examples']})")
    args = parser.parse_args()

    # Determine model list
    if args.all_models:
        models = C.MODELS
    elif args.model:
        models = [args.model]
    else:
        selection = prompt_model_selection()
        models = C.MODELS if selection == "__all__" else [selection]

        # Only prompt for smoke test in interactive mode
        if args.max_steps < 0:  # not already set via CLI
            is_smoke = prompt_smoke_test()
            if is_smoke:
                args.max_steps = 50
                args.eval_n    = 50

    print(f"\n{'='*60}")
    print(f"  tele-SLMs Pipeline")
    print(f"  Models  : {', '.join(model_slug(m) for m in models)}")
    print(f"  GPUs    : {args.nproc}")
    print(f"  Force   : {args.force}")
    print(f"  Eval n  : {args.eval_n}")
    if args.max_steps > 0:
        print(f"  Max steps (smoke): {args.max_steps}")
    print(f"{'='*60}")

    os.makedirs(C.LOGS_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    pipeline_log = []
    t_total      = time.time()

    for model_name in models:
        ok = run_pipeline(model_name, args, pipeline_log)
        if not ok:
            print(f"\n  Pipeline failed on {model_slug(model_name)}. Stopping.")
            break

    # Print leaderboard
    print(f"\n{'─'*60}")
    print(f"  Final leaderboard:")
    subprocess.run([sys.executable, "eval.py", "--summary-only"])

    # Print and save pipeline summary
    total_elapsed = time.time() - t_total
    print_pipeline_summary(pipeline_log)
    save_pipeline_log(pipeline_log, models, total_elapsed)
    print(f"  Total wall time: {total_elapsed/60:.1f} min\n")


if __name__ == "__main__":
    main()