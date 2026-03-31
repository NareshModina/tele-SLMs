"""
tele-SLMs — Training Curve Plotter
=====================================
Plots loss, token accuracy, and learning rate schedule from a
trainer_state.json file produced by the Hugging Face Trainer.

Works with both the LoRA pilot (Phase 1/2) and the full fine-tuning pipeline.

Usage:
    python plot_training.py \
        --json   ./checkpoints/SmolLM2-135M-telespec/trainer_state.json \
        --output ./results/SmolLM2-135M_curves.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_history(json_path: str) -> dict:
    with open(json_path) as f:
        state = json.load(f)

    train_steps, train_loss, train_acc, train_lr = [], [], [], []
    eval_steps,  eval_loss                       = [], []

    for entry in state.get("log_history", []):
        step = entry.get("step", entry.get("global_step"))
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
            train_acc.append(entry.get("mean_token_accuracy"))
            train_lr.append(entry.get("learning_rate"))
        elif "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])

    return {
        "train_steps": train_steps,
        "train_loss":  train_loss,
        "train_acc":   train_acc,
        "train_lr":    train_lr,
        "eval_steps":  eval_steps,
        "eval_loss":   eval_loss,
        "best_step":   state.get("best_global_step"),
        "best_metric": state.get("best_metric"),
        "total_steps": state.get("global_step"),
        "epochs":      state.get("epoch"),
    }


def smooth(values: list, window: int = 20) -> list:
    """Simple centred moving average."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window // 2)
        end   = min(len(values), i + window // 2 + 1)
        result.append(sum(values[start:end]) / (end - start))
    return result


def plot(data: dict, output_path: str, title_prefix: str = ""):
    has_eval = len(data["eval_steps"]) > 0
    has_acc  = any(v is not None for v in data["train_acc"])
    has_lr   = any(v is not None for v in data["train_lr"])

    n_rows = 1 + int(has_acc) + int(has_lr)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]

    subtitle = (f"{data['epochs']:.1f} epochs  |  {data['total_steps']:,} steps"
                if data["epochs"] and data["total_steps"] else "")
    title = f"{title_prefix} — {subtitle}" if title_prefix else subtitle
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(data["train_steps"], data["train_loss"],
            color="#b0c4de", alpha=0.4, linewidth=0.8, label="Train loss (raw)")
    ax.plot(data["train_steps"], smooth(data["train_loss"], window=30),
            color="#1f77b4", linewidth=2.0, label="Train loss (smoothed)")

    if has_eval:
        ax.plot(data["eval_steps"], data["eval_loss"],
                color="#ff7f0e", linewidth=2.0, marker="o",
                markersize=4, label="Eval loss")

    if data["best_step"]:
        ax.axvline(data["best_step"], color="red", linestyle="--",
                   linewidth=1.2, alpha=0.7)
        ax.text(data["best_step"], ax.get_ylim()[0],
                f" best\n step={data['best_step']}\n"
                f" loss={data['best_metric']:.4f}",
                color="red", fontsize=8, va="bottom")

    ax.set_ylabel("Loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("Training & Eval Loss", fontsize=11)

    # ── Token accuracy ────────────────────────────────────────────────────
    if has_acc:
        ax = axes[1]
        acc_vals  = [v for v in data["train_acc"]   if v is not None]
        acc_steps = [s for s, v in zip(data["train_steps"], data["train_acc"]) if v is not None]
        ax.plot(acc_steps, acc_vals,
                color="#aec7a0", alpha=0.4, linewidth=0.8, label="Token accuracy (raw)")
        ax.plot(acc_steps, smooth(acc_vals, window=30),
                color="#2ca02c", linewidth=2.0, label="Token accuracy (smoothed)")
        ax.set_ylabel("Token Accuracy", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Mean Token Accuracy", fontsize=11)

    # ── Learning rate ─────────────────────────────────────────────────────
    if has_lr:
        ax = axes[-1]
        lr_vals  = [v for v in data["train_lr"]   if v is not None]
        lr_steps = [s for s, v in zip(data["train_steps"], data["train_lr"]) if v is not None]
        ax.plot(lr_steps, lr_vals, color="#9467bd", linewidth=1.5, label="Learning rate")
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title("Learning Rate Schedule", fontsize=11)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[-1].set_xlabel("Step", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves from trainer_state.json"
    )
    parser.add_argument("--json",   required=True,
                        help="Path to trainer_state.json")
    parser.add_argument("--output", default="./results/training_curves.png",
                        help="Output image path")
    parser.add_argument("--title",  default="",
                        help="Optional title prefix for the plot")
    args = parser.parse_args()

    data = load_history(args.json)

    print(f"  Total steps    : {data['total_steps']:,}")
    print(f"  Epochs         : {data['epochs']}")
    print(f"  Train entries  : {len(data['train_steps'])}")
    print(f"  Eval entries   : {len(data['eval_steps'])}")
    if data["best_step"]:
        print(f"  Best step      : {data['best_step']}")
        print(f"  Best eval loss : {data['best_metric']:.4f}")

    plot(data, args.output, title_prefix=args.title)


if __name__ == "__main__":
    main()
