"""Plot training loss curve from Hugging Face trainer state files."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_history(output_dir):
    """Load de-duplicated train/eval history from checkpoint trainer_state.json files."""
    checkpoint_dirs = sorted(
        [path for path in Path(output_dir).iterdir() if path.is_dir() and path.name.startswith("checkpoint-")],
        key=lambda path: int(path.name.split("-")[1]),
    )

    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {output_dir}")

    merged_history = {}
    for checkpoint_dir in checkpoint_dirs:
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        if not trainer_state_path.exists():
            continue
        with trainer_state_path.open("r", encoding="utf-8") as handle:
            trainer_state = json.load(handle)
        for entry in trainer_state.get("log_history", []):
            step = entry.get("step")
            if step is None:
                continue
            merged_history[(step, "eval_loss" in entry)] = entry

    history = [merged_history[key] for key in sorted(merged_history)]
    train_entries = [entry for entry in history if "loss" in entry]
    eval_entries = [entry for entry in history if "eval_loss" in entry]
    return train_entries, eval_entries


def plot_loss(train_entries, eval_entries):
    steps = [int(entry["step"]) for entry in train_entries]
    losses = [float(entry["loss"]) for entry in train_entries]
    epochs = [float(entry.get("epoch", 0.0)) for entry in train_entries]
    eval_steps = [int(entry["step"]) for entry in eval_entries]
    eval_losses = [float(entry["eval_loss"]) for entry in eval_entries]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Training loss
    ax1.plot(steps, losses, color='#2196F3', alpha=0.4, linewidth=0.8, label='Train Loss (raw)')

    # Smoothed training loss (moving average)
    window = min(20, len(losses) // 5) if len(losses) > 10 else 1
    if window > 1:
        smoothed = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            smoothed.append(sum(losses[start:i+1]) / (i - start + 1))
        ax1.plot(steps, smoothed, color='#1565C0', linewidth=2, label=f'Train Loss (smoothed, w={window})')

    # Eval loss points
    if eval_losses:
        ax1.scatter(eval_steps, eval_losses, color='#F44336', s=80, zorder=5, 
                   marker='*', label='Eval Loss')
        for s, l in zip(eval_steps, eval_losses):
            ax1.annotate(f'{l:.4f}', (s, l), textcoords="offset points", 
                        xytext=(10, 10), fontsize=8, color='#F44336')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('SinCode MLM Fine-Tuning — Experiment 2\n(9wimu9/sinhala_dataset_59m, 500K samples, 1 epoch)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations
    ax1.annotate(f'Start: {losses[0]:.3f}', (steps[0], losses[0]),
                textcoords="offset points", xytext=(15, -10), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    ax1.annotate(f'End: {losses[-1]:.3f}', (steps[-1], losses[-1]),
                textcoords="offset points", xytext=(-60, 15), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Loss reduction annotation
    reduction = ((losses[0] - losses[-1]) / losses[0]) * 100
    ax1.text(0.02, 0.02, f'Loss reduction: {losses[0]:.3f} → {losses[-1]:.3f} ({reduction:+.1f}%)',
            transform=ax1.transAxes, fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    out_path = 'misc/training_loss_v2.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Chart saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "xlm-roberta-sinhala-v2"

    try:
        train_entries, eval_entries = load_history(output_dir)
    except FileNotFoundError as exc:
        print(f"Usage: python plot_training.py <output_dir>")
        print(f"  {exc}")
        sys.exit(1)

    if not train_entries:
        print("No training loss entries found in checkpoint trainer_state.json files.")
        sys.exit(1)

    steps = [int(entry["step"]) for entry in train_entries]
    losses = [float(entry["loss"]) for entry in train_entries]
    print(f"Found {len(train_entries)} training loss entries, {len(eval_entries)} eval loss entries")
    print(f"Steps: {steps[0]} → {steps[-1]}")
    print(f"Loss:  {losses[0]:.3f} → {losses[-1]:.3f} ({((losses[0]-losses[-1])/losses[0])*100:+.1f}%)")

    plot_loss(train_entries, eval_entries)
