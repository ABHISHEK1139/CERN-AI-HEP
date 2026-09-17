"""Rebuild the loss curve from the actual training log file."""
import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

log_candidates = [
    Path(r"C:\Users\ak612\.gemini\antigravity\brain\c04da7fe-c9d0-4f9d-8866-d00cfac8762c\.system_generated\tasks\task-1579.log"),
    Path(r"C:\Users\ak612\.gemini\antigravity\brain\c04da7fe-c9d0-4f9d-8866-d00cfac8762c\.system_generated\tasks\task-1540.log"),
    Path("local_error.log"),
    Path("training.log"),
]

log_file = next((p for p in log_candidates if p.exists()), None)
if log_file is None:
    # Fall back: reuse checkpoint history if present, else exit gracefully.
    ckpt_candidates = [
        Path("checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"),
        Path("checkpoints/smoke/classifier_gcn_best.pt"),
    ]
    ckpt = next((p for p in ckpt_candidates if p.exists()), None)
    if ckpt is None:
        sys.exit("No training log or checkpoint found; cannot rebuild loss curve.")
    import torch

    history = torch.load(ckpt, map_location="cpu", weights_only=False).get("history", {})
    losses = list(history.get("train_loss", []))
    if not losses:
        sys.exit(f"Checkpoint {ckpt} contains no train_loss history.")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(losses)), losses, color="#2196F3", linewidth=1.5, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Convergence (from checkpoint history)", fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("docs/loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved docs/loss_curve.png from {ckpt}")
    sys.exit(0)

losses = []
iterations = []
epoch_boundaries = []
current_epoch = None

with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # Match patterns like "Epoch 2: 150it [19:45, 7.88s/it, loss=269]"
        match = re.search(r'Epoch (\d+): (\d+)it \[.*loss=(\d+)', line)
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            loss = int(match.group(3))
            
            if epoch != current_epoch:
                if current_epoch is not None:
                    epoch_boundaries.append(len(losses))
                current_epoch = epoch
            
            # Only record every other line (avoid duplicates from tqdm)
            if len(losses) == 0 or loss != losses[-1] or iteration != iterations[-1]:
                losses.append(loss)
                iterations.append(len(losses))

print(f"Parsed {len(losses)} loss data points across epochs")
print(f"Loss range: {max(losses)} -> {min(losses)}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(losses)), losses, color='#2196F3', linewidth=1.5, alpha=0.8)

# Mark epoch boundaries
for b in epoch_boundaries:
    ax.axvline(x=b, color='gray', linestyle='--', alpha=0.4)

ax.set_xlabel('Training Step', fontsize=12)
ax.set_ylabel('Reconstruction Loss (MSE)', fontsize=12)
ax.set_title('EdgeConv Autoencoder Training Convergence', fontsize=14)
ax.grid(True, alpha=0.3)

# Add epoch labels
if epoch_boundaries:
    for i, b in enumerate(epoch_boundaries):
        ax.annotate(f'Epoch {i+2}', xy=(b, losses[min(b, len(losses)-1)]),
                     fontsize=9, color='gray', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('docs/loss_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved docs/loss_curve.png")
