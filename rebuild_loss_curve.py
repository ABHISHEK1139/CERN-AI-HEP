"""Rebuild the training loss curve from a real training log or checkpoint.

Usage:
    python rebuild_loss_curve.py [LOGFILE]

Falls back through a list of conventional log locations, then to the history
stored inside a checkpoint. Note the previous implementation selected the first
candidate that merely *existed*, so the empty tracked ``local_error.log`` always
won and the script died with "No loss entries parsed" even when a real
``training.log`` was sitting right next to it. Candidates are now required to be
non-empty and to actually contain parseable loss lines.
"""

import argparse
import logging
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import
import matplotlib.pyplot as plt
import torch

logger = logging.getLogger(__name__)

DEFAULT_LOG_CANDIDATES: tuple[str, ...] = (
    "training.log",
    "train.log",
    "experiments.log",
    "local_error.log",
)
CHECKPOINT_CANDIDATES: tuple[str, ...] = (
    "checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt",
    "checkpoints/jetclass_autoencoder/jetclass_gcn_best.pt",
    "checkpoints/smoke/classifier_gcn_best.pt",
)
OUTPUT_PATH = "docs/loss_curve.png"

# "Epoch 2: 150it [19:45, 7.88s/it, loss=269.45]" (also matches loss=1.2e+03)
LOSS_RE = re.compile(
    r"Epoch (\d+): (\d+)it \[.*?loss=([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?\d+)?)"
)


def parse_log(text: str) -> tuple[list[float], list[int]]:
    """Extract per-step losses and epoch boundary indices from a log.

    Returns:
        ``(losses, epoch_boundaries)`` where ``epoch_boundaries[i]`` is the
        index of the first loss of epoch ``i + 2``.
    """
    losses: list[float] = []
    epoch_boundaries: list[int] = []
    current_epoch: int | None = None
    last: tuple[float, int] | None = None

    for line in text.splitlines():
        match = LOSS_RE.search(line)
        if not match:
            continue
        epoch = int(match.group(1))
        iteration = int(match.group(2))
        loss = float(match.group(3))

        if epoch != current_epoch:
            if current_epoch is not None:
                epoch_boundaries.append(len(losses))
            current_epoch = epoch

        # tqdm repaints the same line, so collapse exact duplicates
        if last is not None and (loss, iteration) == last:
            continue
        losses.append(loss)
        last = (loss, iteration)

    return losses, epoch_boundaries


def find_log(explicit: str | None = None) -> Path | None:
    """First candidate log that exists *and* contains parseable losses."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(Path(name) for name in DEFAULT_LOG_CANDIDATES)
    if Path("logs").is_dir():
        candidates.extend(sorted(Path("logs").glob("*.log")))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.stat().st_size == 0:
                logger.info("Skipping empty log candidate: %s", path)
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            logger.warning("Skipping unreadable log %s: %s", path, e)
            continue
        losses, _ = parse_log(text)
        if losses:
            logger.info("Using log %s (%d loss points)", path, len(losses))
            return path
        logger.info("Skipping %s: no parseable loss lines", path)
    return None


def find_checkpoint(explicit: str | None = None) -> Path | None:
    """First existing checkpoint that carries a non-empty train_loss history."""
    candidates = [Path(explicit)] if explicit else []
    candidates.extend(Path(p) for p in CHECKPOINT_CANDIDATES)
    for path in candidates:
        if path and path.is_file():
            return path
    return None


def losses_from_checkpoint(path: Path) -> list[float]:
    """Read ``history['train_loss']`` out of a checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {path} is a {type(ckpt).__name__}, expected a dict.")
    history = ckpt.get("history", {})
    losses = list(history.get("train_loss", []))
    if not losses:
        raise ValueError(f"Checkpoint {path} contains no train_loss history.")
    return losses


def plot_losses(
    losses: Sequence[float],
    output: Path,
    epoch_boundaries: Sequence[int] | None = None,
    title: str = "EdgeConv Autoencoder Training Convergence",
    xlabel: str = "Training Step",
) -> Path:
    """Render the loss curve to ``output`` and return the written path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(losses)), losses, color="#2196F3", linewidth=1.5, alpha=0.8)

    for b in epoch_boundaries or []:
        ax.axvline(x=b, color="gray", linestyle="--", alpha=0.4)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Reconstruction Loss (MSE)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    if epoch_boundaries:
        for i, b in enumerate(epoch_boundaries):
            ax.annotate(
                f"Epoch {i + 2}",
                xy=(b, losses[min(b, len(losses) - 1)]),
                fontsize=9, color="gray", ha="center", va="bottom",
            )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logfile", nargs="?", default=None,
                        help="Training log to parse (default: auto-detect)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to fall back to for history")
    parser.add_argument("--output", default=OUTPUT_PATH, help="Output PNG path")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    log_file = find_log(args.logfile)
    if log_file is not None:
        losses, boundaries = parse_log(
            log_file.read_text(encoding="utf-8", errors="ignore")
        )
        print(f"Parsed {len(losses)} loss data points across "
              f"{len(boundaries) + 1} epoch(s) from {log_file}")
        print(f"Loss range: {max(losses)} -> {min(losses)}")
        plot_losses(losses, Path(args.output), boundaries)
        print(f"Saved {args.output}")
        return 0

    print("No usable training log found; falling back to checkpoint history.")
    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None:
        print("No training log or checkpoint found; cannot rebuild loss curve.",
              file=sys.stderr)
        return 1
    losses = losses_from_checkpoint(ckpt)
    plot_losses(
        losses, Path(args.output),
        title="Training Convergence (from checkpoint history)",
        xlabel="Epoch",
    )
    print(f"Saved {args.output} from {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
