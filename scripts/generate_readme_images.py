"""
Regenerate the README's point-cloud figures.

Writes ``docs/bg_event.png`` and ``docs/sig_event.png`` using the same model
and plotting code the Streamlit dashboard uses, so the committed figures stay
consistent with the application.

Requires the JetClass validation ROOT files under ``data/jetclass/val_5M/`` and
the trained checkpoint at ``checkpoints/jetclass_autoencoder/``.

Usage:
    python scripts/generate_readme_images.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import
import matplotlib.pyplot as plt

from demo import get_sample_jet, plot_error_heatmap, run_inference

OUTPUTS = {
    "bg": ("docs/bg_event.png", "Standard Model Background"),
    "sig": ("docs/sig_event.png", "Non-QCD Jet (Higgs)"),
}


def generate(kind: str, seed: int = 42) -> bool:
    """Render one figure. Returns False if the demo data is unavailable."""
    rel_path, title = OUTPUTS[kind]
    jet = get_sample_jet(kind, seed)
    if jet is None:
        print(
            f"  Skipping {rel_path}: JetClass demo data not found "
            f"(expected data/jetclass/val_5M/*.root)."
        )
        return False

    score, node_mse = run_inference(jet)
    fig = plot_error_heatmap(jet, node_mse)
    fig.suptitle(f"{title} (Reconstruction Error: {score:.1f})",
                 color="white", fontsize=10)

    out = Path(rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  Saved {out} (score {score:.2f})")
    return True


def main() -> int:
    print("Generating README figures...")
    written = sum(generate(kind) for kind in OUTPUTS)
    if written == 0:
        print(
            "No figures generated. Download JetClass validation data first "
            "(see README, 'Datasets Supported').",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
