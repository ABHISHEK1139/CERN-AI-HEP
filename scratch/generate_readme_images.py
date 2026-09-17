import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo import get_sample_jet, plot_error_heatmap, run_inference

# Create docs directory if it doesn't exist
os.makedirs("docs", exist_ok=True)

# Generate Background Image (requires demo JetClass data; fail with clear message otherwise)
jet_bg = get_sample_jet("bg", 42)
if jet_bg is None:
    raise SystemExit("Demo data missing: data/jetclass/val_5M/*.root not found. Skipping image generation.")
score_bg, mse_bg = run_inference(jet_bg)
fig_bg = plot_error_heatmap(jet_bg, mse_bg)
fig_bg.suptitle(f"Standard Model Background (Score: {score_bg:.1f})", color="white", fontsize=10)
fig_bg.savefig("docs/bg_event.png", bbox_inches='tight', transparent=True)
plt.close(fig_bg)

# Generate Signal Image
jet_sig = get_sample_jet("sig", 42)
if jet_sig is None:
    raise SystemExit("Demo signal data missing: data/jetclass/val_5M/HTo*.root not found.")
score_sig, mse_sig = run_inference(jet_sig)
fig_sig = plot_error_heatmap(jet_sig, mse_sig)
fig_sig.suptitle(f"Higgs Boson Signal (Score: {score_sig:.1f})", color="white", fontsize=10)
fig_sig.savefig("docs/sig_event.png", bbox_inches='tight', transparent=True)
plt.close(fig_sig)

print("Saved docs/bg_event.png and docs/sig_event.png")
