"""Draw high-resolution, modern pipeline architecture diagram for CERN-AI-HEP."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def draw_diagram():
    # Set 16:9 canvas with GitHub dark-mode background (#0d1117)
    fig, ax = plt.subplots(figsize=(16, 7.5), facecolor='#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Color Palette (Dark theme with glowing accents)
    bg_card = '#161b22'
    border_card = '#30363d'
    accent_blue = '#58a6ff'
    accent_green = '#3fb950'
    accent_purple = '#bc8cff'
    accent_orange = '#d29922'
    accent_red = '#f85149'
    accent_cyan = '#39c5cf'
    text_white = '#f0f6fc'
    text_muted = '#8b949e'
    arrow_color = '#79c0ff'

    # Title & Subtitle Header
    ax.text(8.0, 8.0, "CERN-AI-HEP: Relational GNN Anomaly Detection Pipeline",
            ha="center", va="center", fontsize=18, fontweight="bold", color=text_white)
    ax.text(8.0, 7.55, "End-to-End Unsupervised Reconstruction & Physics Discovery Architecture for LHC Collision Events",
            ha="center", va="center", fontsize=11, color=text_muted)

    # 5 Sequential Pipeline Stages
    stages = [
        {
            "x": 0.5, "y": 1.2, "w": 2.6, "h": 5.7,
            "title": "1. Data Ingestion",
            "accent": accent_cyan,
            "badge": "Multi-Format",
            "items": [
                ("CMS NanoAOD", "ROOT tree via uproot"),
                ("JetClass Benchmark", "6M/100M Jets (Awkward)"),
                ("LHCO 2020 Dijets", "HDF5 Resonant Benchmark"),
                ("Toy Generator", "Offline Synthetic Monte Carlo")
            ],
            "footer": "High-throughput streaming"
        },
        {
            "x": 3.6, "y": 1.2, "w": 2.6, "h": 5.7,
            "title": "2. Graph Construction",
            "accent": accent_blue,
            "badge": "GPU Point Cloud",
            "items": [
                ("3D Coordinates", "(Δη, Δφ, log pT) space"),
                ("16 Node Feats", "Kinematics, PID, Charge"),
                ("Dynamic k-NN", "GPU top-k (k = 8 .. 16)"),
                ("Relational Edges", "Pairwise ΔR & metric weights")
            ],
            "footer": "Sub-millisecond graph build"
        },
        {
            "x": 6.7, "y": 1.2, "w": 2.6, "h": 5.7,
            "title": "3. GNN Encoder",
            "accent": accent_purple,
            "badge": "Dynamic Topology",
            "items": [
                ("EdgeConv (DGCNN)", "Local neighborhood convs"),
                ("GCN / GAT / SAGE", "Alternative GNN baselines"),
                ("PhysicsNeMo", "NVIDIA accelerated kernel"),
                ("Multi-Scale Pooling", "Global mean & max readout")
            ],
            "footer": "37k params, rapid saturation"
        },
        {
            "x": 9.8, "y": 1.2, "w": 2.6, "h": 5.7,
            "title": "4. Latent Space",
            "accent": accent_orange,
            "badge": "Bottleneck z",
            "items": [
                ("Latent Manifold", "Compact z in R^D"),
                ("SM Geometry", "Learns non-anomalous shape"),
                ("t-SNE / UMAP", "Continuous cluster analysis"),
                ("Regularization", "Smooth latent distribution")
            ],
            "footer": "Unsupervised compression"
        },
        {
            "x": 12.9, "y": 1.2, "w": 2.6, "h": 5.7,
            "title": "5. Decoder & Scoring",
            "accent": accent_green,
            "badge": "Anomaly S(G)",
            "items": [
                ("Feature Decoder", "Reconstructs node x_i"),
                ("Loss Metric", "MSE(x, x_hat) residual"),
                ("Anomaly Score", "S(G) = 1/|V| sum ||x - x_hat||^2"),
                ("Prioritization", "AUROC: 0.6808 (JetClass)")
            ],
            "footer": "Exotic / Higgs discovery"
        }
    ]

    for s in stages:
        # Card Background Box
        box = patches.FancyBboxPatch(
            (s["x"], s["y"]), s["w"], s["h"],
            boxstyle="round,pad=0.08,rounding_size=0.22",
            facecolor=bg_card, edgecolor=border_card, linewidth=1.5
        )
        ax.add_patch(box)

        # Stage Header Bar Accent
        header_bar = patches.FancyBboxPatch(
            (s["x"] + 0.08, s["y"] + s["h"] - 0.8), s["w"] - 0.16, 0.72,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor='#21262d', edgecolor=s["accent"], linewidth=1.5
        )
        ax.add_patch(header_bar)

        # Header Title (Centered)
        ax.text(s["x"] + s["w"]/2.0, s["y"] + s["h"] - 0.35, s["title"],
                ha="center", va="center", fontsize=11, fontweight="bold", color=s["accent"])
        ax.text(s["x"] + s["w"]/2.0, s["y"] + s["h"] - 0.62, s["badge"],
                ha="center", va="center", fontsize=8.0, color=text_muted)

        # Content items
        curr_y = s["y"] + s["h"] - 1.25
        for title, detail in s["items"]:
            # Dot bullet
            ax.plot([s["x"] + 0.22], [curr_y], marker="o", markersize=4, color=s["accent"])
            ax.text(s["x"] + 0.40, curr_y, title,
                    ha="left", va="center", fontsize=9.2, fontweight="bold", color=text_white)
            ax.text(s["x"] + 0.40, curr_y - 0.35, detail,
                    ha="left", va="center", fontsize=8.0, color=text_muted)
            curr_y -= 0.92

        # Footer divider line
        ax.plot([s["x"] + 0.2, s["x"] + s["w"] - 0.2], [s["y"] + 0.6, s["y"] + 0.6],
                color='#30363d', linewidth=1)
        ax.text(s["x"] + s["w"]/2.0, s["y"] + 0.3, s["footer"],
                ha="center", va="center", fontsize=8.5, style="italic", color=s["accent"])

    # Connecting Flow Arrows between stages
    for i in range(len(stages) - 1):
        x_start = stages[i]["x"] + stages[i]["w"] + 0.05
        x_end = stages[i+1]["x"] - 0.05
        y_mid = stages[i]["y"] + stages[i]["h"] / 2.0
        ax.annotate("",
                    xy=(x_end, y_mid),
                    xytext=(x_start, y_mid),
                    arrowprops=dict(
                        arrowstyle="-|>", color=arrow_color, lw=2.5,
                        mutation_scale=15
                    ))

    # Bottom Architecture Summary Tag
    tag_bg = patches.FancyBboxPatch(
        (3.0, 0.25), 10.0, 0.6,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        facecolor='#161b22', edgecolor='#30363d', linewidth=1
    )
    ax.add_patch(tag_bg)
    ax.text(8.0, 0.55, "PyTorch Geometric • Dynamic Graph CNN (EdgeConv) • CERN Open Data • PhysicsNeMo Acceleration",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color='#58a6ff')

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    Path("docs").mkdir(parents=True, exist_ok=True)
    plt.savefig("docs/architecture.png", dpi=200, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print("Regenerated docs/architecture.png with modern 16:9 layout.")

if __name__ == "__main__":
    draw_diagram()
