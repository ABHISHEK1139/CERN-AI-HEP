import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_diagram():
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('off')
    
    # Define flowchart steps
    steps = [
        "CMS NanoAOD\n(ROOT Format)",
        "ROOT Ingestion & Streaming\n(uproot / awkward-array)",
        "Particle Extraction\n(Kinematics, Track Parameters, ID)",
        "GPU Graph Construction\n(k-NN in \u0397-\u03A6 Space, k=8)",
        "EdgeConv GNN Encoder\n(Dynamic Latent Topology)",
        "Latent Space Representation\n(Bottleneck Vector z)",
        "Fully Connected Decoder\n(Accelerated by PhysicsNeMo)",
        "Reconstruction Error\n(Mean Squared Error)",
        "Thresholding & Score Evaluation\n(Anomaly Classification)"
    ]
    
    # Set coordinates for boxes
    y_coords = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    
    # Colors
    box_color = '#e3f2fd'
    border_color = '#1e88e5'
    text_color = '#0d47a1'
    arrow_color = '#37474f'
    
    # Draw boxes and arrows
    for i, text in enumerate(steps):
        # Draw box
        bbox_props = dict(boxstyle="round,pad=0.6", fc=box_color, ec=border_color, lw=2)
        ax.text(0.5, y_coords[i], text, ha="center", va="center", size=11, color=text_color, bbox=bbox_props)
        
        # Draw arrow to next box
        if i < len(steps) - 1:
            ax.annotate("",
                        xy=(0.5, y_coords[i+1] + 0.35),
                        xytext=(0.5, y_coords[i] - 0.35),
                        arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2, shrinkA=0, shrinkB=0))
            
    # Save diagram
    plt.tight_layout()
    plt.savefig("docs/architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Architecture diagram successfully generated: docs/architecture.png")

if __name__ == "__main__":
    draw_diagram()
