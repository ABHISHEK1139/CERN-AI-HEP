import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from glob import glob
import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from graph_builder.jetclass_dataset import JetClassDataset
except ImportError:  # optional heavy dep (uproot/awkward) may be absent
    JetClassDataset = None
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from torch_geometric.data import Batch

# Set page config (only when actually running under `streamlit run`)
def _streamlit_runtime_exists() -> bool:
    try:
        from streamlit.runtime import exists as _exists

        return bool(_exists())
    except Exception:
        return False


_IN_STREAMLIT_RUNTIME = _streamlit_runtime_exists()

if _IN_STREAMLIT_RUNTIME:
    st.set_page_config(page_title="CERN AI: Anomaly Detection Platform", layout="wide", initial_sidebar_state="expanded")

    st.title("CERN AI: GNN Anomaly Detection Research Platform")
    st.markdown("""
This interactive research platform demonstrates an unsupervised anomaly detection pipeline on **3D Particle Clouds** representing collision jets at the LHC. 
It uses a pre-trained **EdgeConv Graph Autoencoder** to rank unusual jet topologies by learning the geometry of Standard Model background-like events.
""")

# Load Model (uncached core; cached alias only inside the Streamlit runtime,
# where st.cache_resource requires a runtime context to be called).
def _load_model_uncached(ckpt_path="checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, hidden_dim, latent_dim = 16, 64, 32
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" not in ckpt:
            raise KeyError(f"{ckpt_path} has no 'model_state_dict'.")
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device


if _IN_STREAMLIT_RUNTIME:
    load_model = st.cache_resource(_load_model_uncached)
else:
    load_model = _load_model_uncached


def _get_model():
    """Lazily load the model so `import demo` stays side-effect free for tests."""
    global model, device
    if globals().get("model") is not None:
        return model, device
    model, device = load_model()
    return model, device


try:
    # Eager load only inside the Streamlit runtime; keep imports side-effect free.
    if _IN_STREAMLIT_RUNTIME:
        model, device = load_model()
    else:
        model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    model, device = None, torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= SIDEBAR =================
if _IN_STREAMLIT_RUNTIME:
    with st.sidebar.expander("Training Statistics Card", expanded=True):
        st.markdown("""
        **Architecture:** EdgeConv Graph Autoencoder  
        **Parameters:** 37,296  
        **Graph Construction:** k-NN (k=8)  
        **Training Dataset:** JetClass (SM Background)  
        **Training Jets:** 6,000,000  
        **Validation Jets:** 1,000,000  
        **Epochs:** 50  
        **Best AUROC:** 0.6808  
        """)

    st.sidebar.header("Platform Controls")
    comparison_mode = st.sidebar.checkbox("Side-by-Side Comparison Mode", value=False)
    st.sidebar.slider(
        "Anomaly threshold (illustrative; calibrate on validation scores)",
        min_value=0.0, max_value=1000.0, value=235.0, step=1.0, key="demo_threshold",
    )

    if not comparison_mode:
        sample_type = st.sidebar.selectbox(
            "Choose Jet Origin Type",
            ["Standard Model Background (Z \u2192 \u03BD\u03BD)", "Higgs Boson Signal (Anomaly)"]
        )
    else:
        sample_type = None

    if "seed" not in st.session_state:
        st.session_state.seed = 42

    if st.sidebar.button("Generate New Collision Event(s)"):
        st.session_state.seed = np.random.randint(0, 100000)
else:
    comparison_mode = False
    sample_type = "Standard Model Background (Z \u2192 \u03BD\u03BD)"

# ================= DATA LOADING =================
def _get_sample_jet_impl(stype, seed):
    if JetClassDataset is None:
        return None
    val_bg_files = sorted(glob("data/jetclass/val_5M/ZJetsToNuNu_*.root"))
    val_sig_files = sorted(glob("data/jetclass/val_5M/HTo*.root"))
    
    if stype == "bg":
        if not val_bg_files: return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_bg_files[0]], k_neighbors=8, sample_size=50, tag="demo_bg")
    else:
        if not val_sig_files: return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_sig_files[0]], k_neighbors=8, sample_size=50, tag="demo_sig")
    
    import random
    random.seed(seed)
    if len(ds) == 0:
        return None
    idx = random.randint(0, len(ds) - 1)
    return ds[idx]


# Public helper used by both the Streamlit UI (cached) and tests (plain).
def get_sample_jet(stype, seed):
    return _get_sample_jet_impl(stype, seed)


if _IN_STREAMLIT_RUNTIME:
    try:
        get_sample_jet = st.cache_data(_get_sample_jet_impl)
    except Exception:
        pass

# ================= CORE FUNCTIONS =================
def run_inference(jet, _model=None, _device=None):
    if jet is None:
        raise ValueError("No jet provided (demo data missing).")
    active_model = _model if _model is not None else globals().get("model")
    active_device = _device if _device is not None else globals().get("device")
    if active_model is None:
        active_model, active_device = _get_model()
    if active_model is None:
        raise RuntimeError("Model not loaded (checkpoint missing).")
    batch = Batch.from_data_list([jet]).to(active_device)
    with torch.no_grad():
        res = active_model(batch)
        score = res['per_graph_loss'].item()
        node_mse = res['per_node_loss'].cpu().numpy()
    return score, node_mse

def plot_error_heatmap(jet, node_mse):
    if getattr(jet, "edge_index", None) is None or jet.edge_index.numel() == 0:
        raise ValueError("plot_error_heatmap needs a jet with edges.")
    G = nx.Graph()
    edge_index = jet.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])

    # Scoped style: never leak dark_background into other figures.
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Use actual physics coordinates (eta, phi) for the node layout
        pos = {i: (jet.x[i, 4].item(), jet.x[i, 5].item()) for i in range(jet.x.size(0))}
        
        # Custom colormap visualization
        sc = nx.draw_networkx_nodes(G, pos, node_size=30, node_color=node_mse, cmap=plt.cm.coolwarm, alpha=0.9, ax=ax, linewidths=0.5, edgecolors='white')
        nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.4, ax=ax)
        
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Reconstruction MSE", fontsize=8, color='lightgray')
        cbar.ax.tick_params(labelsize=7, colors='lightgray')
        
        # Format axes to look like a physics plot
        ax.set_xlabel("\u0394\u03B7 (Pseudo-rapidity)", fontsize=8, color='lightgray')
        ax.set_ylabel("\u0394\u03C6 (Azimuthal)", fontsize=8, color='lightgray')
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=7, colors='lightgray')
        ax.grid(True, linestyle=':', alpha=0.3, color='gray')
        for spine in ax.spines.values():
            spine.set_color('#444444')
            
        plt.tight_layout()
    return fig

def display_metrics(jet):
    if jet is None:
        st.warning("No jet available (demo data missing).")
        return
    # JetClass cols: 0=px, 1=py, 10=charge. pT per particle = hypot(px, py).
    px = jet.x[:, 0]
    py = jet.x[:, 1] if jet.x.size(1) > 1 else torch.zeros_like(px)
    pT = torch.hypot(px, py).sum().item()
    n_const = jet.x.size(0)
    avg_charge = jet.x[:, 10].mean().item()
    
    edge_index = jet.edge_index.cpu().numpy()
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        G.add_edge(edge_index[0, i], edge_index[1, i])
    n_edges = G.number_of_edges()
    avg_degree = (n_edges * 2) / n_const if n_const > 0 else 0
    density = (2 * n_edges) / (n_const * (n_const - 1)) if n_const > 1 else 0
    components = nx.number_connected_components(G)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Physics")
        st.markdown("---")
        st.metric("Total Jet pT (norm)", f"{pT:.2f}")
        st.metric("Constituents", f"{n_const}")
        st.metric("Avg Charge", f"{avg_charge:.2f}")
    with c2:
        st.markdown("##### Topology")
        st.markdown("---")
        st.metric("Nodes", f"{n_const}")
        st.metric("Edges", f"{n_edges}")
        st.metric("Density", f"{density:.3f}")
        st.metric("Components", f"{components}")
        st.metric("Avg Degree", f"{avg_degree:.1f}")

def render_event_column(jet, title):
    if jet is None:
        st.error("Demo data missing: expected JetClass ROOT files under data/jetclass/val_5M/. "
                 "Run the smoke test or add demo data to enable interactive inference.")
        return None, None
        
    score, node_mse = run_inference(jet)
    # Demo threshold is illustrative only, NOT calibrated on a held-out set.
    # Tune it in the sidebar; for research use AnomalyScorer.select_threshold
    # (percentile/sigma, see experiments/configs/*.yaml) on validation scores.
    threshold = st.session_state.get("demo_threshold", 235.0) if _IN_STREAMLIT_RUNTIME else 235.0
    is_anomaly = score > threshold
    
    st.markdown(f"### {title}")
    
    with st.container(border=True):
        st.markdown("#### Inference Result")
        c1, c2 = st.columns(2)
        c1.metric("Anomaly Score", f"{score:.2f}")
        if is_anomaly:
            c2.error("Prediction: **ANOMALOUS**")
        else:
            c2.success("Prediction: **STANDARD MODEL**")
            
    st.markdown("#### Reconstruction Error Heatmap")
    st.pyplot(plot_error_heatmap(jet, node_mse))
    
    display_metrics(jet)
    return score, node_mse

# ================= TABS =================
# Only build the interactive UI inside the Streamlit runtime.
# Importing this module (e.g. in tests) must stay side-effect free.
if _IN_STREAMLIT_RUNTIME:
    tab1, tab2, tab3 = st.tabs(["Analysis", "Explainability", "Benchmarks"])

    with tab1:
        if comparison_mode:
            col_bg, col_sig = st.columns(2)
            jet_bg = get_sample_jet("bg", st.session_state.seed)
            jet_sig = get_sample_jet("sig", st.session_state.seed)

            with col_bg:
                score_bg, mse_bg = render_event_column(jet_bg, "Standard Model Background")
            with col_sig:
                score_sig, mse_sig = render_event_column(jet_sig, "Higgs Boson Signal")

            active_jets = [("Background", jet_bg, mse_bg), ("Signal", jet_sig, mse_sig)]
        else:
            stype_arg = "bg" if "Background" in sample_type else "sig"
            jet = get_sample_jet(stype_arg, st.session_state.seed)
            score, mse = render_event_column(jet, sample_type)
            active_jets = [(sample_type, jet, mse)]

    with tab2:
        st.markdown("### Which particles caused the anomaly detection?")
        st.markdown("The autoencoder struggles to reconstruct particles that exhibit out-of-distribution physical interactions. The table below lists the top 5 particles with the highest reconstruction error for the current event(s).")

        for name, jet_data, mse_data in active_jets:
            if jet_data is None or mse_data is None:
                st.warning(f"No data for {name} (demo data missing).")
                continue
            st.markdown(f"#### Top 5 Anomalous Particles: {name}")

            # Sort node MSEs
            top_indices = np.argsort(mse_data)[::-1][:5]

            table_data = []
            for idx in top_indices:
                px = jet_data.x[idx, 0].item()
                py = jet_data.x[idx, 1].item() if jet_data.x.size(1) > 1 else 0.0
                pt = float(np.hypot(px, py))
                table_data.append({
                    "Particle ID": idx,
                    "Reconstruction Error (MSE)": f"{mse_data[idx]:.4f}",
                    "pT (hypot px,py)": f"{pt:.4f}",
                    "eta": f"{jet_data.x[idx, 4].item():.4f}",
                    "phi": f"{jet_data.x[idx, 5].item():.4f}",
                    "charge": f"{jet_data.x[idx, 10].item():.1f}"
                })

            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    with tab3:
        st.markdown("### Model Performance Benchmark")
        st.markdown("JetClass Anomaly Detection Benchmark Results over 6 Million events.")

        benchmark_data = {
            "Model": ["MLP (Baseline)", "GCN", "EdgeConv (1 Epoch)", "EdgeConv (5 Epochs)", "EdgeConv (50 Epochs)"],
            "AUROC": ["0.6233", "0.6541", "0.6536", "0.6628", "0.6808"]
        }
        st.table(pd.DataFrame(benchmark_data))
        st.caption("Historical reported values from the JetClass QCD-vs-non-QCD ranking benchmark (see README).")

        st.markdown("---")
        st.markdown("### Representation Learning Manifold")
        st.markdown("Precomputed t-SNE visualization of the learned latent manifold, showing partial separation between background and signal-like jet topologies.")

        tsne_path = "docs/latent_space_tsne.png"
        if os.path.exists(tsne_path):
            st.image(tsne_path, use_container_width=True)
        else:
            st.warning("t-SNE visualization not found in docs/ directory.")
