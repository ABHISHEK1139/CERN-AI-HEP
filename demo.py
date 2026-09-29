import os
import sys
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from graph_builder.jetclass_dataset import JetClassDataset
except ImportError:  # optional heavy dep (uproot/awkward) may be absent
    JetClassDataset = None
from torch_geometric.data import Batch

from anomaly_engine.checkpoint import load_state_dict
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from anomaly_engine.models.edge_conv import EdgeConvEncoder

DEFAULT_CKPT = "checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"

# JetClass 16-feature layout: px=0, py=1, pz=2, energy=3, deta=4, dphi=5,
# ..., charge=10.
JET_PX, JET_PY, JET_ENERGY, JET_DETA, JET_DPHI, JET_CHARGE = 0, 1, 3, 4, 5, 10
JETCLASS_INPUT_DIM = 16


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
def _build_model(input_dim=JETCLASS_INPUT_DIM, hidden_dim=64, latent_dim=32):
    """Instantiate the demo architecture. Shape-only, no weights loaded."""
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim,
                              latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim,
                           output_dim=input_dim)
    return GraphAutoencoder(encoder=encoder, decoder=decoder)


def _load_model_uncached(ckpt_path=DEFAULT_CKPT):
    """Load the trained EdgeConv autoencoder.

    Raises:
        FileNotFoundError: If the checkpoint is missing. The previous version
            silently fell through to a randomly initialised model and then
            displayed its reconstruction errors as "Anomaly Score", which is
            scientifically meaningless output presented with full confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model().to(device)

    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train it first with\n"
            f"    python experiments/train_jetclass.py --arch edgeconv --large\n"
            f"or pass --checkpoint. Refusing to score jets with an untrained "
            f"(randomly initialised) model."
        )
    load_state_dict(path, model, device=device)
    return model, device


if _IN_STREAMLIT_RUNTIME:
    load_model = st.cache_resource(_load_model_uncached)
else:
    load_model = _load_model_uncached


def count_parameters(model) -> int:
    """Number of trainable parameters (the sidebar used a stale hardcoded 37,296)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        _n_params = count_parameters(_build_model())
        st.markdown(f"""
        **Architecture:** EdgeConv Graph Autoencoder
        **Parameters:** {_n_params:,}
        **Graph Construction:** k-NN (k=8)
        **Training Dataset:** JetClass (SM Background)
        **Training Jets:** 6,000,000
        **Validation Jets:** 1,000,000
        **Epochs:** 50
        **Best AUROC:** 0.6808
        """)
        st.caption(
            "Jet counts and AUROC are the historical values reported in the "
            "README for the published run. Parameter count is computed from "
            "the live architecture."
        )

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
        # local Generator: np.random.randint reseeded the global legacy state.
        st.session_state.seed = int(np.random.default_rng().integers(0, 100000))
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
        if not val_bg_files:
            return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_bg_files[0]], k_neighbors=8, sample_size=50, tag="demo_bg")
    else:
        if not val_sig_files:
            return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_sig_files[0]], k_neighbors=8, sample_size=50, tag="demo_sig")

    if len(ds) == 0:
        return None
    # Local RNG: the previous code called random.seed(), which reseeded the
    # global module and perturbed every other stochastic call in the app.
    rng = np.random.default_rng(int(seed))
    return ds[int(rng.integers(0, len(ds)))]


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

def jet_observables(jet):
    """Physics summary for one JetClass jet.

    In JetClass, node 0 *is* the jet and nodes 1..N-1 are its constituents, so
    the jet's own four-momentum is node 0's ``(px, py)``. The previous code
    labelled ``sum(hypot(px_i, py_i))`` over all constituents as "Total Jet
    pT", which is a sum of magnitudes, not a vector sum — it is systematically
    larger than the true jet pT and is not a collider observable at all.

    Returns:
        dict with ``jet_pt``, ``sum_constituent_pt``, ``n_constituents`` and
        ``avg_charge``.
    """
    x = jet.x
    n = int(x.size(0))
    if x.size(1) < 6:
        raise ValueError(
            f"Expected JetClass's 16-feature layout, got {x.size(1)} columns."
        )

    jet_pt = float(torch.hypot(x[0, JET_PX], x[0, JET_PY]).item())
    constituents = x[1:] if n > 1 else x
    sum_constituent_pt = float(
        torch.hypot(constituents[:, JET_PX], constituents[:, JET_PY]).sum().item()
    )
    charge = constituents[:, JET_CHARGE] if x.size(1) > JET_CHARGE else None
    return {
        "jet_pt": jet_pt,
        "sum_constituent_pt": sum_constituent_pt,
        "n_constituents": int(constituents.size(0)),
        "avg_charge": float(charge.mean().item()) if charge is not None else 0.0,
    }


def plot_error_heatmap(jet, node_mse):
    if getattr(jet, "edge_index", None) is None or jet.edge_index.numel() == 0:
        raise ValueError("plot_error_heatmap needs a jet with edges.")
    G = nx.Graph()
    edge_index = jet.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))

    n_nodes = int(jet.x.size(0))
    # Use actual physics coordinates (eta, phi) for the node layout
    pos = {i: (float(jet.x[i, JET_DETA]), float(jet.x[i, JET_DPHI])) for i in range(n_nodes)}

    # Scoped style: never leak dark_background into other figures.
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')

        # NetworkX colours nodes of G, not all n_nodes. Isolated nodes (no
        # edges) are absent from G, so pass exactly the nodes being drawn -
        # otherwise a length mismatch between node_mse and len(G) is ambiguous.
        draw_order = sorted(G.nodes())
        color_values = [float(np.asarray(node_mse)[i]) for i in draw_order]

        sc = nx.draw_networkx_nodes(
            G, pos, nodelist=draw_order, node_size=30, node_color=color_values,
            cmap=plt.cm.coolwarm, alpha=0.9, ax=ax, linewidths=0.5,
            edgecolors='white',
        )
        nx.draw_networkx_edges(G, pos, edge_color='#666666', alpha=0.4, ax=ax)

        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("Reconstruction MSE", fontsize=8, color='lightgray')
        cbar.ax.tick_params(labelsize=7, colors='lightgray')

        # Format axes to look like a physics plot
        ax.set_xlabel("Δη (Pseudo-rapidity)", fontsize=8, color='lightgray')
        ax.set_ylabel("Δφ (Azimuthal)", fontsize=8, color='lightgray')
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
    obs = jet_observables(jet)
    n_const = obs["n_constituents"]

    edge_index = jet.edge_index.cpu().numpy()
    G = nx.Graph()
    for i in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))
    n_edges = G.number_of_edges()
    avg_degree = (n_edges * 2) / n_const if n_const > 0 else 0
    density = (2 * n_edges) / (n_const * (n_const - 1)) if n_const > 1 else 0
    components = nx.number_connected_components(G)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Physics")
        st.markdown("---")
        st.metric("Jet pT (GeV)", f"{obs['jet_pt']:.2f}")
        st.metric("Σ Constituent pT (GeV)", f"{obs['sum_constituent_pt']:.2f}")
        st.metric("Constituents", f"{n_const}")
        st.metric("Avg Charge", f"{obs['avg_charge']:.2f}")
    with c2:
        st.markdown("##### Topology")
        st.markdown("---")
        st.metric("Nodes (incl. jet)", f"{int(jet.x.size(0))}")
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
            mse_arr = np.asarray(mse_data).flatten()
            top_indices = np.argsort(mse_arr)[::-1][:5]

            table_data = []
            for idx in top_indices:
                if idx >= jet_data.x.size(0):
                    continue
                px = jet_data.x[idx, JET_PX].item()
                py = jet_data.x[idx, JET_PY].item()
                pt = float(np.hypot(px, py))
                has_charge = jet_data.x.size(1) > JET_CHARGE
                table_data.append({
                    "Particle ID": int(idx),
                    "Reconstruction Error (MSE)": f"{mse_arr[idx]:.4f}",
                    "pT (hypot px,py)": f"{pt:.4f}",
                    "eta": f"{jet_data.x[idx, JET_DETA].item():.4f}",
                    "phi": f"{jet_data.x[idx, JET_DPHI].item():.4f}",
                    "charge": f"{jet_data.x[idx, JET_CHARGE].item():.1f}" if has_charge else "n/a",
                })

            if table_data:
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
