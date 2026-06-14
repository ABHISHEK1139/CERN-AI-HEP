import streamlit as st
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from glob import glob

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder.jetclass_dataset import JetClassDataset
from anomaly_engine.models.edge_conv import EdgeConvEncoder
from anomaly_engine.models.autoencoder import GraphAutoencoder, GraphDecoder
from torch_geometric.data import Batch

# Set page config
st.set_page_config(page_title="CERN AI: Anomaly Detection Dashboard", layout="wide")

st.title("🧠 CERN AI: GNN Anomaly Detection in HEP Collision Events")
st.markdown("""
This interactive dashboard demonstrates the unsupervised anomaly detection pipeline on **3D Particle Clouds** representing collision jets at the LHC. 
It uses a pre-trained **EdgeConv Graph Autoencoder** trained exclusively on Standard Model backgrounds ($Z \rightarrow \nu\nu$).
""")

# Load Model
@st.cache_resource
def load_model(ckpt_path="checkpoints/jetclass_autoencoder/jetclass_edgeconv_best.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim, hidden_dim, latent_dim = 16, 64, 32
    encoder = EdgeConvEncoder(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=3)
    decoder = GraphDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim)
    model = GraphAutoencoder(encoder=encoder, decoder=decoder).to(device)
    
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device

model, device = load_model()

# Sidebar: Model Card
st.sidebar.markdown("## ⚙️ Model Card")
st.sidebar.markdown("""
**Model:** EdgeConv Autoencoder  
**Parameters:** 37,296  
**Training Dataset:** JetClass 6M  
**Best AUROC:** 0.6808  
""")
st.sidebar.markdown("---")

# Select Event Type
st.sidebar.header("Select Physics Sample")
sample_type = st.sidebar.selectbox(
    "Choose Jet Origin Type",
    ["Standard Model Background (Z \u2192 \u03BD\u03BD)", "Higgs Boson Signal (Anomaly)"]
)

# Load Sample Data
@st.cache_data
def get_sample_jet(sample_type):
    # Load from val files
    val_bg_files = sorted(glob("data/jetclass/val_5M/ZJetsToNuNu_*.root"))
    val_sig_files = sorted(glob("data/jetclass/val_5M/HTo*.root"))
    
    if sample_type == "Standard Model Background (Z \u2192 \u03BD\u03BD)":
        if not val_bg_files:
            return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_bg_files[0]], k_neighbors=8, sample_size=50, tag="demo_bg")
    else:
        if not val_sig_files:
            return None
        ds = JetClassDataset(root="data/jetclass/graphs_demo", root_file_paths=[val_sig_files[0]], k_neighbors=8, sample_size=50, tag="demo_sig")
    
    # Return a random jet from dataset
    import random
    random.seed(42) # fix seed for reproducibility
    idx = random.randint(0, len(ds) - 1)
    return ds[idx]

jet = get_sample_jet(sample_type)

if jet is None:
    st.error("No raw validation datasets found! Run experiments/produce_evidence.py to prepare the workspace data first.")
else:
    # Run Inference
    batch = Batch.from_data_list([jet]).to(device)
    with torch.no_grad():
        res = model(batch)
        score = res['per_graph_loss'].item()
        
    threshold = 235.0
    is_anomaly = score > threshold

    # PROMINENT ANOMALY SCORE DISPLAY
    st.markdown("---")
    st.subheader("⚡ Real-time Event Classification")
    
    col_score, col_verdict = st.columns(2)
    with col_score:
        st.metric(label="Anomaly Score", value=f"{score:.2f}")
    with col_verdict:
        if is_anomaly:
            st.error("🚨 Classification: **Anomalous**")
        else:
            st.success("✅ Classification: **Standard Model**")
    st.markdown("---")

    # Construct details layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Particle Cloud Statistics")
        n_particles = jet.x.size(0)
        leading_pt = jet.x[:, 0].max().item()
        
        st.metric("Constituent Particles", f"{n_particles}")
        st.metric("Leading Particle pT (normalized)", f"{leading_pt:.2f}")
        
        # Display Feature Table
        st.write("First 5 particles features:")
        feats_df = {
            "pT": jet.x[:5, 0].numpy(),
            "eta": jet.x[:5, 4].numpy(),
            "phi": jet.x[:5, 5].numpy(),
            "charge": jet.x[:5, 10].numpy(),
        }
        st.dataframe(feats_df)
        
    with col2:
        st.subheader("🕸 Graph Representation (k-NN, k=8)")
        G = nx.Graph()
        edge_index = jet.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            G.add_edge(u, v)
            
        fig, ax = plt.subplots(figsize=(6, 5))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, node_size=30, node_color='#1e88e5', edge_color='#cccccc', alpha=0.8, ax=ax)
        st.pyplot(fig)
