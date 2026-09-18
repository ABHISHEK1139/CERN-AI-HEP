# CERN-AI-HEP: Graph Neural Network Anomaly Detection for High Energy Physics

[![PyTorch Geometric](https://img.shields.io/badge/PyTorch%20Geometric-2.8.0-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CERN Open Data](https://img.shields.io/badge/CERN-Open%20Data-blue.svg)](https://opendata.cern.ch/)

An end-to-end unsupervised **Graph Neural Network (GNN)** research platform and anomaly detection pipeline designed to isolate rare particle collisions, exotic signatures, and Beyond the Standard Model (BSM) candidates at the **Large Hadron Collider (LHC)**. 

The pipeline reconstructs collision events as relational **3D Particle Clouds** in $(\Delta\eta, \Delta\phi, p_T)$ space, employing dynamic **EdgeConv (Dynamic Graph CNN)**, **GCN**, **GraphSAGE**, **GAT**, and **NVIDIA PhysicsNeMo** accelerated architectures.

---

## Key Highlights & Performance

- **Top JetClass AUROC**: **0.6808** (Unsupervised EdgeConv Graph Autoencoder)
- **Large-Scale Training**: Evaluated on **6 Million Jets** from the JetClass dataset
- **Real Detector Validation**: Direct ingestion and graph reconstruction on **CERN CMS Run-2 NanoAOD** ROOT data
- **Fast Convergence**: 97% of optimal AUROC (0.6628) attained within the first **5 training epochs**
- **Inference Efficiency**: Sub-2ms graph reconstruction and inference latency, optimized with GPU $k$-NN kernels
- **Interactive UI**: Real-time Streamlit research platform with 3D particle cloud visualization

---

## 3D Particle Cloud Topologies ($\Delta\eta - \Delta\phi$ Plane)

Below are representative event visualizations of collision graphs reconstructed by the pipeline. Node size scales with transverse momentum ($p_T$), while color intensity marks particle-level reconstruction error assigned by the unsupervised autoencoder.

| Standard Model Background ($Z \to \nu\nu$ / QCD) | Exotic Candidate / Anomaly ($H \to b\bar{b}$) |
|:---:|:---:|
| <img src="docs/bg_event.png" width="460" alt="Standard Model Background Event"> | <img src="docs/sig_event.png" width="460" alt="Higgs Boson Anomaly Candidate"> |
| *Diffuse, low-error particle spray characteristic of standard QCD background.* | *Concentrated high-mass sub-clusters flagged by elevated anomaly scores (red).* |

---

## System Architecture

```
                                  [ Collision Event Sources ]
                     ┌────────────────────────┬────────────────────────┐
                     ▼                        ▼                        ▼
             JetClass (100M/6M)      CMS NanoAOD (ROOT)        LHCO 2020 (HDF5)
                     │                        │                        │
                     └────────────────────────┼────────────────────────┘
                                              ▼
                                 [ Event Ingestion Engine ]
                                  • Awkward Array / Uproot
                                  • Kinematic Normalization
                                              ▼
                                 [ Graph Builder Module ]
                                  • Dynamic GPU k-NN (k=8..16)
                                  • Coordinate Embedding (Δη, Δφ, pT)
                                  • Edge Feature Extraction
                                              ▼
                             [ Anomaly Engine (Deep Models) ]
                         ┌────────────────────┼────────────────────┐
                         ▼                    ▼                    ▼
                    EdgeConv (DGCNN)         GCN / GAT        PhysicsNeMo
                         │                    │                    │
                         └────────────────────┼────────────────────┘
                                              ▼
                                 [ Graph Autoencoder Bottleneck ]
                                  • Latent Space z ~ R^D
                                  • Global / Multi-Scale Pooling
                                              ▼
                                   [ Decoders & Scoring ]
                                  • Node & Feature Reconstruction
                                  • Anomaly Metric: S(G) = MSE(x, x̂)
                                              ▼
                                  [ Downstream Evaluation ]
                             • AUROC / Precision-Recall
                             • Latent Manifolds (t-SNE / UMAP)
                             • Streamlit Interactive Dashboard
```

<p align="center">
  <img src="docs/pipeline_architecture.png" width="100%" alt="CERN-AI-HEP Pipeline Architecture">
  <br>
  <em>Figure 1: Full end-to-end relational deep learning pipeline for LHC collision events.</em>
</p>

---

## Datasets Supported

This platform features native multi-dataset loaders with automated batching, graph generation, and memory-efficient streaming.

### 1. JetClass Dataset
- **Reference**: Qu, Li, Gouskos (2022) — *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics* ([arXiv:2202.03772](https://arxiv.org/abs/2202.03772)).
- **Total Scale**: 100 Million simulated jets across 10 distinct particle decay classes.
- **Benchmark Subset**: **6 Million Jets** utilized for unsupervised pre-training and anomaly-ranking experiments.
- **Constituent Representation**: Up to 128 particles per jet cloud, parameterized by 16 kinematics and particle-ID variables:
  - **Kinematics**: $\Delta\eta$, $\Delta\phi$, $\log p_T$, $\log E$, $\log \frac{p_T}{p_{T,\text{jet}}}$, $\log \frac{E}{E_{\text{jet}}}$, $\Delta R$.
  - **Particle Identification**: One-hot indicators for charged hadrons, neutral hadrons, photons, electrons, and muons; charge and displacement metrics.
- **Anomaly Detection Protocol**: Unsupervised training solely on $1\,\text{M}$ $Z \to \nu\nu$ jets (serving as a realistic QCD background proxy). Evaluation ranks unseen Top ($t \to bqq'$), Higgs ($H \to b\bar{b}, c\bar{c}, gg, 4q, \ell\nu qq'$), and Vector Boson ($W \to qq', Z \to q\bar{q}$) decay jets as anomaly signals.

### 2. CERN CMS Open Data (Run-2 NanoAOD)
- **Source**: [CERN Open Data Portal](https://opendata.cern.ch/) (Records e.g. `12353` for `DYJetsToLL.root`, `GluGluToHToTauTau.root`).
- **Format**: Production CERN NanoAOD ROOT tree format.
- **Parsing**: Direct C++-free ingestion using `uproot` and `awkward`.
- **Reconstruction**: Extracts reconstructed physics objects (`Muon`, `Electron`, `Jet_pt`, `Jet_eta`, `Jet_phi`, `Jet_mass`, and missing transverse energy `MET_pt`).
- **Purpose**: Real detector cross-validation ensuring algorithms remain resilient to experimental detector noise, pileup, and real coordinate distortions.

### 3. LHC Olympics 2020 (LHCO R&D Dataset)
- **Reference**: Kasieczka et al. (2021) — *The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics* ([arXiv:2101.08320](https://arxiv.org/abs/2101.08320)).
- **Format**: HDF5 event tables containing dijet collision events.
- **Signal**: Resonant BSM dijet production $W' \to X Y \to (q\bar{q})(q\bar{q})$.
- **Background**: Standard Model QCD multijet background.
- **Purpose**: Community standard benchmark for unsupervised signal hunting and resonant bump hunting.

### 4. Synthetic LHC Collision Generator
- Built-in Monte-Carlo toy generator (`event_ingestion/generator.py`) modeling relativistic kinematics, decay branching, and detector resolution effects.
- Enables instant offline development, CI/CD testing, and smoke tests without requiring gigabyte-scale external downloads.

---

## Experimental Results & Benchmark Comparisons

### Model Architecture Comparison (JetClass 6M Benchmark)

| Model Architecture | Graph Formulation | Trainable Parameters | Best AUROC | Max Inference Throughput |
|:---|:---|:---:|:---:|:---:|
| **MLP Baseline** | Global Flat Features | 6.3 k | 0.6233 | 42,000 jets/sec |
| **GCN Autoencoder** | Static $k$-NN Graph ($k=8$) | 37 k | 0.6541 | 18,500 jets/sec |
| **EdgeConv (1 Epoch)** | Dynamic Feature $k$-NN | 37 k | 0.6536 | 12,200 jets/sec |
| **EdgeConv (5 Epochs)** | Dynamic Feature $k$-NN | 37 k | 0.6628 | 12,200 jets/sec |
| **EdgeConv (50 Epochs)** | Dynamic Feature $k$-NN | 37 k | **0.6808** | 12,200 jets/sec |
| **PhysicsNeMo Fallback** | MeshGraphNet Hybrid | 52 k | 0.6714 | 19,800 jets/sec |

### Rapid Convergence & Saturation Dynamics

Training reaches **97.4%** of its asymptotic AUROC within the first 5 epochs:
- **Epoch 1**: 0.6536 AUROC
- **Epoch 5**: 0.6628 AUROC
- **Epoch 50**: 0.6808 AUROC (an incremental +0.018 improvement over 45 additional epochs)

<div align="center">
  <img src="docs/loss_curve.png" width="700" alt="Training Loss Curve">
  <p><em>Figure 2: Empirical loss convergence curve showing rapid descent and stabilization.</em></p>
</div>

---

## Visualizations & Comprehensive Benchmark Evaluation

### 1. Global Model Performance Curves

| ROC Curve (Overall) | Precision-Recall Curve |
|:---:|:---:|
| <img src="docs/roc_curve.png" width="460" alt="Overall ROC Curve"> | <img src="docs/pr_curve.png" width="460" alt="Precision-Recall Curve"> |
| *Receiver Operating Characteristic across anomaly thresholds.* | *PR performance under heavy Standard Model class imbalance.* |

| Anomaly Score Separation | Latent Manifold Projection (t-SNE) |
|:---:|:---:|
| <img src="docs/anomaly_distribution.png" width="460" alt="Anomaly Score Distribution"> | <img src="docs/latent_space_tsne.png" width="460" alt="Latent Space t-SNE"> |
| *Reconstruction error histogram separating background from anomalies.* | *2D t-SNE projection of the bottleneck latent representation.* |

---

### 2. JetClass Specific Benchmarks

| JetClass EdgeConv ROC | JetClass EdgeConv Anomaly Scores |
|:---:|:---:|
| <img src="docs/jetclass_edgeconv_roc.png" width="460" alt="JetClass EdgeConv ROC"> | <img src="docs/jetclass_edgeconv_scores.png" width="460" alt="JetClass EdgeConv Scores"> |
| *EdgeConv ROC curve reaching 0.6808 AUROC on JetClass.* | *Reconstruction error distribution for 6M JetClass test events.* |

| JetClass Multi-Class ROC | JetClass Multi-Class Anomaly Scores |
|:---:|:---:|
| <img src="docs/jetclass_roc.png" width="460" alt="JetClass Comparative ROC"> | <img src="docs/jetclass_scores.png" width="460" alt="JetClass Score Distribution"> |
| *Comparative AUROC across individual Higgs, Top, and W/Z signal channels.* | *Reconstruction error separation per decay topology.* |

---

### 3. CMS Open Data & LHC Olympics Validation

| CMS Open Data ROC Curve | CMS Anomaly Score Distribution |
|:---:|:---:|
| <img src="docs/cms_roc.png" width="460" alt="CMS Open Data ROC Curve"> | <img src="docs/cms_scores.png" width="460" alt="CMS Score Distribution"> |
| *Validation on authentic CERN CMS NanoAOD collision events.* | *Anomaly score distribution on real detector reconstructed jets.* |

| CMS Detector Validation Analysis | CMS NanoAOD Event Graph |
|:---:|:---:|
| <img src="docs/cms_validation.png" width="460" alt="CMS Validation Analysis"> | <img src="docs/event_graph.png" width="460" alt="CMS Event Graph"> |
| *Kinematic correlation between anomaly score and jet transverse momentum.* | *Reconstructed event graph connecting CMS detector physics objects.* |

| LHCO 2020 Challenge ROC | LHCO Anomaly Score Distribution |
|:---:|:---:|
| <img src="docs/lhco_roc.png" width="460" alt="LHCO ROC Curve"> | <img src="docs/lhco_scores.png" width="460" alt="LHCO Anomaly Scores"> |
| *ROC curve on the LHC Olympics 2020 resonant dijet dataset.* | *Score distribution comparing QCD dijets against resonant W' candidates.* |

---

### 4. Latent Space Representation

<div align="center">
  <img src="docs/latent_space.png" width="700" alt="Latent Space Manifold">
  <p><em>Figure 3: Unsupervised latent embedding space learned by the Graph Autoencoder.</em></p>
</div>

---

## NVIDIA PhysicsNeMo Acceleration

A specialized integration layer (`physicsnemo_integration/`) bridges PyTorch Geometric with NVIDIA PhysicsNeMo acceleration primitives. When native CUDA kernels are unavailable, it seamlessly falls back to a high-throughput PyG MeshGraphNet module.

| Implementation Pipeline | Forward Latency per Batch | Speedup Ratio |
|:---|:---:|:---:|
| Standard PyG Message Passing | 2.79 ms | 1.00x |
| **PhysicsNeMo Hybrid Optimization** | **1.73 ms** | **1.62x** |

---

## Interactive Research UI (Streamlit)

Launch an interactive research dashboard with 3D particle cloud visualization, live anomaly scoring, and graph topological analysis:

```bash
streamlit run demo.py
```

**Features**:
- Interactive 3D particle visualization in $(\Delta\eta, \Delta\phi, p_T)$ space.
- Particle-level anomaly highlighting (red nodes signify high reconstruction residuals).
- Real-time inference across user-selected event files (JetClass, CMS NanoAOD, synthetic).
- Inspection of latent space embeddings and relational adjacency matrices.

---

## Repository Structure

```text
cern-ai/
│
├── anomaly_engine/              # Model architectures, training loops, and evaluation
│   ├── models/
│   │   ├── autoencoder.py       # Graph Autoencoder base architecture
│   │   ├── edge_conv.py         # Dynamic Graph CNN (EdgeConv) module
│   │   ├── baselines.py         # GCN, GAT, GraphSAGE, CNN, and MLP baselines
│   │   └── losses.py            # Chamfer distance, MSE, and anomaly loss metrics
│   ├── trainer.py               # Robust multi-epoch trainer with resume logic
│   └── evaluate.py              # Evaluator (ROC, PR, t-SNE, score distributions)
│
├── graph_builder/               # Point-cloud to relational graph construction
│   ├── dataset.py               # PyG InMemoryDataset with train/val/test splits
│   ├── jetclass_dataset.py      # JetClass ROOT/Awkward array ingestion
│   ├── cms_dataset.py           # CERN CMS NanoAOD ingestion and graph creation
│   ├── graph_constructor.py     # EventGraphConstructor (k-NN, ΔR, fully connected)
│   └── features.py              # Physics feature extractors and normalizers
│
├── event_ingestion/             # Data loading and synthetic collision simulation
│   ├── loader.py                # Multi-format event loader (ROOT, HDF5, NPZ)
│   ├── generator.py             # Relativistic particle collision generator
│   └── statistics.py            # Event kinematic distributions and summary stats
│
├── physicsnemo_integration/     # NVIDIA PhysicsNeMo integration and fallbacks
│   ├── wrapper.py               # PhysicsNeMo hybrid wrapper
│   └── meshgraphnet.py          # MeshGraphNet architecture implementation
│
├── experiments/                 # Production training and evaluation scripts
│   ├── run_6m_ablation.py       # 6-Million JetClass large-scale ablation study
│   ├── run_5epochs_edgeconv.py  # Rapid EdgeConv convergence experiment
│   ├── phase1_4.py              # CMS NanoAOD end-to-end extraction pipeline
│   └── train_classifier.py      # Supervised and unsupervised training entry point
│
├── scripts/                     # Data acquisition and utility scripts
│   ├── download_robust.py       # Resilient CERN Open Data downloader
│   └── download_jetclass.py     # Multi-threaded JetClass download utility
│
├── docs/                        # Reference benchmark plots, figures, and diagrams
├── checkpoints/                 # Trained model weights and checkpoint files
├── demo.py                      # Interactive Streamlit 3D visualization dashboard
├── rebuild_loss_curve.py        # Utility to reconstruct convergence plots from logs
└── requirements.txt             # Project dependencies and environment specification
```

---

## Installation & Quickstart

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/ABHISHEK1139/CERN-AI-HEP.git
cd CERN-AI-HEP

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Run Smoke Test (CPU or GPU)

Verify that data ingestion, graph construction, training, and evaluation function seamlessly using the synthetic configuration:

```bash
python experiments/train_classifier.py --config experiments/configs/smoke.yaml --model gcn --epochs 2 --device cpu
```

### 3. Run the Automated Test Suite

Run the full pytest suite (including all regression tests):

```bash
python -m pytest
```

### 4. Re-Running the 6M JetClass Ablation Experiment

To execute the large-scale 6-million jet training pipeline on a CUDA-enabled GPU:

```bash
python experiments/run_6m_ablation.py
```

### 5. Evaluating CMS Open Data

To process and visualize real CMS NanoAOD detector events:

```bash
python experiments/phase1_4.py
```

---

## Hardware Specifications

| Hardware Component | Specification |
|:---|:---|
| **Host Processor** | Intel Core i5 (12th Generation) |
| **Graphics Processing Unit** | NVIDIA GeForce RTX 3050 (4 GB VRAM) |
| **System Memory (RAM)** | 16 GB DDR4 |
| **Storage** | NVMe SSD (High I/O for Awkward / ROOT reading) |
| **JetClass 6M Training Time** | ~45 hours total |

---

## Citation & References

If you use this codebase or research in your work, please cite:

```bibtex
@software{cern_ai_hep_2026,
  author = {Abhishek Kumar},
  title = {CERN-AI-HEP: Graph Neural Network Anomaly Detection for High Energy Physics},
  url = {https://github.com/ABHISHEK1139/CERN-AI-HEP},
  year = {2026}
}
```

Key reference datasets:
- **JetClass**: Qu, H., Li, C., & Gouskos, L. (2022). *JetClass: A Large-Scale Dataset for Deep Learning in Jet Physics*, [arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
- **LHC Olympics 2020**: Kasieczka, G., et al. (2021). *The LHC Olympics 2020: A Community Challenge for Anomaly Detection in High Energy Physics*, [arXiv:2101.08320](https://arxiv.org/abs/2101.08320).
- **CMS Open Data**: CERN Open Data Portal, *CMS Collaboration NanoAOD Samples*, [opendata.cern.ch](https://opendata.cern.ch/).
