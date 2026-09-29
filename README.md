# CERN-AI-HEP: Graph Neural Network Anomaly Detection for High Energy Physics

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch--Geometric-2.4%2B-orange.svg)](https://pytorch-geometric.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/ABHISHEK1139/CERN-AI-HEP/actions/workflows/ci.yml/badge.svg)](https://github.com/ABHISHEK1139/CERN-AI-HEP/actions/workflows/ci.yml)
[![CERN Open Data](https://img.shields.io/badge/CERN-Open%20Data-blue.svg)](https://opendata.cern.ch/)

An end-to-end **unsupervised Graph Neural Network (GNN)** research platform for
anomaly detection on LHC collision events.

The pipeline reconstructs each jet or event as a relational **point cloud**: its
constituents are nodes, and edges are the $k$ nearest neighbours in the
$(\Delta\eta, \Delta\phi)$ plane, with $p_T$ carried as a node feature. On top
of that graph sit **EdgeConv (DGCNN)**, **GCN**, **GraphSAGE** and **GAT**
encoders, plus an **NVIDIA PhysicsNeMo** bridge.

> **What this measures.** The headline benchmark is a *QCD-vs-non-QCD ranking*
> task: the autoencoder is trained on Standard Model QCD jets only, then asked
> to rank known SM classes (Higgs, top, W/Z) above them. It is a proxy for
> anomaly detection, **not** a BSM discovery search. See
> [Scope of the benchmark](#scope-of-the-benchmark).

---

## Key Highlights

- **Top JetClass AUROC**: **0.6808** (Unsupervised EdgeConv Graph Autoencoder)
- **Large-Scale Training**: Evaluated on **6 Million Jets** from the JetClass dataset
- **Real Detector Validation**: Direct ingestion and graph reconstruction on **CERN CMS Run-2 NanoAOD** ROOT data
- **Fast Convergence**: 97.4% of the 50-epoch AUROC reached within the first **5 epochs**
- **Efficiency**: ~1.7–2.8 ms per batch of 256 graphs, GPU $k$-NN graph construction
- **Interactive UI**: Streamlit research dashboard with per-particle error attribution
- **Self-contained test suite**: 402 tests, ~84% coverage, no dataset download required

---

## Point-Cloud Topologies ($\Delta\eta - \Delta\phi$ Plane)

Below are representative jet visualizations produced by the pipeline. The graph
is built in the $(\Delta\eta, \Delta\phi)$ plane; $p_T$ enters as a node feature
and is shown through node size, while colour intensity marks the per-particle
reconstruction error assigned by the unsupervised autoencoder.

| Standard Model Background ($Z \to \nu\nu$ / QCD) | Non-QCD Jets (Higgs $H \to b\bar{b}$) |
|:---:|:---:|
| <img src="docs/bg_event.png" width="460" alt="Standard Model Background Jet"> | <img src="docs/sig_event.png" width="460" alt="Non-QCD Jet"> |
| *Diffuse, low-error particle spray characteristic of QCD background.* | *Tighter substructure with elevated reconstruction error (red).* |

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
                                   • k-NN in (Δη, Δφ)  (k = 8)
                                   • GPU batched k-NN for 6M-scale runs
                                   • Edge Features: [ΔR, Δη, Δφ, ΔpT]
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
- Built-in Monte-Carlo toy generator (`event_ingestion/synthetic.py`) modeling relativistic kinematics,
  decay branching, and detector resolution effects.
- Enables instant offline development, CI/CD testing, and smoke tests without requiring gigabyte-scale external downloads.

---

## Experimental Results & Benchmark Comparisons

### Model Architecture Comparison (JetClass 6M Benchmark)

| Model Architecture | Graph Formulation | Trainable Parameters | Best AUROC | Max Inference Throughput |
|:---|:---|:---:|:---:|:---:|
| **MLP Baseline** | Global Flat Features | 8.0 k | 0.6233 | 42,000 jets/sec |
| **GCN Autoencoder** | Static $k$-NN Graph ($k=8$) | 15.2 k | 0.6541 | 18,500 jets/sec |
| **EdgeConv (1 Epoch)** | Dynamic Feature $k$-NN | 37.3 k | 0.6536 | 12,200 jets/sec |
| **EdgeConv (5 Epochs)** | Dynamic Feature $k$-NN | 37.3 k | 0.6628 | 12,200 jets/sec |
| **EdgeConv (50 Epochs)** | Dynamic Feature $k$-NN | 37.3 k | **0.6808** | 12,200 jets/sec |
| **PhysicsNeMo Fallback** | MeshGraphNet Hybrid | 52 k | 0.6714 | 19,800 jets/sec |

<a id="scope-of-the-benchmark"></a>
### Scope of the Benchmark

This is a **QCD-vs-non-QCD ranking** task. The autoencoder is trained on
Standard Model QCD jets only, then evaluated on whether it ranks known SM
classes (Higgs, top, $W/Z$) separately from QCD. Those "signal" classes are
themselves Standard Model, so the AUROC values above measure
**"separating non-QCD jets from QCD jets"** — they are *not* a new-physics
sensitivity, and a BSM search would additionally require a signal
distribution that is genuinely absent from training.

Reported AUROC is always the **raw, un-flipped** value. Plots may use
direction-corrected scores when the model happens to score the classes
inversely, and those figures are labelled as such.

Parameter counts below are for the 16-feature JetClass input with
`hidden_dim=64`, `latent_dim=32`, `num_layers=3`, and are reproducible with:

```python
from anomaly_engine.models import get_autoencoder, get_classifier
get_classifier("mlp", input_dim=16, hidden_dim=64)
get_autoencoder("gcn", input_dim=16, hidden_dim=64, latent_dim=32, num_layers=3)
get_autoencoder("edgeconv", input_dim=16, hidden_dim=64, latent_dim=32, num_layers=3)
```

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

Launch an interactive research dashboard for per-particle anomaly attribution:

```bash
# Requires the JetClass validation ROOT files under data/jetclass/val_5M/
streamlit run demo.py
```

The dashboard has three tabs:

| Tab | What it does |
|:--|:--|
| **Analysis** | Scores a randomly sampled jet with the pre-trained EdgeConv autoencoder, renders the $(\Delta\eta, \Delta\phi)$ reconstruction-error heatmap, and reports graph topology metrics (nodes, edges, density, connected components, average degree). |
| **Explainability** | Ranks the top-5 particles by reconstruction error for the current jet, with per-particle $p_T$, $\eta$, $\phi$ and charge. |
| **Benchmarks** | Static table of the published JetClass AUROC values plus a precomputed t-SNE figure. |

**Notes**

- It reads **JetClass** validation data only (`ZJetsToNuNu_*.root` for
  background, `HTo*.root` for signal). CMS NanoAOD and LHCO are handled by the
  training/evaluation scripts, not the dashboard.
- The "Generate New Collision Event(s)" button re-samples the jet; the sidebar
  threshold is **illustrative and not calibrated**. For research decisions use
  `AnomalyScorer.select_threshold` on held-out validation scores.
- If the checkpoint is missing the app refuses to start rather than scoring
  jets with a randomly initialised model.

---

## Repository Structure

```text
cern-ai-hep/
│
├── anomaly_engine/              # Models, training, scoring, evaluation
│   ├── models/
│   │   ├── __init__.py          # Model registry (get_classifier/get_encoder/get_autoencoder)
│   │   ├── autoencoder.py       # GraphAutoencoder + GraphDecoder
│   │   ├── edge_conv.py         # EdgeConv (DGCNN) encoder
│   │   ├── gcn.py               # GCNConv encoder + classifier
│   │   ├── graphsage.py         # GraphSAGE encoder + classifier
│   │   ├── gat.py               # GAT (multi-head) encoder + classifier
│   │   ├── baselines.py         # MLP and 1D-CNN (non-graph) baselines
│   │   └── norm.py              # SafeBatchNorm1d (tolerates size-1 batches)
│   ├── trainer.py               # Trainer: classification + autoencoder, resume, MLflow
│   ├── anomaly_scorer.py        # Anomaly scoring, ranking, thresholds, reports
│   ├── evaluate.py              # Metrics and plots (ROC, PR, t-SNE/UMAP, curves)
│   └── checkpoint.py            # Strict checkpoint loading
│
├── graph_builder/               # Point-cloud → relational graph construction
│   ├── dataset.py               # CollisionEventDataset (generic PyG InMemoryDataset)
│   ├── splitting.py             # Shared split/index_select/get_loaders mixin
│   ├── jetclass_dataset.py      # JetClass ROOT/Awkward ingestion
│   ├── jetclass_iterable.py     # Streaming JetClass IterableDataset (100M scale)
│   ├── cms_dataset.py           # CERN CMS NanoAOD ingestion and graph creation
│   ├── lhco_dataset.py          # LHCO 2020 HDF5 ingestion
│   ├── graph_constructor.py     # EventGraphConstructor (k-NN, ΔR, fully connected)
│   └── features.py              # Physics feature extractors and normalizers
│
├── event_ingestion/             # Data loading and synthetic collision simulation
│   ├── config.py                # EventConfig, particle branches, feature dims
│   ├── loader.py                # EventLoader (ROOT, NPZ)
│   ├── synthetic.py             # SyntheticEventGenerator (relativistic toy MC)
│   ├── statistics.py            # Event kinematic distributions and summary stats
│   └── downloader.py            # CMS Open Data downloader
│
├── physicsnemo_integration/     # NVIDIA PhysicsNeMo integration with fallback
│   ├── wrapper.py               # PhysicsNeMoWrapper + MeshGraphNetLayer
│   └── benchmark.py             # Cross-architecture benchmark harness
│
├── experiments/                 # Training and evaluation entry points
│   ├── data_pipeline.py         # Shared synthetic → graphs → loaders preparation
│   ├── configs/                 # default.yaml, smoke.yaml
│   ├── train_classifier.py      # Supervised signal/background training
│   ├── run_benchmark.py         # Multi-architecture comparison
│   ├── train_jetclass.py        # JetClass autoencoder (--large for streaming)
│   ├── train_cms.py             # CMS NanoAOD autoencoder
│   ├── train_autoencoder.py     # LHCO autoencoder
│   ├── run_6m_ablation.py       # 6-Million JetClass large-scale ablation
│   ├── run_5epochs_edgeconv.py  # Rapid EdgeConv convergence experiment
│   ├── produce_evidence.py      # Regenerates the figures in docs/
│   └── physics_analysis.py      # Score vs. jet kinematics analysis
│
├── scripts/                     # Data acquisition and plotting utilities
│   ├── download_robust.py       # Resilient CERN Open Data downloader
│   ├── draw_architecture.py     # Regenerates docs/ architecture diagrams
│   ├── generate_readme_images.py# Regenerates the bg/sig point-cloud figures
│   └── *.ps1                    # Windows download helpers
│
├── tests/                       # pytest suite (402 tests, ~84% coverage)
├── docs/                        # Reference benchmark plots, figures, and diagrams
├── checkpoints/                 # Published reference model weights
├── demo.py                      # Interactive Streamlit dashboard
├── rebuild_loss_curve.py        # Rebuilds convergence plots from logs/checkpoints
├── pyproject.toml               # Packaging, dependency groups, ruff + pytest + coverage config
├── Makefile                     # Common workflows (make help)
├── Dockerfile                   # CUDA runtime image
└── LICENSE                      # MIT
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

# Install the project with its development tooling (ruff, pytest, coverage)
pip install -e ".[dev]"

# Optional: CPU-only torch (useful on CI runners)
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dependency groups are declared in `pyproject.toml`:

| Extra | Contents |
|:--|:--|
| *(default)* | torch, PyG, uproot, awkward, h5py, scikit-learn, matplotlib, streamlit, mlflow, ... |
| `dev` | pytest, pytest-cov, ruff |
| `docs` | markdown-pdf, PyMuPDF |
| `accelerators` | torch-scatter / torch-sparse (optional; need a matching C++/CUDA build) |

Every model runs **without** `torch-scatter`/`torch-sparse` — PyG falls back to
native torch ops. Install them only if you want the speedup.

### 2. Run Smoke Test (CPU or GPU)

Verify that data ingestion, graph construction, training, and evaluation function seamlessly using the synthetic configuration. No downloads required:

```bash
python experiments/train_classifier.py --config experiments/configs/smoke.yaml --model gcn --epochs 2 --device cpu
```

Or run everything the CI runs:

```bash
make smoke
```

### 3. Run the Automated Test Suite

The suite is fully self-contained: the CMS/JetClass/LHCO processors are
exercised against ROOT and HDF5 files written on the fly with `uproot` and
`pandas`, so no dataset download is needed.

```bash
python -m pytest              # 402 tests
python -m pytest --cov        # with coverage (~84%)
python -m pytest -m "not slow"
make lint                     # ruff
```

### 4. Re-Running the 6M JetClass Ablation Experiment

To execute the large-scale 6-million jet training pipeline on a CUDA-enabled GPU
(requires the pre-processed chunks from `experiments/preprocess_6m.py`):

```bash
python experiments/run_6m_ablation.py
```

### 5. Evaluating CMS Open Data

To process and visualise real CMS NanoAOD detector events:

```bash
python experiments/train_cms.py      # train the CMS autoencoder
python experiments/phase1_4.py       # visualise a real NanoAOD event graph
```

### Docker

```bash
docker build -t cern-ai-hep .
docker run -p 8501:8501 cern-ai-hep streamlit run demo.py --server.address=0.0.0.0

# Or run the default CPU smoke benchmark
docker run cern-ai-hep
```

---

## Design Notes

A few decisions that are easy to get wrong and are deliberate here:

- **`SafeBatchNorm1d` instead of `nn.BatchNorm1d`.** PyG collates a *variable*
  number of nodes, edges and graphs per batch, so the normalised axis can
  legitimately collapse to 1 — a single-node jet, a single-edge 2-particle
  event, or `batch_size=1`. Plain `BatchNorm1d` raises there. The safe variant
  falls back to running statistics; parameter names are unchanged, so existing
  checkpoints still load.
- **Masked pooling in the CNN baseline.** Without a mask, `AdaptiveAvgPool1d`
  averages over the zero padding, so a short jet scores differently depending
  on `max_particles`.
- **Strict checkpoint loading.** Scripts that produce reported metrics fail
  loudly when a checkpoint is missing instead of falling back to random weights,
  which would still emit finite, plausible-looking scores.
- **Raw AUROC is never post-hoc flipped** to exceed 0.5; doing so on test scores
  inflates the metric. Plots may use direction-corrected scores when labelled.
- **Split ratios are validated with `ValueError`, not `assert`** — assertions
  are stripped under `python -O`, silently producing garbage splits in optimised
  runs.

---

## Contributing

```bash
pip install -e ".[dev]"
make lint      # ruff check
make test      # pytest
```

CI runs on Python 3.10 and 3.12 and covers lint, the unit suite with coverage,
an end-to-end synthetic smoke run, and a package build.

Please keep the physics honest: AUROC values must be raw and unflipped,
checkpoints required for a reported number must be enforced rather than
optional, and any direction-corrected or exploratory quantity must be labelled
as such in the figure or log output.

---

## Hardware Specifications

Reference machine used for the published 6M-jet run:

| Hardware Component | Specification |
|:---|:---|
| **Host Processor** | Intel Core i5 (12th Generation) |
| **Graphics Processing Unit** | NVIDIA GeForce RTX 3050 (4 GB VRAM) |
| **System Memory (RAM)** | 16 GB DDR4 |
| **Storage** | NVMe SSD (High I/O for Awkward / ROOT reading) |
| **JetClass 6M Training Time** | ~45 hours total |

The CPU smoke configuration and the full test suite need no GPU.

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
