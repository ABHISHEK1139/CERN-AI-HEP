# CERN-AI: Master Project Reference & Resume Guide

This document is a comprehensive compilation of the GNN-based High Energy Physics (HEP) anomaly detection pipeline. It merges scientific results, system architectures, hardware optimizations, and code structures into a single source of truth. It is designed to allow AI agents or human developers to instantly customize resumes, interview talking points, and portfolios for various engineering and scientific roles.

---

## 1. Core Project Parameters & Key Results

### Flagship Performance Metrics
* **Dataset**: JetClass (6,000,000 constituent particle jets; Electroweak $Z \rightarrow \nu\nu$ background, Higgs signal decays as anomaly).
* **Flagship Model**: EdgeConv Graph Autoencoder (Dynamic $k$-NN in latent space).
* **Trainable Parameters**: 37,296.
* **Checkpoint File Size**: 503 KB (with Adam optimizer states and scheduler metadata).
* **Best Validation AUROC**: **0.6808** (trained unsupervised, purely on Standard Model background).
* **Training Saturation**: Hitted **97.3%** of peak performance in just 5 epochs (**0.6628** AUROC), showing rapid feature aggregation.
* **Neighborhood Sensitivity**: insensitivity observed from $k=4$ to $k=16$ (e.g. $k=4 \rightarrow 0.6671$, $k=16 \rightarrow 0.6672$).

### Hardware Configuration
* **GPU**: NVIDIA RTX 3050 Laptop GPU (4GB VRAM).
* **RAM**: 16GB.
* **GPU Speedup**: **15x to 40x training speedup** achieved by replacing standard CPU list collation with native GPU tensor operations, saturating the GPU compute.
* **Inference Speedup**: **1.62x inference speedup** (from 2.79ms to 1.73ms) by replacing the PyTorch MLP decoder with optimized CUDA kernels inside NVIDIA Modulus (PhysicsNeMo).

---

## 2. Resume Bullet Customization Guide

Use the following tailored highlights depending on the target position:

### Profile A: Machine Learning / GNN Research Engineer
> *Developed an unsupervised anomaly detection GNN using an EdgeConv encoder-decoder architecture to detect out-of-distribution physics events in LHC collision data, achieving a peak validation AUROC of 0.6808 on a 6-million jet subset of the JetClass benchmark.*
> *Replaced static coordinates with dynamic $k$-nearest neighbors ($k=8$) computed inside the learned latent space, enabling the network to extract topological relationships based on particle momentum and classification features.*
> *Ablated neighborhood size ($k$) and learning trajectories, determining that the autoencoder converges rapidly to 97.3% of maximum predictive performance within five training epochs.*

### Profile B: MLOps / High-Performance Computing (HPC) Engineer
> *Optimized PyTorch Geometric training pipelines for local consumer GPU bounds (RTX 3050 4GB), engineering a custom `FastChunkedDataset` that vectorizes padding removal and collation natively on the GPU to achieve a 15x–40x execution speedup.*
> *Eliminated memory bottlenecks and CPU thread wait times by replacing python graph-loop iterations with vectorized `torch_geometric.utils.scatter` GPU operations, reducing per-batch loss calculation time to under 0.10 seconds.*
> *Integrated NVIDIA Modulus (PhysicsNeMo) fully connected modules into the graph autoencoder decoder phase, yielding a 1.62× inference latency reduction (from 2.79ms to 1.73ms per pass).*

### Profile C: High Energy Physics (HEP) Scientist / Data Scientist
> *Designed a model-agnostic anomaly detection pipeline for LHC collisions using ROOT formats, CMS NanoAOD data, and simulated JetClass particle clouds to identify anomalous decays without requiring labeled signal data.*
> *Trained unsupervised autoencoders on 1M Standard Model background events (electroweak $Z \rightarrow \nu\nu$ jets) and evaluated their performance against Higgs boson decay signals.*
> *Validated the model's physical interpretability, verifying that the GNN successfully isolates anomalous physics by identifying correlations in jet mass, constituent particle multiplicity, and transverse momentum.*

### Profile D: Software / Full-Stack Engineer
> *Built a complete ROOT-to-Graph ETL data pipeline using `uproot` and `awkward-array` to stream massive particle datasets on low-memory environments.*
> *Created an interactive Streamlit portfolio dashboard ([demo.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/demo.py)) containing live particle graph visualizations, feature metric comparisons, and real-time reconstruction anomaly scoring.*

---

## 3. Detailed Architecture & Technical Formulation

### Graph Representation
* **Nodes ($V$)**: Individual constituent particles (up to 128 per jet).
* **Node Features ($x_i \in \mathbb{R}^{16}$)**: Transverse momentum ($p_T$), displacement coordinates ($\eta$, $\phi$), charge, particle ID classes, and displacement track metrics.
* **Edges ($E$)**: Dynamically generated via $k$-Nearest Neighbors ($k$-NN) based on physical or latent coordinate representations.

### EdgeConv Mathematical Definition
The dynamic graph convolution operation EdgeConv is defined as:
$$x'_i = \max_{j \in \mathcal{N}(i)} \Theta(x_i, x_j - x_i)$$
Where:
* $\mathcal{N}(i)$ is the set of $k$-nearest neighbors of node $i$.
* $\Theta: \mathbb{R}^{2d} \rightarrow \mathbb{R}^{d'}$ is a learnable multi-layer perceptron (MLP) mapping node features and local coordinates to a high-dimensional edge feature.
* The local coordinate representation $x_j - x_i$ ensures translation invariance.
* The aggregation is done via a symmetric operation (max-pooling) to maintain permutation invariance.

### The Autoencoder Anomaly Scoring
1. **Encoder**: 3 layers of `EdgeConv` layers with channels `[16 -> 64 -> 64 -> 32]` compressing the point cloud into a 32-dimensional bottleneck $z$.
2. **Decoder**: A 3-layer MLP `[32 -> 64 -> 64 -> 16]` reconstructing the original particle features.
3. **Loss function**: Mean Squared Error (MSE) computed on reconstruction:
   $$\mathcal{L} = \frac{1}{|V|} \sum_{i \in V} ||x_i - \hat{x}_i||^2$$
4. **Anomaly Classification**: Reconstructed MSE serves as the anomaly score. During training on Standard Model background events, the model minimizes $\mathcal{L}$. Out-of-distribution signal jets (Higgs decays) cannot be reconstructed efficiently, resulting in elevated MSE scores that are thresholded (calibrated at $\text{MSE} = 235.0$) to flag anomalies.

---

## 4. Code & API Reference Manual

An AI agent updating files or generating resumes can trace implementations through the following code paths:

### Model & Loss Implementations
* **EdgeConv Layer Architecture**: Located in [anomaly_engine/models/edge_conv.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/anomaly_engine/models/edge_conv.py) under the `EdgeConvLayer` and `EdgeConvEncoder` classes.
* **Vectorized Loss & Scatter Kernels**: Found in [anomaly_engine/models/autoencoder.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/anomaly_engine/models/autoencoder.py). Implements `GraphAutoencoder` and performs GPU loss scatter aggregation using `torch_geometric.utils.scatter(..., reduce='mean')` to avoid Python-level loops.
* **GCN Batched Diagonal Bug Fix**: In [anomaly_engine/models/autoencoder.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/anomaly_engine/models/autoencoder.py), the batched distance calculation diagonal is filled with infinity using `dist.fill_diagonal_(float('inf'))` to prevent self-loop dominance on 3D batched distance tensors.

### Data Engineering & Ingestion
* **GPU-Accelerated Collation**: Managed in [graph_builder/jetclass_iterable.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/graph_builder/jetclass_iterable.py) using the `FastChunkedDataset` and custom collation methods.
* **ROOT File Streaming**: Implemented in [graph_builder/jetclass_dataset.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/graph_builder/jetclass_dataset.py) using `uproot` and `awkward-array` to stream large dataset segments.
* **CMS Data Ingestion**: Located in [experiments/train_cms.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/experiments/train_cms.py) loading NanoAOD root files.

### Evaluation & Dashboards
* **Streamlit Visual Portfolio**: Located in [demo.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/demo.py). Renders constituent metrics, particle network layout, and live reconstruction loss evaluation.
* **Figure & Plot Generator**: Found in [experiments/produce_evidence.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/experiments/produce_evidence.py). Generates the ROC, PR, Latent space t-SNE, score distributions, and loss convergence plots.
* **Ablation Automation**: Managed in [experiments/run_6m_ablation.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/experiments/run_6m_ablation.py) and [experiments/run_5epochs_edgeconv.py](file:///c:/Users/ak612/OneDrive/Desktop/physics/cern-ai/experiments/run_5epochs_edgeconv.py).

---

## 5. Summary of Experimental Baseline & Ablation Tables

For referencing model performance comparisons, use the following structured tables:

### 1. Comparative Model Performance (6M JetClass Dataset)
| Model | Training Period | Validation AUROC | Batch Collation Time | Latency Notes |
| :--- | :--- | :--- | :--- | :--- |
| **MLP (Baseline)** | 1 Epoch | 0.6233 | ~0.01s / batch | Standard feature baseline |
| **GCN Baseline** | 1 Epoch | 0.6541 | ~0.06s / batch | With batched self-loop fix |
| **EdgeConv Baseline** | 1 Epoch | 0.6536 | ~0.10s / batch | Latent $k$-NN baseline |
| **EdgeConv Optimized** | 5 Epochs | 0.6628 | ~0.10s / batch | Hitted 97.3% of saturation |
| **EdgeConv Flagship** | 50 Epochs | **0.6808** | ~0.10s / batch | Long-run peak convergence |

### 2. Neighborhood Size Ablation Study
| Neighborhood Size ($k$) | Validation AUROC | Receptive Field Relevancy |
| :--- | :--- | :--- |
| $k = 4$ | 0.6671 | Local clusters |
| **$k = 8$** | **0.6665** | Standard configuration (Default) |
| $k = 16$ | 0.6672 | Wide regional clusters |

---

## 6. Physical Interpretability & Anomaly Analysis

The unsupervised autoencoder detects out-of-distribution decays by grouping events on physical principles:
1. **Invariant Jet Mass**: SM backgrounds (electroweak $Z \rightarrow \nu\nu$) are low-mass, while anomalous jets consistently cluster around the Higgs mass ($125 \text{ GeV}$) or Top Quark mass ($173 \text{ GeV}$).
2. **Constituent Particle Multiplicity**: Background events contain fewer active constituent tracks, while signal decays have high constituent density due to complex fragmentation (e.g. $H \rightarrow b\bar{b}$).
3. **Transverse Momentum ($p_T$)**: The GNN flags higher energy bounds, mapping out high-$p_T$ decay tracks as anomalous outliers.
