# CERN AI Project Highlights

A summary of the core results, engineering optimizations, and scientific findings of the GNN-based Anomaly Detection pipeline.

---

### 📊 Key Performance Metrics
* **Dataset**: JetClass 6-Million Jet Subset (trained unsupervised on SM background, evaluated on Higgs decays)
* **MLP Baseline**: **0.6233** AUROC
* **GCN Baseline**: **0.6541** AUROC (with batched GPU $k$-NN graph bug fix)
* **EdgeConv (5 epochs)**: **0.6628** AUROC
* **EdgeConv (50 epochs)**: **0.6808** AUROC (Flagship performance)

---

### 🧠 Major Scientific Finding (Training Saturation)
* **Early Convergence**: The EdgeConv autoencoder converged rapidly, capturing **97.3%** of its peak anomaly classification performance (0.6628) within the first **5 epochs**.
* **Diminishing Returns**: Extending training from 5 epochs to 50 epochs only yielded a marginal **+0.018** improvement in AUROC, indicating that representation capacity and feature encoding (not training time) are the primary optimization bottlenecks.

---

### 💻 Hardware & GPU Infrastructure
* **RTX 3050 (4GB VRAM)**: Pipeline fully optimized to run large-scale training under consumer-grade constraints.
* **15x–40x Execution Speedup**: Designed custom GPU collation logic (`FastChunkedDataset`) that bypassed single-threaded Python CPU DataLoader collation bottlenecks.
* **1.62× Inference Speedup**: Integrated NVIDIA Modulus (PhysicsNeMo) FullyConnected decoders to accelerate the inference decoding phase from 2.79ms to 1.73ms natively on the GPU.

---

### 🚀 Core Engineering Accomplishments
1. **CMS Open Data Pipeline**: End-to-end ingestion from raw ROOT (NanoAOD) format to particle graph construction.
2. **Dynamic latent graph convolution**: Implemented EdgeConv to dynamically rebuild spatial topologies at each layer.
3. **GPU-native graphs**: $k$-NN construction ($k=8$) computed entirely on-device via PyTorch tensor operations.
