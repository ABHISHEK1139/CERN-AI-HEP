# CERN-AI: Final Comparison Report

## Graph Neural Network Based Anomaly Detection for LHC Events

---

## 1. Abstract

This report presents a comparative study of machine learning architectures for anomaly detection in high-energy physics (HEP) collision events from the Large Hadron Collider (LHC). We evaluate five architectures — MLP, 1D-CNN, GCN, GraphSAGE, and GAT — alongside NVIDIA's PhysicsNeMo MeshGraphNet framework. Collision events are represented as graphs where particles are nodes and spatial/detector relations are edges. A graph autoencoder trained on normal events identifies anomalies through reconstruction error.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | CMS Open Data / Synthetic Generator |
| Normal Events | 10,000 |
| Anomalous Events | 1,000 |
| Anomaly Types | High multiplicity, High pT, Unusual topology, Rare particles |
| Node Features | 11 (type one-hot + pT + η + φ + mass + charge + energy) |
| Edge Features | 4 (ΔR + Δη + Δφ + relative pT) |
| Graph Construction | k-NN in η-φ space (k=8) |

---

## 3. Models

### 3.1 Baselines (Non-Graph)

**MLP**: Pools node features (mean + max), ignores graph structure entirely.
**1D-CNN**: Sorts particles by pT, applies 1D convolutions over the sequence.

### 3.2 Graph Neural Networks

**GCN**: Graph Convolutional Network (Kipf & Welling, 2017). Spectral-based.
**GraphSAGE**: Sample & Aggregate (Hamilton et al., 2017). Inductive.
**GAT**: Graph Attention Network (Velickovic et al., 2018). Attention-based.

### 3.3 PhysicsNeMo

**MeshGraphNet**: NVIDIA's physics-informed GNN with encode-process-decode architecture.

---

## 4. Classification Results

| Model | Accuracy | AUROC | F1 | Precision | Recall | Parameters | Time (s) |
|---|---|---|---|---|---|---|---|
| MLP | — | — | — | — | — | — | — |
| CNN | — | — | — | — | — | — | — |
| GCN | — | — | — | — | — | — | — |
| GraphSAGE | — | — | — | — | — | — | — |
| GAT | — | — | — | — | — | — | — |
| PhysicsNeMo | — | — | — | — | — | — | — |

> Fill in after running: `python experiments/run_benchmark.py`

---

## 5. Anomaly Detection Results

| Encoder | AUROC | AUPRC | Normal Mean Error | Anomaly Mean Error | Separation |
|---|---|---|---|---|---|
| GCN | — | — | — | — | — |
| GraphSAGE | — | — | — | — | — |
| GAT | — | — | — | — | — |

> Fill in after running: `python experiments/train_autoencoder.py --encoder <name>`

---

## 6. Key Findings

1. **Graph structure matters**: GNN models (GCN, GraphSAGE, GAT) are expected to outperform baselines (MLP, CNN) that ignore particle relationships.

2. **Attention mechanisms**: GAT's attention mechanism can learn which particle interactions are most informative for anomaly detection.

3. **Unsupervised detection**: The autoencoder approach requires no anomaly labels during training, making it applicable to real physics searches.

4. **PhysicsNeMo**: The MeshGraphNet architecture provides a competitive baseline with physics-informed design principles.

---

## 7. Figures

- Training curves: `reports/figures/training_*.png`
- Anomaly score distributions: `reports/figures/anomaly_scores_*.png`
- ROC curves: `reports/figures/roc_*.png`
- Latent space visualization: `reports/figures/latent_space_*.png`
- Model comparison: `reports/figures/model_comparison.png`

---

## 8. Conclusion

This project demonstrates a complete scientific ML pipeline for anomaly detection in LHC collision events: from raw data ingestion and graph construction to GNN-based autoencoder anomaly scoring. The system learns normal collision patterns and flags events with unusual particle configurations, providing physicists with a prioritized list of events worthy of further investigation.

---

## References

1. Kipf & Welling. "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017.
2. Hamilton et al. "Inductive Representation Learning on Large Graphs." NeurIPS 2017.
3. Velickovic et al. "Graph Attention Networks." ICLR 2018.
4. CMS Collaboration. "CMS Open Data."
5. NVIDIA. "PhysicsNeMo: AI for Scientific Computing."
