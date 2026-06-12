# CERN-AI

## Graph Neural Network Based Anomaly Detection for LHC Events

A scientific machine learning system that learns normal particle collision patterns from CERN Open Data and flags anomalous events for physicist review — using graph neural networks, autoencoders, and NVIDIA PhysicsNeMo.

---

## Problem

The LHC produces ~1 billion collision events. Most are standard model background. A tiny fraction may contain unusual physics. Humans cannot inspect every event. Traditional trigger systems use hand-crafted rules.

**This project learns what "normal" looks like and flags what looks strange — without predefined labels.**

## Approach

```
CMS Open Data (ROOT files)
        ↓
  Event Ingestion (uproot, awkward-array)
        ↓
  Graph Construction (particles as nodes, relations as edges)
        ↓
  GNN Encoder (GCN / GraphSAGE / GAT)
        ↓
  Graph Autoencoder (latent space compression)
        ↓
  Reconstruction Error → Anomaly Score
        ↓
  Top-k Anomalous Events for Review
```

## Architecture

| Component | Description |
|---|---|
| `event-ingestion/` | ROOT file parsing, event statistics, feature extraction |
| `graph-builder/` | Collision event → PyTorch Geometric graph conversion |
| `anomaly-engine/` | Graph autoencoder anomaly detection pipeline |
| `physicsnemo/` | PhysicsNeMo benchmarking and physics-informed models |
| `experiments/` | Training runs, hyperparameter sweeps, MLflow tracking |
| `reports/` | Analysis reports, figures, benchmark comparisons |
| `docs/` | Technical documentation and paper notes |

## Tech Stack

- **Python 3.11** — Core language
- **PyTorch** — Deep learning framework
- **PyTorch Geometric** — Graph neural network library
- **uproot / awkward-array** — ROOT file I/O without ROOT
- **CERN Open Data** — CMS collision event datasets
- **PhysicsNeMo** — NVIDIA scientific ML framework
- **MLflow** — Experiment tracking
- **Docker** — Reproducible environments

## Project Timeline

| Month | Milestone | Deliverable |
|---|---|---|
| 1 | Data Understanding | Event Explorer notebook, ROOT parser |
| 2 | Graph Construction | Event → Graph pipeline with PyG |
| 3 | GNN Classifier | Signal vs. background graph classifier |
| 4 | Anomaly Detection | Graph autoencoder anomaly engine |
| 5 | PhysicsNeMo | Benchmark against custom implementations |
| 6 | Research Report | Full comparison: MLP vs CNN vs GCN vs GraphSAGE vs PhysicsNeMo |

## License

MIT
