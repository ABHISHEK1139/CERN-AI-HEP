# CERN-AI Roadmap

## Month 1: Data Understanding

### Week 1 — Learn HEP Concepts
- [ ] Read about LHC, ATLAS, CMS
- [ ] Understand: collision events, muons, electrons, jets
- [ ] Identify CMS Open Data sample dataset
- [ ] Read 1 paper (GNNs in HEP)

### Week 2 — Event Explorer
- [ ] Download small CMS Open Data sample
- [ ] Build `event_explorer.ipynb`
- [ ] Event statistics, particle counts, momentum distributions

### Week 3 — ROOT File Mastery
- [ ] Install uproot, awkward-array
- [ ] Load events from ROOT files
- [ ] Extract: pt, eta, phi, mass, charge

### Week 4 — Graph Construction
- [ ] Convert collision event → graph
- [ ] Nodes = particles, Edges = spatial/detector relations
- [ ] Store as `torch_geometric.data.Data`

---

## Month 2: PyTorch Geometric

- [ ] Implement GCN on event graphs
- [ ] Implement GraphSAGE
- [ ] Implement GAT
- [ ] Understand message passing framework
- [ ] `graph-builder/` module complete

---

## Month 3: First Classifier

- [ ] Signal vs. background graph classifier
- [ ] Training pipeline with MLflow tracking
- [ ] Evaluation metrics (AUC, precision, recall)

---

## Month 4: Anomaly Detection

- [ ] Build graph autoencoder
- [ ] Encoder → latent space → decoder pipeline
- [ ] Reconstruction error as anomaly score
- [ ] `anomaly-engine/` module complete
- [ ] Top-k anomaly ranking

---

## Month 5: PhysicsNeMo

- [ ] Run official PhysicsNeMo tutorials
- [ ] GNN examples from PhysicsNeMo
- [ ] Scientific ML examples
- [ ] Benchmark against custom implementation
- [ ] `physicsnemo/` module complete

---

## Month 6: Research Report

- [ ] Full benchmark: MLP vs CNN vs GCN vs GraphSAGE vs PhysicsNeMo
- [ ] Final report with figures and tables
- [ ] Clean repository documentation
- [ ] `reports/` and `experiments/` complete
