# CERN-AI Glossary

Quick reference for HEP terminology used in this project.

## Accelerators & Experiments

| Term | Definition |
|---|---|
| **LHC** | Large Hadron Collider — world's largest particle accelerator at CERN |
| **ATLAS** | A Toroidal LHC Apparatus — general-purpose LHC detector |
| **CMS** | Compact Muon Solenoid — general-purpose LHC detector |
| **CERN** | European Organization for Nuclear Research, Geneva |

## Particles

| Term | Definition |
|---|---|
| **Muon** | Heavy electron-like particle (μ). Penetrates most detectors. |
| **Electron** | Light charged lepton (e⁻). Detected in electromagnetic calorimeter. |
| **Photon** | Massless boson (γ). Electromagnetic radiation carrier. |
| **Jet** | Collimated spray of hadrons from quark/gluon fragmentation. |
| **MET** | Missing transverse energy — inferred from momentum imbalance (neutrinos). |

## Event Properties

| Term | Definition |
|---|---|
| **pT** | Transverse momentum — momentum perpendicular to beam axis |
| **η (eta)** | Pseudorapidity — angular position relative to beam axis |
| **φ (phi)** | Azimuthal angle around beam axis |
| **ΔR** | Distance in η-φ space: √(Δη² + Δφ²) |

## Data Formats

| Term | Definition |
|---|---|
| **ROOT** | CERN's data analysis framework and file format |
| **NanoAOD** | Compact CMS data format (~1-2 KB/event) |
| **MiniAOD** | Intermediate CMS data format (~30-50 KB/event) |

## ML Terms (Project-Specific)

| Term | Definition |
|---|---|
| **GCN** | Graph Convolutional Network |
| **GraphSAGE** | Graph Sample and Aggregate |
| **GAT** | Graph Attention Network |
| **Message Passing** | GNN computation: nodes exchange info along edges |
| **Reconstruction Error** | Autoencoder output vs input difference → anomaly signal |
