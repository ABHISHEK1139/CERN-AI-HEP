"""
Configuration for CMS NanoAOD event data.

Defines particle collections, feature branches, and dataset paths
following the CMS NanoAOD schema.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# ---- Particle Feature Definitions ----

PARTICLE_FEATURES = {
    "Muon": {
        "branches": [
            "Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass",
            "Muon_charge", "Muon_dxy", "Muon_dz",
            "Muon_pfRelIso04_all", "Muon_tightId",
        ],
        "core": ["pt", "eta", "phi", "mass", "charge"],
        "pdgid": 13,
        "node_type": 0,
    },
    "Electron": {
        "branches": [
            "Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass",
            "Electron_charge", "Electron_dxy", "Electron_dz",
            "Electron_pfRelIso03_all", "Electron_cutBased",
        ],
        "core": ["pt", "eta", "phi", "mass", "charge"],
        "pdgid": 11,
        "node_type": 1,
    },
    "Photon": {
        "branches": [
            "Photon_pt", "Photon_eta", "Photon_phi", "Photon_mass",
            "Photon_pfRelIso03_all", "Photon_cutBased",
        ],
        "core": ["pt", "eta", "phi", "mass"],
        "pdgid": 22,
        "node_type": 2,
    },
    "Jet": {
        "branches": [
            "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass",
            "Jet_btagDeepFlavB", "Jet_nConstituents",
        ],
        "core": ["pt", "eta", "phi", "mass"],
        "pdgid": 0,  # jets are composite
        "node_type": 3,
    },
    "MET": {
        "branches": ["MET_pt", "MET_phi"],
        "core": ["pt", "phi"],
        "pdgid": -1,
        "node_type": 4,
    },
}

# Number of particle types (for one-hot encoding)
NUM_PARTICLE_TYPES = len(PARTICLE_FEATURES)

# Core node feature dimension: [type_onehot(5), pt, eta, phi, mass, charge, energy]
NODE_FEATURE_DIM = NUM_PARTICLE_TYPES + 6  # 11

# Edge feature dimension: [delta_r, delta_eta, delta_phi, relative_pt]
EDGE_FEATURE_DIM = 4


@dataclass
class EventConfig:
    """Configuration for event data loading and processing."""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    raw_dir: Path = field(default_factory=lambda: Path("data/raw"))
    synthetic_dir: Path = field(default_factory=lambda: Path("data/synthetic"))
    graph_dir: Path = field(default_factory=lambda: Path("data/graphs"))

    # CMS Open Data
    cms_dataset_name: str = "CMS MonteCarlo2012 NanoAODv9"
    cms_record_id: int = 12350  # /DoubleMuParked/Run2012B NanoAOD

    # Particle selection cuts (minimum pT in GeV)
    min_pt: Dict[str, float] = field(default_factory=lambda: {
        "Muon": 5.0,
        "Electron": 7.0,
        "Photon": 10.0,
        "Jet": 25.0,
        "MET": 0.0,
    })

    # Event selection
    min_particles: int = 3  # minimum particles per event to build a graph
    max_particles: int = 50  # cap particles per event

    # Graph construction
    graph_strategy: str = "knn"  # knn, fully_connected, delta_r
    knn_k: int = 8
    delta_r_threshold: float = 1.5

    # Particle collections to use
    particle_types: List[str] = field(
        default_factory=lambda: ["Muon", "Electron", "Photon", "Jet", "MET"]
    )

    def ensure_dirs(self):
        """Create all data directories."""
        for d in [self.data_dir, self.raw_dir, self.synthetic_dir, self.graph_dir]:
            d.mkdir(parents=True, exist_ok=True)
