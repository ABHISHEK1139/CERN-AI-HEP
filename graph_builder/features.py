"""
Feature engineering for collision event graphs.

Handles:
- Particle type one-hot encoding
- Physics feature normalization (log-pT, standardized eta/phi)
- Edge feature computation (ΔR, Δη, Δφ, relative pT)
- Composite feature construction (invariant mass, transverse energy)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from event_ingestion.config import NUM_PARTICLE_TYPES, PARTICLE_FEATURES

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract and normalize node/edge features for graph construction."""

    def __init__(
        self,
        normalize_pt: bool = True,
        log_pt: bool = True,
        standardize: bool = True,
    ):
        """
        Args:
            normalize_pt: Whether to normalize pT values.
            log_pt: Use log(pT) instead of raw pT.
            standardize: Z-score standardize features.
        """
        self.normalize_pt = normalize_pt
        self.log_pt = log_pt
        self.standardize = standardize

        # Running statistics for standardization
        self._fitted = False
        self._mean = None
        self._std = None

    def particle_to_node_features(
        self, particle: Dict[str, Any]
    ) -> np.ndarray:
        """
        Convert a particle dict to a node feature vector.

        Feature vector: [type_onehot(5), pt, eta, phi, mass, charge, energy]
        Total dimension: 11

        Args:
            particle: Particle dict from EventLoader.

        Returns:
            Feature vector of shape (11,).
        """
        # One-hot particle type
        type_onehot = np.zeros(NUM_PARTICLE_TYPES, dtype=np.float32)
        node_type = particle.get("node_type", 0)
        if 0 <= node_type < NUM_PARTICLE_TYPES:
            type_onehot[node_type] = 1.0

        # Physics features
        pt = particle["pt"]
        if self.log_pt:
            pt = np.log1p(pt)  # log(1 + pT) for numerical stability

        eta = particle.get("eta", 0.0)
        phi = particle.get("phi", 0.0)
        mass = particle.get("mass", 0.0)
        charge = particle.get("charge", 0.0)
        energy = particle.get("energy", particle["pt"])

        if self.log_pt:
            energy = np.log1p(energy)
            mass = np.log1p(mass)

        physics_features = np.array(
            [pt, eta, phi, mass, charge, energy], dtype=np.float32
        )

        return np.concatenate([type_onehot, physics_features])

    def compute_edge_features(
        self,
        node_i: Dict[str, Any],
        node_j: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute edge features between two particles.

        Features: [ΔR, Δη, Δφ, relative_pT]

        Args:
            node_i: First particle.
            node_j: Second particle.

        Returns:
            Edge feature vector of shape (4,).
        """
        eta_i = node_i.get("eta", 0.0)
        eta_j = node_j.get("eta", 0.0)
        phi_i = node_i.get("phi", 0.0)
        phi_j = node_j.get("phi", 0.0)
        pt_i = node_i["pt"]
        pt_j = node_j["pt"]

        delta_eta = eta_i - eta_j
        delta_phi = self._delta_phi(phi_i, phi_j)
        delta_r = np.sqrt(delta_eta**2 + delta_phi**2)

        # Relative pT: log ratio
        relative_pt = np.log1p(pt_i) - np.log1p(pt_j)

        return np.array(
            [delta_r, delta_eta, delta_phi, relative_pt], dtype=np.float32
        )

    @staticmethod
    def _delta_phi(phi1: float, phi2: float) -> float:
        """Compute Δφ wrapped to [-π, π]."""
        dphi = phi1 - phi2
        while dphi > np.pi:
            dphi -= 2 * np.pi
        while dphi < -np.pi:
            dphi += 2 * np.pi
        return dphi

    def extract_event_features(
        self, particles: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Extract node feature matrix for all particles in an event.

        Args:
            particles: List of particle dicts.

        Returns:
            Node feature matrix of shape (n_particles, feature_dim).
        """
        features = [self.particle_to_node_features(p) for p in particles]
        return np.stack(features, axis=0)

    def fit(self, all_features: np.ndarray) -> "FeatureExtractor":
        """
        Compute standardization statistics from training data.

        Args:
            all_features: Concatenated node features, shape (N, D).

        Returns:
            self
        """
        if self.standardize:
            self._mean = np.mean(all_features, axis=0)
            self._std = np.std(all_features, axis=0)
            # Avoid division by zero (e.g., one-hot columns)
            self._std[self._std < 1e-8] = 1.0
            self._fitted = True
            logger.info(f"Fitted standardizer on {all_features.shape[0]} samples")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Apply standardization to features.

        Args:
            features: Node feature matrix, shape (N, D).

        Returns:
            Standardized features.
        """
        if self.standardize and self._fitted:
            return (features - self._mean) / self._std
        return features

    def compute_delta_r_matrix(
        self, particles: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Compute pairwise ΔR distance matrix.

        Args:
            particles: List of particle dicts.

        Returns:
            Distance matrix of shape (n, n).
        """
        n = len(particles)
        eta = np.array([p.get("eta", 0.0) for p in particles])
        phi = np.array([p.get("phi", 0.0) for p in particles])

        # Vectorized ΔR computation
        d_eta = eta[:, None] - eta[None, :]
        d_phi = phi[:, None] - phi[None, :]

        # Wrap Δφ
        d_phi = np.mod(d_phi + np.pi, 2 * np.pi) - np.pi

        delta_r = np.sqrt(d_eta**2 + d_phi**2)
        return delta_r
