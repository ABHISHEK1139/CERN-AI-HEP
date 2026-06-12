"""
Synthetic collision event generator.

Generates physically-motivated synthetic collision events for testing
the full pipeline without downloading CERN data. Normal events follow
standard model-like distributions; anomalous events have injected
unusual particle configurations.

Usage:
    gen = SyntheticEventGenerator(seed=42)
    events, labels = gen.generate(n_normal=5000, n_anomaly=500)
    gen.save("data/synthetic/events.npz", events, labels)

CLI:
    python -m event_ingestion.synthetic --n-normal 10000 --n-anomaly 1000
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np

from event_ingestion.config import PARTICLE_FEATURES, EventConfig

logger = logging.getLogger(__name__)

# Particle type probabilities in a typical CMS event
PARTICLE_PROBS = {
    "Jet": 0.40,
    "Muon": 0.15,
    "Electron": 0.15,
    "Photon": 0.20,
    "MET": 0.10,
}

# Typical pT distributions (exponential scale parameter, GeV)
PT_SCALES = {
    "Muon": 25.0,
    "Electron": 30.0,
    "Photon": 20.0,
    "Jet": 50.0,
    "MET": 40.0,
}

# Mass values (approximate, GeV)
MASS_VALUES = {
    "Muon": 0.1057,
    "Electron": 0.000511,
    "Photon": 0.0,
    "Jet": 10.0,  # average jet mass
    "MET": 0.0,
}


class SyntheticEventGenerator:
    """Generate synthetic collision events with controllable anomalies."""

    def __init__(self, config: Optional[EventConfig] = None, seed: int = 42):
        self.config = config or EventConfig()
        self.rng = np.random.RandomState(seed)

    def _generate_particle(self, ptype: str, anomalous: bool = False) -> Dict[str, Any]:
        """Generate a single particle with physics-motivated features."""
        pinfo = PARTICLE_FEATURES[ptype]

        # pT: exponentially falling spectrum (typical for SM)
        scale = PT_SCALES[ptype]
        if anomalous:
            # Anomalous: boost pT or shift distribution
            scale *= self.rng.uniform(2.0, 5.0)

        pt = self.rng.exponential(scale) + self.config.min_pt.get(ptype, 0.0)

        # Eta: roughly Gaussian centered at 0, width ~2.5 (detector acceptance)
        if ptype == "MET":
            eta = 0.0  # MET has no eta
        elif anomalous and self.rng.random() < 0.3:
            # Anomalous: particles at unusual eta
            eta = self.rng.uniform(-5.0, 5.0)
        else:
            eta = self.rng.normal(0, 2.0)
            eta = np.clip(eta, -5.0, 5.0)

        # Phi: uniform in [-pi, pi]
        phi = self.rng.uniform(-np.pi, np.pi)

        # Mass
        mass = MASS_VALUES.get(ptype, 0.0)
        if ptype == "Jet":
            mass = self.rng.exponential(10.0)
            if anomalous and self.rng.random() < 0.4:
                # Anomalous jets: unusual mass (heavy resonance decay)
                mass = self.rng.uniform(80.0, 200.0)

        # Charge
        if ptype in ("Muon", "Electron"):
            charge = self.rng.choice([-1.0, 1.0])
        else:
            charge = 0.0

        # Energy
        if ptype == "MET":
            energy = pt
        else:
            energy = float(np.sqrt(pt**2 * np.cosh(eta) ** 2 + mass**2))

        return {
            "type": ptype,
            "node_type": pinfo["node_type"],
            "pt": float(pt),
            "eta": float(eta),
            "phi": float(phi),
            "mass": float(mass),
            "charge": float(charge),
            "energy": float(energy),
        }

    def _generate_event(
        self, event_id: int, anomalous: bool = False
    ) -> Dict[str, Any]:
        """Generate a single collision event."""
        # Number of particles: Poisson-distributed
        if anomalous:
            # Anomalous events may have unusual multiplicity
            anomaly_type = self.rng.choice(
                ["high_multiplicity", "high_pt", "unusual_topology", "rare_particles"]
            )
        else:
            anomaly_type = None

        if anomaly_type == "high_multiplicity":
            n_particles = self.rng.poisson(25) + 5
        else:
            n_particles = self.rng.poisson(8) + 3

        n_particles = min(n_particles, self.config.max_particles)
        n_particles = max(n_particles, self.config.min_particles)

        # Generate particle types
        ptypes = list(PARTICLE_PROBS.keys())
        probs = list(PARTICLE_PROBS.values())

        if anomaly_type == "rare_particles":
            # Boost probability of rare types
            probs = [0.15, 0.30, 0.30, 0.15, 0.10]

        probs = np.array(probs) / np.sum(probs)

        # Ensure at most 1 MET per event
        has_met = False
        particles = []

        for _ in range(n_particles):
            ptype = self.rng.choice(ptypes, p=probs)

            if ptype == "MET":
                if has_met:
                    ptype = self.rng.choice(["Jet", "Muon", "Electron", "Photon"])
                else:
                    has_met = True

            is_anomalous_particle = anomalous and anomaly_type in (
                "high_pt", "unusual_topology", "rare_particles"
            )
            particle = self._generate_particle(ptype, anomalous=is_anomalous_particle)
            particles.append(particle)

        # For unusual topology: add correlated particles
        if anomaly_type == "unusual_topology" and len(particles) >= 2:
            # Make some particles nearly collinear (unusual for SM)
            ref = particles[0]
            for p in particles[1:3]:
                p["eta"] = ref["eta"] + self.rng.normal(0, 0.1)
                p["phi"] = ref["phi"] + self.rng.normal(0, 0.1)

        return {
            "event_id": event_id,
            "n_particles": len(particles),
            "particles": particles,
            "is_anomaly": anomalous,
            "anomaly_type": anomaly_type,
        }

    def generate(
        self,
        n_normal: int = 10000,
        n_anomaly: int = 1000,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Generate a dataset of normal and anomalous events.

        Args:
            n_normal: Number of normal (background) events.
            n_anomaly: Number of anomalous (signal) events.

        Returns:
            Tuple of (events list, labels array).
            Labels: 0 = normal, 1 = anomaly.
        """
        logger.info(f"Generating {n_normal} normal + {n_anomaly} anomalous events...")

        events = []

        # Normal events
        for i in range(n_normal):
            event = self._generate_event(i, anomalous=False)
            events.append(event)

        # Anomalous events
        for i in range(n_anomaly):
            event = self._generate_event(n_normal + i, anomalous=True)
            events.append(event)

        # Labels
        labels = np.array([0] * n_normal + [1] * n_anomaly)

        # Shuffle
        indices = self.rng.permutation(len(events))
        events = [events[i] for i in indices]
        labels = labels[indices]

        logger.info(
            f"Generated {len(events)} events "
            f"({n_normal} normal, {n_anomaly} anomalous)"
        )

        return events, labels

    def save(
        self,
        output_path: Union[str, Path],
        events: List[Dict[str, Any]],
        labels: np.ndarray,
    ) -> Path:
        """
        Save generated events to .npz file.

        Args:
            output_path: Output file path.
            events: List of event dicts.
            labels: Array of labels.

        Returns:
            Path to saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            output_path,
            events=np.array(events, dtype=object),
            labels=labels,
        )

        logger.info(f"Saved {len(events)} events to {output_path}")
        return output_path



def main():
    """CLI entry point for synthetic data generation."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate synthetic collision events")
    parser.add_argument("--n-normal", type=int, default=10000, help="Normal events")
    parser.add_argument("--n-anomaly", type=int, default=1000, help="Anomalous events")
    parser.add_argument("--output", type=str, default="data/synthetic/", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    gen = SyntheticEventGenerator(seed=args.seed)
    events, labels = gen.generate(n_normal=args.n_normal, n_anomaly=args.n_anomaly)

    output_dir = Path(args.output)
    gen.save(output_dir / "events.npz", events, labels)

    # Print summary
    from event_ingestion.statistics import EventStatistics

    stats = EventStatistics()
    stats.print_summary(events)

    print(f"\nAnomaly ratio: {labels.sum()}/{len(labels)} ({labels.mean():.1%})")
    print(f"Saved to: {output_dir / 'events.npz'}")


if __name__ == "__main__":
    main()
