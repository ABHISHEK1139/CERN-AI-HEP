"""
Event statistics computation and visualization.

Computes per-event and dataset-level statistics for particle collisions:
multiplicities, pT distributions, eta-phi coverage, and energy spectra.

Usage:
    stats = EventStatistics()
    summary = stats.compute(events)
    stats.plot_distributions(events, output_dir="reports/figures/")
"""

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EventStatistics:
    """Compute and visualize collision event statistics."""

    def compute(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute summary statistics over a list of events.

        Args:
            events: List of event dicts from EventLoader.

        Returns:
            Dictionary of computed statistics.
        """
        n_events = len(events)
        if n_events == 0:
            return {"n_events": 0}

        # Particle multiplicities
        multiplicities = [e["n_particles"] for e in events]

        # Per-type counts
        type_counts = Counter()
        all_pt = []
        all_eta = []
        all_phi = []
        all_energy = []
        all_mass = []

        for event in events:
            for p in event["particles"]:
                type_counts[p["type"]] += 1
                all_pt.append(p["pt"])
                all_eta.append(p.get("eta", 0.0))
                all_phi.append(p.get("phi", 0.0))
                all_energy.append(p.get("energy", p["pt"]))
                all_mass.append(p.get("mass", 0.0))

        all_pt = np.array(all_pt)
        all_eta = np.array(all_eta)
        all_phi = np.array(all_phi)
        all_energy = np.array(all_energy)
        all_mass = np.array(all_mass)

        summary = {
            "n_events": n_events,
            "n_particles_total": len(all_pt),
            "particles_per_event": {
                "mean": float(np.mean(multiplicities)),
                "std": float(np.std(multiplicities)),
                "min": int(np.min(multiplicities)),
                "max": int(np.max(multiplicities)),
                "median": float(np.median(multiplicities)),
            },
            "type_counts": dict(type_counts),
            "type_fractions": {
                k: v / len(all_pt) for k, v in type_counts.items()
            },
            "pt_GeV": {
                "mean": float(np.mean(all_pt)),
                "std": float(np.std(all_pt)),
                "min": float(np.min(all_pt)),
                "max": float(np.max(all_pt)),
                "median": float(np.median(all_pt)),
            },
            "eta": {
                "mean": float(np.mean(all_eta)),
                "std": float(np.std(all_eta)),
                "min": float(np.min(all_eta)),
                "max": float(np.max(all_eta)),
            },
            "phi": {
                "mean": float(np.mean(all_phi)),
                "std": float(np.std(all_phi)),
                "min": float(np.min(all_phi)),
                "max": float(np.max(all_phi)),
            },
            "energy_GeV": {
                "mean": float(np.mean(all_energy)),
                "std": float(np.std(all_energy)),
            },
        }

        return summary

    def print_summary(self, events: List[Dict[str, Any]]) -> None:
        """Print formatted statistics."""
        s = self.compute(events)
        if s["n_events"] == 0:
            print("No events to summarize.")
            return

        print("=" * 60)
        print("COLLISION EVENT STATISTICS")
        print("=" * 60)
        print(f"  Events:           {s['n_events']:,}")
        print(f"  Total particles:  {s['n_particles_total']:,}")
        print()
        print("  Particles per event:")
        ppe = s["particles_per_event"]
        print(f"    Mean:   {ppe['mean']:.1f} ± {ppe['std']:.1f}")
        print(f"    Range:  [{ppe['min']}, {ppe['max']}]")
        print(f"    Median: {ppe['median']:.0f}")
        print()
        print("  Particle types:")
        for ptype, count in sorted(s["type_counts"].items(), key=lambda x: -x[1]):
            frac = s["type_fractions"][ptype]
            print(f"    {ptype:12s}  {count:>8,}  ({frac:.1%})")
        print()
        print("  Transverse momentum (pT):")
        pt = s["pt_GeV"]
        print(f"    Mean:   {pt['mean']:.1f} ± {pt['std']:.1f} GeV")
        print(f"    Range:  [{pt['min']:.1f}, {pt['max']:.1f}] GeV")
        print(f"    Median: {pt['median']:.1f} GeV")
        print()
        print("  Pseudorapidity (η):")
        eta = s["eta"]
        print(f"    Mean:   {eta['mean']:.2f} ± {eta['std']:.2f}")
        print(f"    Range:  [{eta['min']:.2f}, {eta['max']:.2f}]")
        print("=" * 60)

    def plot_distributions(
        self,
        events: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Plot key physics distributions.

        Args:
            events: List of event dicts.
            output_dir: If set, save figures to this directory.
            show: If True, display plots interactively.
        """
        import matplotlib.pyplot as plt

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect arrays
        all_pt, all_eta, all_phi = [], [], []
        type_labels = []
        multiplicities = []

        for event in events:
            multiplicities.append(event["n_particles"])
            for p in event["particles"]:
                all_pt.append(p["pt"])
                all_eta.append(p.get("eta", 0.0))
                all_phi.append(p.get("phi", 0.0))
                type_labels.append(p["type"])

        all_pt = np.array(all_pt)
        all_eta = np.array(all_eta)
        all_phi = np.array(all_phi)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Collision Event Distributions", fontsize=16, fontweight="bold")

        # 1. Particle multiplicity
        ax = axes[0, 0]
        ax.hist(multiplicities, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Particles per Event")
        ax.set_ylabel("Events")
        ax.set_title("Multiplicity Distribution")

        # 2. pT distribution (log scale)
        ax = axes[0, 1]
        ax.hist(all_pt, bins=50, color="#e74c3c", edgecolor="black", alpha=0.8, log=True)
        ax.set_xlabel("pT [GeV]")
        ax.set_ylabel("Particles (log)")
        ax.set_title("Transverse Momentum")

        # 3. Eta distribution
        ax = axes[0, 2]
        ax.hist(all_eta, bins=50, color="#2ecc71", edgecolor="black", alpha=0.8)
        ax.set_xlabel("η (pseudorapidity)")
        ax.set_ylabel("Particles")
        ax.set_title("Pseudorapidity")

        # 4. Phi distribution
        ax = axes[1, 0]
        ax.hist(all_phi, bins=50, color="#9b59b6", edgecolor="black", alpha=0.8)
        ax.set_xlabel("φ (azimuthal angle)")
        ax.set_ylabel("Particles")
        ax.set_title("Azimuthal Angle")

        # 5. η-φ scatter
        ax = axes[1, 1]
        sample_idx = np.random.choice(len(all_eta), min(5000, len(all_eta)), replace=False)
        ax.scatter(all_eta[sample_idx], all_phi[sample_idx], s=1, alpha=0.3, c="#e67e22")
        ax.set_xlabel("η")
        ax.set_ylabel("φ")
        ax.set_title("η-φ Coverage (sampled)")

        # 6. Particle type bar chart
        ax = axes[1, 2]
        type_counter = Counter(type_labels)
        types = sorted(type_counter.keys())
        counts = [type_counter[t] for t in types]
        colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
        ax.bar(types, counts, color=colors[: len(types)], edgecolor="black")
        ax.set_xlabel("Particle Type")
        ax.set_ylabel("Count")
        ax.set_title("Particle Composition")

        plt.tight_layout()

        if output_dir:
            path = Path(output_dir) / "event_distributions.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved figure: {path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
