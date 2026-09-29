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
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _describe(values) -> dict[str, float]:
    """Mean/std/min/max/median of a 1-D sequence, as plain floats."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def _empty_summary() -> dict[str, Any]:
    """Zero-particle summary with the same keys as a populated one.

    The previous early-return omitted ``pt_GeV``/``eta``/``phi``/``energy_GeV``,
    so ``print_summary`` raised ``KeyError: 'pt_GeV'`` on any dataset where
    every event was filtered out.
    """
    return {
        "n_events": 0,
        "n_particles_total": 0,
        "particles_per_event": {
            "mean": 0.0, "std": 0.0, "min": 0, "max": 0, "median": 0.0,
        },
        "type_counts": {},
        "type_fractions": {},
        "pt_GeV": {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
        },
        "eta": {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        },
        "phi": {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        },
        "mass_GeV": {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
        },
        "energy_GeV": {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0,
        },
    }


class EventStatistics:
    """Compute and visualize collision event statistics."""

    def compute(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Compute summary statistics over a list of events.

        Args:
            events: List of event dicts from EventLoader.

        Returns:
            Dictionary of computed statistics. The schema is identical whether
            or not any particles were found, so callers (including
            :meth:`print_summary`) never hit a KeyError on the zero-particle
            path.
        """
        n_events = len(events)
        if n_events == 0:
            return _empty_summary()

        # Particle multiplicities
        multiplicities = [e.get("n_particles", len(e.get("particles", []))) for e in events]

        # Per-type counts
        type_counts = Counter()
        all_pt: list[float] = []
        all_eta: list[float] = []
        all_phi: list[float] = []
        all_energy: list[float] = []
        all_mass: list[float] = []

        for event in events:
            for p in event.get("particles", []):
                if "type" not in p or "pt" not in p:
                    continue
                type_counts[p["type"]] += 1
                all_pt.append(p["pt"])
                all_eta.append(p.get("eta", 0.0))
                all_phi.append(p.get("phi", 0.0))
                all_energy.append(p.get("energy", p["pt"]))
                all_mass.append(p.get("mass", 0.0))

        if len(all_pt) == 0:
            summary = _empty_summary()
            summary["n_events"] = n_events
            summary["particles_per_event"] = _describe(multiplicities)
            return summary

        pt_arr = np.asarray(all_pt, dtype=float)
        eta_arr = np.asarray(all_eta, dtype=float)
        phi_arr = np.asarray(all_phi, dtype=float)
        energy_arr = np.asarray(all_energy, dtype=float)
        mass_arr = np.asarray(all_mass, dtype=float)

        return {
            "n_events": n_events,
            "n_particles_total": int(pt_arr.size),
            "particles_per_event": _describe(multiplicities),
            "type_counts": dict(type_counts),
            "type_fractions": {
                k: v / pt_arr.size for k, v in type_counts.items()
            },
            "pt_GeV": _describe(pt_arr),
            "eta": _describe(eta_arr),
            "phi": _describe(phi_arr),
            "mass_GeV": _describe(mass_arr),
            "energy_GeV": {
                "mean": float(np.mean(energy_arr)),
                "std": float(np.std(energy_arr)),
                "min": float(np.min(energy_arr)),
                "max": float(np.max(energy_arr)),
                "median": float(np.median(energy_arr)),
            },
        }

    def print_summary(self, events: list[dict[str, Any]]) -> None:
        """Print formatted statistics."""
        s = self.compute(events)
        if s["n_events"] == 0:
            print("No events to summarize.")
            return

        print("=" * 60)
        print("COLLISION EVENT STATISTICS")
        print("=" * 60)
        print(f"  Events:           {s['n_events']:,}")
        print(f"  Total particles:  {s.get('n_particles_total', 0):,}")
        print()
        print("  Particles per event:")
        ppe = s["particles_per_event"]
        print(f"    Mean:   {ppe['mean']:.1f} +/- {ppe['std']:.1f}")
        print(f"    Range:  [{ppe['min']}, {ppe['max']}]")
        print(f"    Median: {ppe['median']:.0f}")
        print()
        print("  Particle types:")
        for ptype, count in sorted(s["type_counts"].items(), key=lambda x: -x[1]):
            frac = s["type_fractions"][ptype]
            print(f"    {ptype:12s}  {count:>8,}  ({frac:.1%})")
        if not s["type_counts"]:
            print("    (none - every event had zero usable particles)")

        if s.get("n_particles_total", 0) == 0:
            print()
            print("  No particles survived selection; distribution stats omitted.")
            print("=" * 60)
            return

        print()
        print("  Transverse momentum (pT):")
        pt = s["pt_GeV"]
        print(f"    Mean:   {pt['mean']:.1f} +/- {pt['std']:.1f} GeV")
        print(f"    Range:  [{pt['min']:.1f}, {pt['max']:.1f}] GeV")
        print(f"    Median: {pt['median']:.1f} GeV")
        print()
        print("  Pseudorapidity (eta):")
        eta = s["eta"]
        print(f"    Mean:   {eta['mean']:.2f} +/- {eta['std']:.2f}")
        print(f"    Range:  [{eta['min']:.2f}, {eta['max']:.2f}]")
        print("=" * 60)

    def plot_distributions(
        self,
        events: list[dict[str, Any]],
        output_dir: str | None = None,
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

        if not events:
            logger.warning("plot_distributions received 0 events; nothing to plot.")
            return

        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Collect arrays
        all_pt, all_eta, all_phi = [], [], []
        type_labels = []
        multiplicities = []

        for event in events:
            multiplicities.append(event.get("n_particles", len(event.get("particles", []))))
            for p in event.get("particles", []):
                if "type" not in p or "pt" not in p:
                    continue
                all_pt.append(p["pt"])
                all_eta.append(p.get("eta", 0.0))
                all_phi.append(p.get("phi", 0.0))
                type_labels.append(p["type"])

        all_pt = np.array(all_pt)
        all_eta = np.array(all_eta)
        all_phi = np.array(all_phi)

        if len(all_pt) == 0:
            logger.warning("plot_distributions found 0 particles; nothing to plot.")
            return

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
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(all_eta), min(5000, len(all_eta)), replace=False)
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
