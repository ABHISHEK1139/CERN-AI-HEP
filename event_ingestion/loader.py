"""
Event loader for CMS NanoAOD ROOT files and synthetic data.

Loads particle-level event data using uproot + awkward-array,
applies selection cuts, and yields structured event dictionaries
ready for graph construction.

Usage:
    loader = EventLoader(config)
    for event in loader.iter_events("data/raw/file.root"):
        print(event["particles"])  # list of particle dicts
"""

import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np

from event_ingestion.config import EventConfig, PARTICLE_FEATURES

logger = logging.getLogger(__name__)


class EventLoader:
    """Load collision events from ROOT files or synthetic .npz files."""

    def __init__(self, config: Optional[EventConfig] = None):
        self.config = config or EventConfig()

    # ----------------------------------------------------------------
    # ROOT file loading (CMS NanoAOD)
    # ----------------------------------------------------------------

    def load_root(
        self,
        filepath: Union[str, Path],
        max_events: Optional[int] = None,
        tree_name: str = "Events",
    ) -> List[Dict[str, Any]]:
        """
        Load events from a CMS NanoAOD ROOT file.

        Args:
            filepath: Path to .root file.
            max_events: Limit number of events loaded.
            tree_name: TTree name in ROOT file.

        Returns:
            List of event dicts, each containing particle info.
        """
        try:
            import uproot
            import awkward as ak
        except ImportError:
            raise ImportError(
                "uproot and awkward required. Install: pip install uproot awkward"
            )

        filepath = Path(filepath)
        logger.info(f"Loading ROOT file: {filepath}")

        with uproot.open(filepath) as f:
            tree = f[tree_name]
            available_branches = tree.keys()

            # Collect branches we need
            branches_to_load = []
            for ptype, pinfo in PARTICLE_FEATURES.items():
                if ptype not in self.config.particle_types:
                    continue
                for branch in pinfo["branches"]:
                    if branch in available_branches:
                        branches_to_load.append(branch)

            # Also load nParticle count branches
            for ptype in self.config.particle_types:
                count_branch = f"n{ptype}"
                if count_branch in available_branches:
                    branches_to_load.append(count_branch)

            # Load all branches at once
            entry_stop = max_events if max_events else None
            arrays = tree.arrays(branches_to_load, entry_stop=entry_stop, library="ak")
            n_events = len(arrays)
            logger.info(f"Loaded {n_events} events with {len(branches_to_load)} branches")

        # Convert to event list
        events = []
        for i in range(n_events):
            event = self._extract_event_from_arrays(arrays, i)
            if event is not None:
                events.append(event)

        logger.info(
            f"Extracted {len(events)} events "
            f"(filtered {n_events - len(events)} below min_particles={self.config.min_particles})"
        )
        return events

    def _extract_event_from_arrays(
        self, arrays: Any, idx: int
    ) -> Optional[Dict[str, Any]]:
        """Extract a single event from awkward arrays."""
        import awkward as ak

        particles = []

        for ptype, pinfo in PARTICLE_FEATURES.items():
            if ptype not in self.config.particle_types:
                continue

            # Get pT array for this particle type
            pt_branch = f"{ptype}_pt"
            if pt_branch not in arrays.fields:
                continue

            pts = ak.to_numpy(arrays[pt_branch][idx])
            n_particles = len(pts)

            for j in range(n_particles):
                pt = float(pts[j])

                # Apply pT cut
                if pt < self.config.min_pt.get(ptype, 0.0):
                    continue

                particle = {
                    "type": ptype,
                    "node_type": pinfo["node_type"],
                    "pt": pt,
                }

                # Extract available core features
                for feat in pinfo["core"]:
                    if feat == "pt":
                        continue  # already have it
                    branch = f"{ptype}_{feat}"
                    if branch in arrays.fields:
                        val = arrays[branch][idx]
                        if hasattr(val, "__len__") and j < len(val):
                            particle[feat] = float(val[j])
                        elif not hasattr(val, "__len__"):
                            particle[feat] = float(val)

                # Compute energy from pT, eta, mass (if available)
                if "eta" in particle and "mass" in particle:
                    eta = particle["eta"]
                    mass = particle["mass"]
                    particle["energy"] = float(
                        np.sqrt(pt**2 * np.cosh(eta) ** 2 + mass**2)
                    )
                else:
                    particle["energy"] = pt  # fallback

                # Fill missing features with defaults
                particle.setdefault("eta", 0.0)
                particle.setdefault("phi", 0.0)
                particle.setdefault("mass", 0.0)
                particle.setdefault("charge", 0.0)

                particles.append(particle)

        # Apply particle count filter
        if len(particles) < self.config.min_particles:
            return None

        # Cap particles
        if len(particles) > self.config.max_particles:
            # Keep highest pT particles
            particles.sort(key=lambda p: p["pt"], reverse=True)
            particles = particles[: self.config.max_particles]

        return {
            "event_id": idx,
            "n_particles": len(particles),
            "particles": particles,
        }

    # ----------------------------------------------------------------
    # Synthetic data loading
    # ----------------------------------------------------------------

    def load_synthetic(
        self, filepath: Union[str, Path]
    ) -> List[Dict[str, Any]]:
        """
        Load events from synthetic .npz file.

        Args:
            filepath: Path to .npz file from SyntheticEventGenerator.

        Returns:
            List of event dicts.
        """
        filepath = Path(filepath)
        logger.info(f"Loading synthetic data: {filepath}")

        data = np.load(filepath, allow_pickle=True)
        events = data["events"].tolist()

        logger.info(f"Loaded {len(events)} synthetic events")
        return events

    # ----------------------------------------------------------------
    # Unified loading
    # ----------------------------------------------------------------

    def load(
        self,
        filepath: Union[str, Path],
        max_events: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load events from any supported format.

        Args:
            filepath: Path to .root or .npz file.
            max_events: Limit events loaded.

        Returns:
            List of event dicts.
        """
        filepath = Path(filepath)

        if filepath.suffix == ".root":
            return self.load_root(filepath, max_events=max_events)
        elif filepath.suffix == ".npz":
            events = self.load_synthetic(filepath)
            if max_events:
                events = events[:max_events]
            return events
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def iter_events(
        self,
        filepath: Union[str, Path],
        max_events: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield events one at a time (memory efficient)."""
        for event in self.load(filepath, max_events=max_events):
            yield event
