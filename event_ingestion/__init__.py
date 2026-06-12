"""
event_ingestion: ROOT file parsing, CMS data download, event statistics, 
feature extraction, and synthetic event generation.

Modules:
    config       — Dataset paths, particle collections, feature definitions
    downloader   — CMS Open Data download from CERN portal
    loader       — Event loading with uproot + awkward-array
    statistics   — Event-level and dataset-level statistics
    synthetic    — Synthetic collision event generator for testing
"""

from event_ingestion.config import EventConfig, PARTICLE_FEATURES
from event_ingestion.loader import EventLoader
from event_ingestion.statistics import EventStatistics
from event_ingestion.synthetic import SyntheticEventGenerator

__all__ = [
    "EventConfig",
    "PARTICLE_FEATURES",
    "EventLoader",
    "EventStatistics",
    "SyntheticEventGenerator",
]
