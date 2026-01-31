"""
Data generation module for VOD Causal Uplift.

Contains:
- Schemas for VOD entities (titles, users, treatments, outcomes)
- Oracle simulation for ground truth causal effects
- Synthetic data generation pipeline
"""

from .schemas import TitleMetadata, UserMetadata, TreatmentLog, InteractionOutcome
from .oracle import CausalOracle
from .generator import VODSyntheticData

__all__ = [
    "TitleMetadata",
    "UserMetadata", 
    "TreatmentLog",
    "InteractionOutcome",
    "CausalOracle",
    "VODSyntheticData",
]
