"""
Evaluation module for VOD Causal Uplift.

Contains:
- Uplift metrics (Qini curves, AUUC)
- Visualization utilities
- Policy ranking and simulation
"""

from .metrics import UpliftMetrics
from .visualization import plot_qini_curve, plot_cate_distribution
from .policy_ranker import PolicyRanker

__all__ = [
    "UpliftMetrics",
    "plot_qini_curve",
    "plot_cate_distribution",
    "PolicyRanker",
]
