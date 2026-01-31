"""
Evaluation module for VOD Causal Uplift.

Contains:
- Uplift metrics (Qini curves, AUUC)
- Visualization utilities
- Policy ranking and simulation
"""

from .metrics import UpliftMetrics
from .visualization import (
    plot_qini_curve,
    plot_cate_distribution,
    plot_cate_calibration,
    plot_uplift_by_percentile,
    plot_propensity_distribution,
    plot_feature_importance,
    create_evaluation_dashboard,
)
from .policy_ranker import PolicyRanker

__all__ = [
    "UpliftMetrics",
    "plot_qini_curve",
    "plot_cate_distribution",
    "plot_cate_calibration",
    "plot_uplift_by_percentile",
    "plot_propensity_distribution",
    "plot_feature_importance",
    "create_evaluation_dashboard",
    "PolicyRanker",
]
