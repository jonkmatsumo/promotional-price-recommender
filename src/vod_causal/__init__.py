"""
VOD Causal Uplift Package

A causal inference toolkit for Video-on-Demand promotional optimization.
Implements X-Learner and Double Machine Learning for CATE estimation.
"""

__version__ = "0.1.0"
__author__ = "VOD Analytics Team"

from . import data
from . import preprocessing
from . import models
from . import evaluation

__all__ = ["data", "preprocessing", "models", "evaluation"]
