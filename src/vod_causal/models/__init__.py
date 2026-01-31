"""
Models module for VOD Causal Uplift.

Contains:
- Base learners (XGBoost regressors for control/treatment)
- X-Learner meta-learner for CATE estimation
- Double Machine Learning for price elasticity
"""

from .base_learners import BaseLearners
from .xlearner import XLearner
from .dml import DoubleMachineLearning

__all__ = ["BaseLearners", "XLearner", "DoubleMachineLearning"]
