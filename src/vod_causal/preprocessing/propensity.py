"""
Propensity scoring model.

Estimates P(Treatment | X) - the probability that a unit receives treatment
given their features. This is required for the weighting stage of the X-Learner
and for assessing overlap/positivity assumptions.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


class PropensityModel:
    """
    Propensity Score Model: P(Treatment | X)

    Estimates the probability of receiving treatment given observed features.
    This is essential for:
    1. Inverse propensity weighting in causal estimators
    2. Assessing overlap (positivity) assumptions
    3. The final weighting stage of the X-Learner

    Supports multiple model types (XGBoost, Logistic Regression) and includes
    calibration and clipping utilities to ensure stable weights.

    Example:
        >>> propensity = PropensityModel(model_type='xgboost')
        >>> propensity.fit(X, treatment)
        >>> e_x = propensity.predict_propensity(X)

    Attributes:
        model_type: Type of classifier ('xgboost' or 'logistic')
        model: The fitted classifier
        calibrated_model: Calibrated version (optional)
        fitted: Whether the model has been fitted
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        calibrate: bool = False,
        random_state: int = 42,
        **model_kwargs,
    ):
        """
        Initialize propensity model.

        Args:
            model_type: 'xgboost' or 'logistic'
            calibrate: Whether to calibrate predictions (Platt scaling)
            random_state: Random seed for reproducibility
            **model_kwargs: Additional arguments passed to the classifier
        """
        self.model_type = model_type.lower()
        self.calibrate = calibrate
        self.random_state = random_state
        self.model_kwargs = model_kwargs

        self.model = None
        self.calibrated_model = None
        self.fitted = False

        # Store feature importance for diagnostics
        self.feature_importances_: Optional[np.ndarray] = None

    def _create_model(self):
        """Create the base classifier based on model_type."""
        if self.model_type == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": self.random_state,
            }
            default_params.update(self.model_kwargs)
            return XGBClassifier(**default_params)

        elif self.model_type == "logistic":
            default_params = {
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": self.random_state,
            }
            default_params.update(self.model_kwargs)
            return LogisticRegression(**default_params)

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "PropensityModel":
        """
        Train propensity model.

        Args:
            X: Feature matrix (n_samples, n_features)
            treatment: Binary treatment indicator (0/1)
            sample_weight: Optional sample weights

        Returns:
            self for method chaining
        """
        # Validate inputs
        if len(X) != len(treatment):
            raise ValueError("X and treatment must have same length")

        treatment_array = np.asarray(treatment).ravel()
        if not np.all(np.isin(treatment_array, [0, 1])):
            raise ValueError("Treatment must be binary (0/1)")

        # Create and fit model
        self.model = self._create_model()

        if sample_weight is not None:
            self.model.fit(X, treatment_array, sample_weight=sample_weight)
        else:
            self.model.fit(X, treatment_array)

        # Optionally calibrate
        if self.calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method="sigmoid",
                cv="prefit"
            )
            self.calibrated_model.fit(X, treatment_array)

        # Store feature importances
        if hasattr(self.model, "feature_importances_"):
            self.feature_importances_ = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            self.feature_importances_ = np.abs(self.model.coef_).ravel()

        self.fitted = True
        return self

    def predict_propensity(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict e(x) = P(T=1 | X).

        Args:
            X: Feature matrix

        Returns:
            Array of propensity scores in [0, 1]
        """
        if not self.fitted:
            raise RuntimeError("PropensityModel must be fitted before predict")

        if self.calibrate and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]

    @staticmethod
    def clip_propensity(
        propensity: np.ndarray,
        min_val: float = 0.01,
        max_val: float = 0.99,
    ) -> np.ndarray:
        """
        Clip extreme propensities to avoid numerical instability.

        Extreme propensity scores (very close to 0 or 1) can lead to
        unstable inverse propensity weights. Clipping ensures stable estimates.

        Args:
            propensity: Array of propensity scores
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clipped propensity scores
        """
        return np.clip(propensity, min_val, max_val)

    def compute_weights(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        weight_type: str = "ipw",
        stabilized: bool = True,
    ) -> np.ndarray:
        """
        Compute inverse propensity weights.

        Args:
            X: Feature matrix
            treatment: Binary treatment indicator
            weight_type: 'ipw' (inverse propensity) or 'overlap' (overlap weights)
            stabilized: Whether to use stabilized weights

        Returns:
            Array of weights
        """
        e_x = self.predict_propensity(X)
        e_x = self.clip_propensity(e_x)
        t = np.asarray(treatment).ravel()

        if weight_type == "ipw":
            # Standard IPW weights
            weights = t / e_x + (1 - t) / (1 - e_x)

            if stabilized:
                # Stabilize by treatment probability
                p_t = t.mean()
                weights = t * p_t / e_x + (1 - t) * (1 - p_t) / (1 - e_x)

        elif weight_type == "overlap":
            # Overlap weights (emphasize common support)
            weights = t * (1 - e_x) + (1 - t) * e_x

        else:
            raise ValueError(f"Unknown weight_type: {weight_type}")

        return weights

    def check_overlap(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        threshold: float = 0.1,
    ) -> dict:
        """
        Check positivity/overlap assumption.

        Positivity requires that all units have non-zero probability of
        being assigned to both treatment and control. This checks for
        violations by looking at extreme propensity scores.

        Args:
            X: Feature matrix
            treatment: Binary treatment indicator
            threshold: Threshold for "extreme" propensity

        Returns:
            Dict with overlap diagnostics
        """
        e_x = self.predict_propensity(X)
        t = np.asarray(treatment).ravel()

        # Check for near-zero propensities
        near_zero = e_x < threshold
        near_one = e_x > (1 - threshold)

        diagnostics = {
            "propensity_mean": e_x.mean(),
            "propensity_std": e_x.std(),
            "propensity_min": e_x.min(),
            "propensity_max": e_x.max(),
            "near_zero_count": near_zero.sum(),
            "near_zero_pct": near_zero.mean() * 100,
            "near_one_count": near_one.sum(),
            "near_one_pct": near_one.mean() * 100,
            "overlap_ok": (near_zero.sum() + near_one.sum()) / len(e_x) < 0.1,
        }

        # Propensity by treatment group
        diagnostics["propensity_treated_mean"] = e_x[t == 1].mean()
        diagnostics["propensity_control_mean"] = e_x[t == 0].mean()

        return diagnostics

    def get_feature_importance(
        self,
        feature_names: Optional[list] = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Get feature importances for treatment assignment.

        Understanding what drives treatment assignment helps diagnose
        confounding and design better experiments.

        Args:
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            DataFrame with features and their importances
        """
        if self.feature_importances_ is None:
            raise RuntimeError("No feature importances available")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": self.feature_importances_,
        })

        return df.nlargest(top_k, "importance").reset_index(drop=True)
