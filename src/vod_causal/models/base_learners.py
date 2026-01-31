"""
Base learners for the meta-learner framework.

Provides the foundation models used by meta-learners (T-Learner, X-Learner, etc.)
to estimate response functions for control and treatment groups.
"""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from xgboost import XGBRegressor


class BaseLearners:
    """
    Base predictors for the meta-learner framework.

    Trains separate models on control and treatment groups:
    - μ₀ (Control Model): Predicts outcome E[Y|X, T=0]
    - μ₁ (Treatment Model): Predicts outcome E[Y|X, T=1]

    These are used as the first stage in meta-learners like T-Learner, X-Learner,
    and DR-Learner to estimate response surfaces.

    Example:
        >>> learners = BaseLearners()
        >>> learners.fit(X, y, treatment)
        >>> mu_0 = learners.predict_control(X)
        >>> mu_1 = learners.predict_treatment(X)
        >>> simple_cate = mu_1 - mu_0  # T-Learner CATE estimate

    Attributes:
        mu_0: Fitted control response model
        mu_1: Fitted treatment response model
        fitted: Whether models have been fitted
    """

    DEFAULT_PARAMS: Dict = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        base_model_params: Optional[Dict] = None,
    ):
        """
        Initialize base learners.

        Args:
            base_model: Base estimator to use (if None, uses XGBRegressor)
            base_model_params: Parameters for XGBRegressor (ignored if base_model provided)
        """
        if base_model is not None:
            self._base_model = base_model
        else:
            params = self.DEFAULT_PARAMS.copy()
            if base_model_params:
                params.update(base_model_params)
            self._base_model = XGBRegressor(**params)

        self.mu_0: Optional[BaseEstimator] = None  # Control model
        self.mu_1: Optional[BaseEstimator] = None  # Treatment model
        self.fitted = False

        # Store training info for diagnostics
        self._n_control: int = 0
        self._n_treatment: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseLearners":
        """
        Fit separate models on control and treatment groups.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Outcome variable (revenue, conversion, etc.)
            treatment: Binary treatment indicator (0/1)

        Returns:
            self for method chaining
        """
        # Validate inputs
        X = self._validate_X(X)
        y = np.asarray(y).ravel()
        treatment = np.asarray(treatment).ravel()

        if len(X) != len(y) or len(X) != len(treatment):
            raise ValueError("X, y, and treatment must have same length")

        # Split data by treatment status
        control_mask = treatment == 0
        treatment_mask = treatment == 1

        self._n_control = control_mask.sum()
        self._n_treatment = treatment_mask.sum()

        if self._n_control == 0 or self._n_treatment == 0:
            raise ValueError("Both control and treatment groups must have samples")

        # Prepare sample weights if provided
        if sample_weight is not None:
            weight_0 = sample_weight[control_mask]
            weight_1 = sample_weight[treatment_mask]
        else:
            weight_0 = None
            weight_1 = None

        # Fit μ₀ on control group
        self.mu_0 = clone(self._base_model)
        if weight_0 is not None:
            self.mu_0.fit(X[control_mask], y[control_mask], sample_weight=weight_0)
        else:
            self.mu_0.fit(X[control_mask], y[control_mask])

        # Fit μ₁ on treatment group
        self.mu_1 = clone(self._base_model)
        if weight_1 is not None:
            self.mu_1.fit(X[treatment_mask], y[treatment_mask], sample_weight=weight_1)
        else:
            self.mu_1.fit(X[treatment_mask], y[treatment_mask])

        self.fitted = True
        return self

    def predict_control(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict μ₀(X) - expected outcome under control.

        Args:
            X: Feature matrix

        Returns:
            Predicted outcomes under control
        """
        if not self.fitted:
            raise RuntimeError("BaseLearners must be fitted before predict")

        X = self._validate_X(X)
        return self.mu_0.predict(X)

    def predict_treatment(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict μ₁(X) - expected outcome under treatment.

        Args:
            X: Feature matrix

        Returns:
            Predicted outcomes under treatment
        """
        if not self.fitted:
            raise RuntimeError("BaseLearners must be fitted before predict")

        X = self._validate_X(X)
        return self.mu_1.predict(X)

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate CATE using T-Learner approach: τ(x) = μ₁(x) - μ₀(x).

        This is the simplest meta-learner approach. For better estimates
        with imbalanced treatment/control, use X-Learner.

        Args:
            X: Feature matrix

        Returns:
            Estimated CATE for each sample
        """
        return self.predict_treatment(X) - self.predict_control(X)

    def get_feature_importance(
        self,
        model: str = "both",
        feature_names: Optional[list] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get feature importances from fitted models.

        Args:
            model: 'control', 'treatment', or 'both'
            feature_names: List of feature names

        Returns:
            Dict with feature importance DataFrames
        """
        if not self.fitted:
            raise RuntimeError("BaseLearners must be fitted first")

        result = {}

        def _get_importance(m, name):
            if hasattr(m, "feature_importances_"):
                imp = m.feature_importances_
            elif hasattr(m, "coef_"):
                imp = np.abs(m.coef_).ravel()
            else:
                return None

            if feature_names is None:
                names = [f"feature_{i}" for i in range(len(imp))]
            else:
                names = feature_names

            return pd.DataFrame({
                "feature": names,
                f"{name}_importance": imp,
            }).sort_values(f"{name}_importance", ascending=False)

        if model in ["control", "both"]:
            result["control"] = _get_importance(self.mu_0, "control")

        if model in ["treatment", "both"]:
            result["treatment"] = _get_importance(self.mu_1, "treatment")

        return result

    def _validate_X(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Convert X to numpy array format."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def get_training_summary(self) -> Dict:
        """Get summary of training data splits."""
        return {
            "n_control": self._n_control,
            "n_treatment": self._n_treatment,
            "control_fraction": self._n_control / (self._n_control + self._n_treatment),
            "treatment_fraction": self._n_treatment / (self._n_control + self._n_treatment),
        }


class SLearner(BaseEstimator, RegressorMixin):
    """
    S-Learner: Single model with treatment as feature.

    The simplest meta-learner approach - trains one model on all data
    with treatment as an additional feature. CATE is estimated as the
    difference in predictions when T=1 vs T=0.

    Less preferred for VOD due to potential regularization bias, but
    useful as a baseline.
    """

    def __init__(
        self,
        base_model: Optional[BaseEstimator] = None,
        base_model_params: Optional[Dict] = None,
    ):
        if base_model is not None:
            self._base_model = base_model
        else:
            params = BaseLearners.DEFAULT_PARAMS.copy()
            if base_model_params:
                params.update(base_model_params)
            self._base_model = XGBRegressor(**params)

        self.model = None
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "SLearner":
        """Fit single model with treatment as feature."""
        X_with_t = self._add_treatment_feature(X, treatment)
        self.model = clone(self._base_model)
        self.model.fit(X_with_t, y)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame, treatment: pd.Series) -> np.ndarray:
        """Predict outcome given features and treatment."""
        if not self.fitted:
            raise RuntimeError("SLearner must be fitted first")
        X_with_t = self._add_treatment_feature(X, treatment)
        return self.model.predict(X_with_t)

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate CATE as difference between T=1 and T=0 predictions."""
        if not self.fitted:
            raise RuntimeError("SLearner must be fitted first")

        n = len(X)
        t1 = np.ones(n)
        t0 = np.zeros(n)

        y1 = self.predict(X, t1)
        y0 = self.predict(X, t0)

        return y1 - y0

    def _add_treatment_feature(
        self,
        X: pd.DataFrame,
        treatment: Union[pd.Series, np.ndarray],
    ) -> np.ndarray:
        """Add treatment as a feature column."""
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        t_arr = np.asarray(treatment).reshape(-1, 1)
        return np.hstack([X_arr, t_arr])
