"""
X-Learner meta-learner for CATE estimation.

The X-Learner is a powerful meta-learner particularly effective when there
is treatment imbalance (e.g., few titles on promotion compared to the whole
catalog) - a common scenario in VOD promotional campaigns.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from xgboost import XGBRegressor

from .base_learners import BaseLearners
from ..preprocessing.propensity import PropensityModel


class XLearner:
    """
    X-Learner for Conditional Average Treatment Effect (CATE) estimation.

    The X-Learner is recommended for VOD promotional optimization due to its
    effectiveness with treatment imbalance (few titles on sale compared to
    the whole catalog). It uses cross-imputation of counterfactuals to
    leverage information across treatment groups.

    Algorithm:
        Stage 1: Estimate response functions μ₀, μ₁
        Stage 2: Impute individual treatment effects
            - For treated: D₁ = Y_obs - μ₀(X)
            - For control: D₀ = μ₁(X) - Y_obs
        Stage 3: Train CATE models τ₀, τ₁ on imputed effects
        Stage 4: Combine with propensity weighting
            τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)

    The propensity weighting allows the model to borrow strength from
    whichever group (control or treatment) has more samples.

    Example:
        >>> xlearner = XLearner()
        >>> xlearner.fit(X, y, treatment)
        >>> cate = xlearner.predict(X)

    Attributes:
        base_learners: Fitted response models (μ₀, μ₁)
        propensity_model: Fitted propensity model
        tau_0: CATE model trained on control group
        tau_1: CATE model trained on treatment group
        fitted: Whether the model has been fitted
    """

    def __init__(
        self,
        base_learner_params: Optional[Dict] = None,
        propensity_model_type: str = "xgboost",
        cate_model_params: Optional[Dict] = None,
        random_state: int = 42,
    ):
        """
        Initialize X-Learner.

        Args:
            base_learner_params: Parameters for base response models
            propensity_model_type: Type of propensity model ('xgboost' or 'logistic')
            cate_model_params: Parameters for CATE models (τ₀, τ₁)
            random_state: Random seed for reproducibility
        """
        self.base_learner_params = base_learner_params or {}
        self.propensity_model_type = propensity_model_type
        self.cate_model_params = cate_model_params or {}
        self.random_state = random_state

        self.base_learners: Optional[BaseLearners] = None
        self.propensity_model: Optional[PropensityModel] = None
        self.tau_0: Optional[BaseEstimator] = None  # CATE from control perspective
        self.tau_1: Optional[BaseEstimator] = None  # CATE from treatment perspective
        self.fitted = False

        # Store imputed effects for diagnostics
        self._d0: Optional[np.ndarray] = None
        self._d1: Optional[np.ndarray] = None

    def _create_cate_model(self) -> BaseEstimator:
        """Create CATE model (τ) using XGBoost."""
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": self.random_state,
        }
        default_params.update(self.cate_model_params)
        return XGBRegressor(**default_params)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
        propensity: Optional[np.ndarray] = None,
    ) -> "XLearner":
        """
        Fit the X-Learner model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Outcome variable (revenue, conversion probability)
            treatment: Binary treatment indicator (0/1)
            propensity: Pre-computed propensity scores (optional)

        Returns:
            self for method chaining
        """
        # Convert to numpy for indexing
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y).ravel()
        t_arr = np.asarray(treatment).ravel()

        if len(X_arr) != len(y_arr) or len(X_arr) != len(t_arr):
            raise ValueError("X, y, and treatment must have same length")

        # Masks for treatment groups
        control_mask = t_arr == 0
        treatment_mask = t_arr == 1

        # ===== Stage 1: Fit base learners =====
        self.base_learners = BaseLearners(base_model_params=self.base_learner_params)
        self.base_learners.fit(X, y, treatment)

        # ===== Stage 2: Impute counterfactuals =====
        # For treated units: D₁ = Y_observed - μ₀(X)
        # This is what the treatment effect appears to be from treated perspective
        mu_0_for_treated = self.base_learners.predict_control(X_arr[treatment_mask])
        self._d1 = y_arr[treatment_mask] - mu_0_for_treated

        # For control units: D₀ = μ₁(X) - Y_observed
        # This is what the treatment effect appears to be from control perspective
        mu_1_for_control = self.base_learners.predict_treatment(X_arr[control_mask])
        self._d0 = mu_1_for_control - y_arr[control_mask]

        # ===== Stage 3: Train CATE models on imputed effects =====
        # τ₁: learns CATE pattern from treatment group
        self.tau_1 = self._create_cate_model()
        self.tau_1.fit(X_arr[treatment_mask], self._d1)

        # τ₀: learns CATE pattern from control group
        self.tau_0 = self._create_cate_model()
        self.tau_0.fit(X_arr[control_mask], self._d0)

        # ===== Fit or use provided propensity model =====
        if propensity is not None:
            self._precomputed_propensity = propensity
            self.propensity_model = None
        else:
            self.propensity_model = PropensityModel(
                model_type=self.propensity_model_type,
                random_state=self.random_state,
            )
            self.propensity_model.fit(X, treatment)
            self._precomputed_propensity = None

        self.fitted = True
        return self

    def predict(
        self,
        X: pd.DataFrame,
        propensity: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Predict CATE τ(x) using propensity-weighted combination.

        The final estimate combines predictions from both CATE models,
        weighted by propensity to leverage the group with more data:
            τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)

        Args:
            X: Feature matrix
            propensity: Pre-computed propensity scores (optional)

        Returns:
            Array of CATE estimates
        """
        if not self.fitted:
            raise RuntimeError("XLearner must be fitted before predict")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Get propensity scores
        if propensity is not None:
            e_x = propensity
        elif self.propensity_model is not None:
            e_x = self.propensity_model.predict_propensity(X)
        else:
            raise ValueError("No propensity model fitted and none provided")

        # Clip propensities for numerical stability
        e_x = PropensityModel.clip_propensity(e_x)

        # Predict from both CATE models
        tau_0_pred = self.tau_0.predict(X_arr)
        tau_1_pred = self.tau_1.predict(X_arr)

        # ===== Stage 4: Propensity-weighted combination =====
        # Weight by propensity: when e(x) is high (likely treated), we trust
        # the control group's estimate (τ₀) more, and vice versa
        cate = e_x * tau_0_pred + (1 - e_x) * tau_1_pred

        return cate

    def predict_interval(
        self,
        X: pd.DataFrame,
        propensity: Optional[np.ndarray] = None,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict CATE with pseudo-confidence intervals.

        Uses the range between τ₀ and τ₁ predictions as a rough measure
        of uncertainty. Not a formal confidence interval, but useful for
        identifying high-uncertainty predictions.

        Args:
            X: Feature matrix
            propensity: Pre-computed propensity scores (optional)
            alpha: Not used (for API compatibility)

        Returns:
            Tuple of (cate, lower, upper)
        """
        if not self.fitted:
            raise RuntimeError("XLearner must be fitted before predict")

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        tau_0_pred = self.tau_0.predict(X_arr)
        tau_1_pred = self.tau_1.predict(X_arr)

        cate = self.predict(X, propensity)

        # Use min/max of the two models as pseudo-interval
        lower = np.minimum(tau_0_pred, tau_1_pred)
        upper = np.maximum(tau_0_pred, tau_1_pred)

        return cate, lower, upper

    def get_feature_importance(
        self,
        model: str = "cate",
        feature_names: Optional[list] = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """
        Get feature importances from CATE models.

        Args:
            model: 'cate' (average of τ₀, τ₁), 'tau_0', 'tau_1', or 'base'
            feature_names: List of feature names
            top_k: Number of top features to return

        Returns:
            DataFrame with feature importances
        """
        if not self.fitted:
            raise RuntimeError("XLearner must be fitted first")

        if model == "base":
            return self.base_learners.get_feature_importance(
                model="both",
                feature_names=feature_names,
            )

        # Get importances from CATE models
        if hasattr(self.tau_0, "feature_importances_"):
            imp_0 = self.tau_0.feature_importances_
            imp_1 = self.tau_1.feature_importances_
        else:
            raise RuntimeError("CATE models don't have feature_importances_")

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(imp_0))]

        if model == "tau_0":
            importances = imp_0
        elif model == "tau_1":
            importances = imp_1
        else:  # 'cate' - average
            importances = (imp_0 + imp_1) / 2

        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })

        return df.nlargest(top_k, "importance").reset_index(drop=True)

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the fitted model.

        Returns:
            Dict with training statistics and quality metrics
        """
        if not self.fitted:
            raise RuntimeError("XLearner must be fitted first")

        return {
            "n_control": len(self._d0) if self._d0 is not None else 0,
            "n_treatment": len(self._d1) if self._d1 is not None else 0,
            "d0_mean": float(self._d0.mean()) if self._d0 is not None else None,
            "d0_std": float(self._d0.std()) if self._d0 is not None else None,
            "d1_mean": float(self._d1.mean()) if self._d1 is not None else None,
            "d1_std": float(self._d1.std()) if self._d1 is not None else None,
            "base_learners": self.base_learners.get_training_summary(),
        }


class XLearnerWithEconML:
    """
    X-Learner wrapper using EconML library.

    Provides a simpler interface to EconML's XLearner implementation
    with additional convenience methods for VOD use cases.

    This is the recommended approach for production as EconML provides:
    - Proper uncertainty quantification
    - Cross-fitting for honest inference
    - Well-tested implementation
    """

    def __init__(
        self,
        models: Optional[BaseEstimator] = None,
        propensity_model: Optional[BaseEstimator] = None,
        random_state: int = 42,
    ):
        """
        Initialize EconML-based X-Learner.

        Args:
            models: Base model for response functions
            propensity_model: Model for propensity estimation
            random_state: Random seed
        """
        try:
            from econml.metalearners import XLearner as EconMLXLearner
        except ImportError:
            raise ImportError(
                "econml is required for XLearnerWithEconML. "
                "Install with: pip install econml"
            )

        if models is None:
            models = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=random_state,
            )

        if propensity_model is None:
            from sklearn.linear_model import LogisticRegressionCV
            propensity_model = LogisticRegressionCV(cv=3, random_state=random_state)

        self._model = EconMLXLearner(
            models=models,
            propensity_model=propensity_model,
        )
        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "XLearnerWithEconML":
        """Fit the EconML X-Learner."""
        y_arr = np.asarray(y).ravel()
        t_arr = np.asarray(treatment).ravel()

        self._model.fit(y_arr, t_arr, X=X)
        self.fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CATE."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")
        return self._model.effect(X)

    def predict_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict CATE with confidence intervals.

        Args:
            X: Feature matrix
            alpha: Significance level (0.1 = 90% CI)

        Returns:
            Tuple of (cate, lower, upper)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        cate = self._model.effect(X)
        lower, upper = self._model.effect_interval(X, alpha=alpha)

        return cate, lower.ravel(), upper.ravel()
