"""
Double Machine Learning (DML) for price elasticity estimation.

DML is the preferred approach when treatment is continuous (e.g., variable
discount levels rather than binary promotion). It provides valid causal
inference while allowing flexible ML models for nuisance estimation.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor


class DoubleMachineLearning:
    """
    Double Machine Learning (DML) for Price Elasticity Estimation.

    Use when treatment is continuous (variable price/discount level) rather
    than binary. DML orthogonalizes the estimation to remove regularization
    bias and provide valid inference for the causal effect.

    Algorithm (Partially Linear Model):
        Y = θ·T + g(X) + ε   (true model)

        Step A: Residualize treatment
            - Regress T on X → get residual R_T = T - f̂(X)
        Step B: Residualize outcome  
            - Regress Y on X → get residual R_Y = Y - ĥ(X)
        Step C: Second stage
            - Regress R_Y ~ R_T → coefficient θ is the causal effect

    Cross-fitting is used to avoid overfitting bias: the nuisance functions
    (f, h) are estimated on a different fold than where residuals are computed.

    Example:
        >>> dml = DoubleMachineLearning()
        >>> dml.fit(X, treatment=price, outcome=demand)
        >>> theta, se = dml.get_elasticity()
        >>> print(f"Price elasticity: {theta:.3f} ± {1.96*se:.3f}")

    Attributes:
        theta: Estimated causal effect (elasticity)
        theta_se: Standard error of the estimate
        fitted: Whether the model has been fitted
    """

    def __init__(
        self,
        n_folds: int = 5,
        first_stage_model: str = "xgboost",
        first_stage_params: Optional[Dict] = None,
        random_state: int = 42,
    ):
        """
        Initialize DML estimator.

        Args:
            n_folds: Number of folds for cross-fitting
            first_stage_model: Model for nuisance functions ('xgboost' or 'linear')
            first_stage_params: Parameters for first-stage models
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.first_stage_model = first_stage_model.lower()
        self.first_stage_params = first_stage_params or {}
        self.random_state = random_state

        self.theta: Optional[float] = None  # Estimated elasticity
        self.theta_se: Optional[float] = None  # Standard error
        self.fitted = False

        # Store residuals for diagnostics
        self._residuals_treatment: Optional[np.ndarray] = None
        self._residuals_outcome: Optional[np.ndarray] = None

    def _get_first_stage_model(self) -> BaseEstimator:
        """Create a first-stage model for nuisance estimation."""
        if self.first_stage_model == "xgboost":
            default_params = {
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": self.random_state,
            }
            default_params.update(self.first_stage_params)
            return XGBRegressor(**default_params)

        elif self.first_stage_model == "linear":
            return LinearRegression()

        else:
            raise ValueError(f"Unknown first_stage_model: {self.first_stage_model}")

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,  # Continuous: price/discount level
        outcome: pd.Series,    # Demand/revenue
    ) -> "DoubleMachineLearning":
        """
        Fit DML model using cross-fitting.

        Cross-fitting ensures valid inference by using out-of-fold predictions
        for residualization, avoiding overfitting bias.

        Args:
            X: Confounders/features (n_samples, n_features)
            treatment: Continuous treatment variable (price, discount level)
            outcome: Outcome variable (demand, revenue, conversion)

        Returns:
            self for method chaining
        """
        # Convert to numpy
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        T_arr = np.asarray(treatment).ravel()
        Y_arr = np.asarray(outcome).ravel()

        n = len(X_arr)
        if n != len(T_arr) or n != len(Y_arr):
            raise ValueError("X, treatment, and outcome must have same length")

        # Initialize residual arrays
        residuals_treatment = np.zeros(n)
        residuals_outcome = np.zeros(n)

        # Cross-fitting
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for train_idx, test_idx in kf.split(X_arr):
            # First stage: predict treatment from X
            model_t = self._get_first_stage_model()
            model_t.fit(X_arr[train_idx], T_arr[train_idx])
            t_pred = model_t.predict(X_arr[test_idx])
            residuals_treatment[test_idx] = T_arr[test_idx] - t_pred

            # First stage: predict outcome from X
            model_y = self._get_first_stage_model()
            model_y.fit(X_arr[train_idx], Y_arr[train_idx])
            y_pred = model_y.predict(X_arr[test_idx])
            residuals_outcome[test_idx] = Y_arr[test_idx] - y_pred

        # Store for diagnostics
        self._residuals_treatment = residuals_treatment
        self._residuals_outcome = residuals_outcome

        # Second stage: OLS regression of residuals
        # R_Y ~ R_T
        self.theta, self.theta_se = self._second_stage_ols(
            residuals_treatment,
            residuals_outcome
        )

        self.fitted = True
        return self

    def _second_stage_ols(
        self,
        residuals_t: np.ndarray,
        residuals_y: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Perform second-stage OLS regression.

        θ̂ = Σ(R_T · R_Y) / Σ(R_T²)

        Args:
            residuals_t: Treatment residuals
            residuals_y: Outcome residuals

        Returns:
            Tuple of (theta, standard_error)
        """
        n = len(residuals_t)

        # OLS estimate
        theta = np.sum(residuals_t * residuals_y) / np.sum(residuals_t ** 2)

        # Compute robust standard error
        # Residual from second stage
        epsilon = residuals_y - theta * residuals_t

        # Heteroskedasticity-robust variance (HC0)
        var_theta = (
            np.sum(residuals_t ** 2 * epsilon ** 2)
            / (np.sum(residuals_t ** 2) ** 2)
        )

        se_theta = np.sqrt(var_theta)

        return float(theta), float(se_theta)

    def get_elasticity(self) -> Tuple[float, float]:
        """
        Get estimated price elasticity with standard error.

        Returns:
            Tuple of (theta, se):
                theta: Point estimate of the causal effect/elasticity
                se: Standard error (multiply by 1.96 for 95% CI)
        """
        if not self.fitted:
            raise RuntimeError("DML must be fitted first")

        return self.theta, self.theta_se

    def get_confidence_interval(self, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Get confidence interval for the elasticity estimate.

        Args:
            alpha: Significance level (0.05 = 95% CI)

        Returns:
            Tuple of (lower, upper) bounds
        """
        if not self.fitted:
            raise RuntimeError("DML must be fitted first")

        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)

        lower = self.theta - z * self.theta_se
        upper = self.theta + z * self.theta_se

        return float(lower), float(upper)

    def get_diagnostics(self) -> Dict:
        """
        Get diagnostic information about the DML fit.

        Returns:
            Dict with model diagnostics
        """
        if not self.fitted:
            raise RuntimeError("DML must be fitted first")

        # R-squared analog from residuals
        if self._residuals_treatment is not None and self._residuals_outcome is not None:
            # Fraction of treatment variation explained by X
            t_variance = np.var(self._residuals_treatment)
            # Fraction of outcome variation explained by X (approximately)
            y_variance = np.var(self._residuals_outcome)
        else:
            t_variance = None
            y_variance = None

        return {
            "theta": self.theta,
            "theta_se": self.theta_se,
            "t_statistic": self.theta / self.theta_se if self.theta_se > 0 else None,
            "p_value": self._compute_p_value(),
            "n_observations": len(self._residuals_treatment) if self._residuals_treatment is not None else None,
            "n_folds": self.n_folds,
            "residual_variance_treatment": t_variance,
            "residual_variance_outcome": y_variance,
        }

    def _compute_p_value(self) -> float:
        """Compute two-sided p-value for the null hypothesis θ=0."""
        from scipy import stats

        if self.theta_se == 0 or self.theta_se is None:
            return 1.0

        t_stat = self.theta / self.theta_se
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return float(p_value)


class DMLWithEconML:
    """
    Double Machine Learning wrapper using EconML library.

    EconML provides a more feature-complete implementation with:
    - Various DML variants (linear, forest, etc.)
    - Proper confidence intervals via bootstrap/influence functions
    - Heterogeneous treatment effects via CATE forests

    This is the recommended approach for production DML.
    """

    def __init__(
        self,
        model_type: str = "linear",
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize EconML-based DML.

        Args:
            model_type: 'linear' for LinearDML, 'forest' for CausalForestDML
            n_folds: Number of cross-fitting folds
            random_state: Random seed
        """
        try:
            from econml.dml import LinearDML, CausalForestDML
        except ImportError:
            raise ImportError(
                "econml is required for DMLWithEconML. "
                "Install with: pip install econml"
            )

        self.model_type = model_type.lower()
        self.n_folds = n_folds
        self.random_state = random_state

        if self.model_type == "linear":
            self._model = LinearDML(
                model_y=XGBRegressor(n_estimators=100, random_state=random_state),
                model_t=XGBRegressor(n_estimators=100, random_state=random_state),
                cv=n_folds,
                random_state=random_state,
            )
        elif self.model_type == "forest":
            self._model = CausalForestDML(
                model_y=XGBRegressor(n_estimators=100, random_state=random_state),
                model_t=XGBRegressor(n_estimators=100, random_state=random_state),
                cv=n_folds,
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        treatment: pd.Series,
        outcome: pd.Series,
        W: Optional[pd.DataFrame] = None,
    ) -> "DMLWithEconML":
        """
        Fit the EconML DML model.

        Args:
            X: Effect modifiers (heterogeneity features)
            treatment: Continuous treatment
            outcome: Outcome variable
            W: Additional confounders (controls only, not effect modifiers)

        Returns:
            self for method chaining
        """
        Y = np.asarray(outcome).ravel()
        T = np.asarray(treatment).ravel()

        if W is not None:
            self._model.fit(Y, T, X=X, W=W)
        else:
            self._model.fit(Y, T, X=X)

        self.fitted = True
        return self

    def effect(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get heterogeneous treatment effects.

        For LinearDML, returns the same effect for all X.
        For CausalForestDML, returns personalized effects.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")
        return self._model.effect(X)

    def effect_interval(
        self,
        X: pd.DataFrame,
        alpha: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get confidence intervals for effects."""
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")
        return self._model.effect_interval(X, alpha=alpha)

    def const_marginal_effect(self) -> Tuple[float, float]:
        """
        Get the average treatment effect (for LinearDML).

        Returns:
            Tuple of (effect, std_error)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        if self.model_type == "linear":
            effect = self._model.const_marginal_effect()
            se = self._model.const_marginal_effect_inference().std_err
            return float(effect), float(se)
        else:
            raise RuntimeError("const_marginal_effect only for LinearDML")
