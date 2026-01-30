"""
Uplift metrics (Qini curves, AUUC).

Provides evaluation metrics specifically designed for causal/uplift models,
which differ from standard ML metrics as they measure the ability to
identify treatment-responsive individuals.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


class UpliftMetrics:
    """
    Evaluation metrics for causal uplift models.

    Key metrics:
    - Qini Curve: Cumulative gain plot showing incremental response
    - AUUC: Area Under Uplift Curve (normalized Qini)
    - MSE vs Oracle: Error against ground truth (when available)
    - Uplift by Decile: Response rates across predicted uplift bins

    These metrics measure how well the model identifies "persuadables" -
    users who will convert if treated but not otherwise.

    Example:
        >>> metrics = UpliftMetrics()
        >>> qini_x, qini_y = metrics.compute_qini_curve(y, treatment, predictions)
        >>> auuc = metrics.compute_auuc(qini_x, qini_y)
        >>> print(f"AUUC: {auuc:.4f}")
    """

    @staticmethod
    def compute_qini_curve(
        y_true: np.ndarray,
        treatment: np.ndarray,
        predictions: np.ndarray,
        n_bins: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Qini curve coordinates.

        The Qini curve shows the cumulative uplift as we target increasingly
        larger fractions of the population, sorted by predicted uplift.
        A good model should show significant lift in the early percentiles.

        Logic:
            1. Sort users by predicted uplift (descending)
            2. For each percentile, compute incremental conversions in treated vs control
            3. Plot cumulative incremental response vs fraction targeted

        Args:
            y_true: Binary outcome (0/1)
            treatment: Binary treatment indicator (0/1)
            predictions: Predicted CATE/uplift scores
            n_bins: Number of points on the curve

        Returns:
            Tuple of (x_coords, y_coords) for plotting
        """
        y_true = np.asarray(y_true).ravel()
        treatment = np.asarray(treatment).ravel()
        predictions = np.asarray(predictions).ravel()

        n = len(y_true)
        if n == 0:
            return np.array([0, 1]), np.array([0, 0])

        # Sort by predicted uplift descending
        order = np.argsort(-predictions)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        # Total counts
        n_t = np.sum(treatment)  # Total treated
        n_c = n - n_t             # Total control

        if n_t == 0 or n_c == 0:
            return np.array([0, 1]), np.array([0, 0])

        # Percentile points to evaluate
        percentiles = np.linspace(0, 1, n_bins + 1)
        qini_y = []

        for p in percentiles:
            k = int(p * n)
            if k == 0:
                qini_y.append(0.0)
                continue

            # Get top-k units
            y_k = y_sorted[:k]
            t_k = t_sorted[:k]

            # Conversions in treated and control within top-k
            n_t_k = np.sum(t_k)      # Treated in top-k
            n_c_k = k - n_t_k         # Control in top-k

            if n_t_k > 0 and n_c_k > 0:
                # Response rates
                response_t = np.sum(y_k[t_k == 1]) / n_t_k
                response_c = np.sum(y_k[t_k == 0]) / n_c_k

                # Qini: cumulative uplift scaled by population fraction
                # This is the incremental number of conversions from treatment
                qini = (response_t - response_c) * (n_t_k + n_c_k) / n
            elif n_t_k > 0:
                # Only treated in top-k
                response_t = np.sum(y_k[t_k == 1]) / n_t_k
                qini = response_t * n_t_k / n
            else:
                # Only control in top-k
                qini = 0.0

            qini_y.append(qini)

        # Cumulative sum to get cumulative Qini
        qini_y = np.cumsum(qini_y)

        return percentiles, np.array(qini_y)

    @staticmethod
    def compute_cumulative_gain(
        y_true: np.ndarray,
        treatment: np.ndarray,
        predictions: np.ndarray,
        n_bins: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative gain curve (alternative Qini formulation).

        This is sometimes preferred as it's easier to interpret:
        shows the cumulative number of incremental conversions.

        Args:
            y_true: Binary outcome
            treatment: Binary treatment indicator
            predictions: Predicted uplift
            n_bins: Number of points

        Returns:
            Tuple of (percentiles, cumulative_gain)
        """
        y_true = np.asarray(y_true).ravel()
        treatment = np.asarray(treatment).ravel()
        predictions = np.asarray(predictions).ravel()

        n = len(y_true)
        order = np.argsort(-predictions)
        y_sorted = y_true[order]
        t_sorted = treatment[order]

        # Overall response rates
        n_t = np.sum(treatment)
        n_c = n - n_t

        if n_t == 0 or n_c == 0:
            return np.array([0, 1]), np.array([0, 0])

        overall_response_t = np.sum(y_true[treatment == 1]) / n_t
        overall_response_c = np.sum(y_true[treatment == 0]) / n_c

        percentiles = np.linspace(0, 1, n_bins + 1)
        gains = [0.0]

        for i, p in enumerate(percentiles[1:], 1):
            k = int(p * n)
            y_k = y_sorted[:k]
            t_k = t_sorted[:k]

            # Cumulative conversions
            treated_conversions = np.sum(y_k[t_k == 1])
            control_conversions = np.sum(y_k[t_k == 0])

            n_t_k = np.sum(t_k)
            n_c_k = k - n_t_k

            # Expected if random targeting
            expected_t = n_t_k * overall_response_t
            expected_c = n_c_k * overall_response_c

            # Gain over random
            gain_t = treated_conversions - expected_t
            gain_c = control_conversions - expected_c

            gains.append(gain_t - gain_c)

        return percentiles, np.array(gains)

    @staticmethod
    def compute_auuc(
        qini_x: np.ndarray,
        qini_y: np.ndarray,
        normalize: bool = True,
    ) -> float:
        """
        Compute Area Under Uplift Curve.

        A higher AUUC indicates better uplift model performance.
        Can be normalized to [0, 1] range for comparison.

        Args:
            qini_x: X coordinates from qini curve
            qini_y: Y coordinates from qini curve
            normalize: Whether to normalize by the area of random

        Returns:
            AUUC score
        """
        # Area under model curve
        auuc = np.trapz(qini_y, qini_x)

        if normalize:
            # Area under random (diagonal) baseline
            random_auuc = qini_y[-1] * 0.5
            if random_auuc > 0:
                # Normalized AUUC: how much better than random
                auuc = (auuc - random_auuc) / random_auuc
            else:
                auuc = 0.0

        return float(auuc)

    @staticmethod
    def compute_oracle_mse(
        predicted_cate: np.ndarray,
        true_cate: np.ndarray,
    ) -> float:
        """
        MSE between predicted and oracle ground truth CATE.

        Only usable with synthetic data where ground truth is available.

        Args:
            predicted_cate: Model's CATE predictions
            true_cate: Ground truth CATE (from oracle)

        Returns:
            Mean squared error
        """
        predicted = np.asarray(predicted_cate).ravel()
        true = np.asarray(true_cate).ravel()

        return float(np.mean((predicted - true) ** 2))

    @staticmethod
    def compute_oracle_correlation(
        predicted_cate: np.ndarray,
        true_cate: np.ndarray,
    ) -> float:
        """
        Pearson correlation between predicted and true CATE.

        High correlation indicates the model correctly ranks individuals
        by treatment effect, even if the scale is different.

        Args:
            predicted_cate: Model's CATE predictions
            true_cate: Ground truth CATE

        Returns:
            Correlation coefficient
        """
        predicted = np.asarray(predicted_cate).ravel()
        true = np.asarray(true_cate).ravel()

        corr = np.corrcoef(predicted, true)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    @staticmethod
    def compute_uplift_by_percentile(
        y_true: np.ndarray,
        treatment: np.ndarray,
        predictions: np.ndarray,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """
        Compute response rates and uplift by predicted uplift percentile.

        Useful for understanding how uplift varies across the population
        and validating model calibration.

        Args:
            y_true: Binary outcome
            treatment: Binary treatment indicator
            predictions: Predicted uplift
            n_bins: Number of percentile bins

        Returns:
            DataFrame with columns: percentile, n_treated, n_control,
            response_treated, response_control, observed_uplift, predicted_uplift
        """
        y_true = np.asarray(y_true).ravel()
        treatment = np.asarray(treatment).ravel()
        predictions = np.asarray(predictions).ravel()

        # Create percentile bins
        percentiles = np.percentile(predictions, np.linspace(0, 100, n_bins + 1))
        bins = np.digitize(predictions, percentiles[1:-1])

        results = []
        for i in range(n_bins):
            mask = bins == i
            if mask.sum() == 0:
                continue

            y_bin = y_true[mask]
            t_bin = treatment[mask]
            p_bin = predictions[mask]

            treated_mask = t_bin == 1
            control_mask = t_bin == 0

            n_treated = treated_mask.sum()
            n_control = control_mask.sum()

            if n_treated > 0:
                response_treated = y_bin[treated_mask].mean()
            else:
                response_treated = np.nan

            if n_control > 0:
                response_control = y_bin[control_mask].mean()
            else:
                response_control = np.nan

            if n_treated > 0 and n_control > 0:
                observed_uplift = response_treated - response_control
            else:
                observed_uplift = np.nan

            results.append({
                "percentile": (i + 1) * (100 // n_bins),
                "n_treated": n_treated,
                "n_control": n_control,
                "response_treated": response_treated,
                "response_control": response_control,
                "observed_uplift": observed_uplift,
                "predicted_uplift": p_bin.mean(),
            })

        return pd.DataFrame(results)

    @staticmethod
    def compute_ate(
        y_true: np.ndarray,
        treatment: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute naive Average Treatment Effect.

        Simple difference in means - biased if treatment is confounded,
        but useful as a baseline.

        Args:
            y_true: Outcome variable
            treatment: Binary treatment indicator

        Returns:
            Tuple of (ATE, standard_error)
        """
        y_true = np.asarray(y_true).ravel()
        treatment = np.asarray(treatment).ravel()

        treated_outcomes = y_true[treatment == 1]
        control_outcomes = y_true[treatment == 0]

        ate = treated_outcomes.mean() - control_outcomes.mean()

        # Standard error (assuming independent samples)
        se = np.sqrt(
            treated_outcomes.var() / len(treated_outcomes)
            + control_outcomes.var() / len(control_outcomes)
        )

        return float(ate), float(se)

    @staticmethod
    def compute_policy_value(
        y_true: np.ndarray,
        treatment: np.ndarray,
        predictions: np.ndarray,
        threshold: float = 0.0,
        cost_per_treatment: float = 0.0,
    ) -> dict:
        """
        Compute value of targeting policy based on predictions.

        Simulates targeting users with predicted CATE above threshold
        and computes expected value.

        Args:
            y_true: Outcome variable
            treatment: Actual treatment indicator
            predictions: Predicted CATE
            threshold: Only target users with CATE > threshold
            cost_per_treatment: Cost of applying treatment

        Returns:
            Dict with policy metrics
        """
        y_true = np.asarray(y_true).ravel()
        treatment = np.asarray(treatment).ravel()
        predictions = np.asarray(predictions).ravel()

        # Who would we target?
        would_target = predictions > threshold

        # Among actually treated who were targeted
        targeted_treated = would_target & (treatment == 1)
        targeted_control = would_target & (treatment == 0)

        # Response rates
        if targeted_treated.sum() > 0:
            response_targeted_treated = y_true[targeted_treated].mean()
        else:
            response_targeted_treated = 0

        if targeted_control.sum() > 0:
            response_targeted_control = y_true[targeted_control].mean()
        else:
            response_targeted_control = 0

        # Estimated incremental value per targeted user
        incremental_value = response_targeted_treated - response_targeted_control
        net_value = incremental_value - cost_per_treatment

        return {
            "n_targeted": would_target.sum(),
            "pct_targeted": would_target.mean() * 100,
            "incremental_response": incremental_value,
            "net_value_per_user": net_value,
            "total_expected_value": net_value * would_target.sum(),
        }
