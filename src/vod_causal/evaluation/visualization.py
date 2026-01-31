"""
Visualization utilities for uplift analysis.

Provides plotting functions for Qini curves, CATE distributions,
and other diagnostic visualizations for causal models.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_qini_curve(
    qini_x: np.ndarray,
    qini_y: np.ndarray,
    random_baseline: bool = True,
    perfect_model: Optional[np.ndarray] = None,
    title: str = "Qini Curve",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot Qini curve with optional random baseline.

    The Qini curve shows cumulative uplift as we target increasingly
    larger fractions of the population. A good model should show
    significant lift above the random baseline.

    Args:
        qini_x: X coordinates (fraction of population)
        qini_y: Y coordinates (cumulative uplift)
        random_baseline: Whether to show random baseline
        perfect_model: Optional perfect model curve for comparison
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Model curve
    ax.plot(qini_x, qini_y, "b-", linewidth=2, label="Model")
    ax.fill_between(qini_x, qini_y, alpha=0.2)

    # Random baseline (diagonal from origin to final value)
    if random_baseline:
        ax.plot([0, 1], [0, qini_y[-1]], "r--", linewidth=1.5, label="Random")

    # Perfect model curve
    if perfect_model is not None:
        ax.plot(qini_x, perfect_model, "g:", linewidth=1.5, label="Perfect")

    ax.set_xlabel("Fraction of Population Targeted", fontsize=12)
    ax.set_ylabel("Cumulative Uplift", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig


def plot_cate_distribution(
    predicted_cate: np.ndarray,
    true_cate: Optional[np.ndarray] = None,
    title: str = "CATE Distribution",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot distribution of predicted (and true) CATE.

    Useful for understanding:
    - The range of treatment effects
    - Whether the model captures heterogeneity
    - Calibration (when true CATE available)

    Args:
        predicted_cate: Model's CATE predictions
        true_cate: Optional ground truth CATE
        title: Plot title
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Predicted CATE distribution
    ax.hist(
        predicted_cate,
        bins=50,
        alpha=0.7,
        label="Predicted CATE",
        color="blue",
        density=True,
    )

    # True CATE distribution (if available)
    if true_cate is not None:
        ax.hist(
            true_cate,
            bins=50,
            alpha=0.5,
            label="True CATE",
            color="green",
            density=True,
        )

    ax.axvline(
        x=np.mean(predicted_cate),
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"Predicted Mean: {np.mean(predicted_cate):.3f}",
    )

    if true_cate is not None:
        ax.axvline(
            x=np.mean(true_cate),
            color="green",
            linestyle="--",
            linewidth=1.5,
            label=f"True Mean: {np.mean(true_cate):.3f}",
        )

    ax.set_xlabel("CATE", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cate_calibration(
    predicted_cate: np.ndarray,
    true_cate: np.ndarray,
    n_bins: int = 10,
    title: str = "CATE Calibration",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Plot calibration of predicted vs true CATE.

    Perfect calibration means predicted CATE equals true CATE on average.
    Points should lie on the diagonal.

    Args:
        predicted_cate: Model's CATE predictions
        true_cate: Ground truth CATE
        n_bins: Number of bins for averaging
        title: Plot title
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Bin by predicted CATE
    percentiles = np.percentile(predicted_cate, np.linspace(0, 100, n_bins + 1))
    bins = np.digitize(predicted_cate, percentiles[1:-1])

    bin_means_pred = []
    bin_means_true = []
    bin_stds_true = []

    for i in range(n_bins):
        mask = bins == i
        if mask.sum() > 0:
            bin_means_pred.append(predicted_cate[mask].mean())
            bin_means_true.append(true_cate[mask].mean())
            bin_stds_true.append(true_cate[mask].std() / np.sqrt(mask.sum()))

    # Scatter with error bars
    ax.errorbar(
        bin_means_pred,
        bin_means_true,
        yerr=bin_stds_true,
        fmt="o",
        markersize=8,
        capsize=4,
        label="Binned Average",
    )

    # Perfect calibration line
    min_val = min(min(bin_means_pred), min(bin_means_true))
    max_val = max(max(bin_means_pred), max(bin_means_true))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5,
        label="Perfect Calibration",
    )

    # Correlation
    corr = np.corrcoef(predicted_cate, true_cate)[0, 1]
    ax.text(
        0.05, 0.95,
        f"Correlation: {corr:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
    )

    ax.set_xlabel("Predicted CATE", fontsize=12)
    ax.set_ylabel("True CATE", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_uplift_by_percentile(
    uplift_df: pd.DataFrame,
    title: str = "Uplift by Predicted Percentile",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot observed vs predicted uplift by percentile.

    Args:
        uplift_df: DataFrame from compute_uplift_by_percentile()
        title: Plot title
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax.get_figure()
        axes = [ax, ax.twinx()]

    # Left plot: Response rates by treatment group
    ax1 = axes[0]
    x = uplift_df["percentile"]

    ax1.plot(x, uplift_df["response_treated"], "b-o", label="Treated", markersize=6)
    ax1.plot(x, uplift_df["response_control"], "r-s", label="Control", markersize=6)

    ax1.set_xlabel("Predicted Uplift Percentile", fontsize=12)
    ax1.set_ylabel("Response Rate", fontsize=12)
    ax1.set_title("Response Rates by Percentile", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Observed vs Predicted Uplift
    ax2 = axes[1]

    ax2.bar(x - 1.5, uplift_df["observed_uplift"], width=3, alpha=0.7, label="Observed")
    ax2.plot(x, uplift_df["predicted_uplift"], "g-^", label="Predicted", markersize=8)

    ax2.set_xlabel("Predicted Uplift Percentile", fontsize=12)
    ax2.set_ylabel("Uplift", fontsize=12)
    ax2.set_title("Observed vs Predicted Uplift", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    return fig


def plot_propensity_distribution(
    propensity: np.ndarray,
    treatment: np.ndarray,
    title: str = "Propensity Score Distribution",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot propensity score distribution by treatment group.

    Used to check overlap/positivity assumption. Good overlap means
    both treated and control groups have similar propensity distributions.

    Args:
        propensity: Propensity scores
        treatment: Treatment indicators
        title: Plot title
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    treatment = np.asarray(treatment).ravel()

    # Separate by treatment group
    prop_treated = propensity[treatment == 1]
    prop_control = propensity[treatment == 0]

    # Plot histograms
    ax.hist(
        prop_control,
        bins=50,
        alpha=0.6,
        label=f"Control (n={len(prop_control)})",
        color="blue",
        density=True,
    )
    ax.hist(
        prop_treated,
        bins=50,
        alpha=0.6,
        label=f"Treated (n={len(prop_treated)})",
        color="orange",
        density=True,
    )

    ax.set_xlabel("Propensity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    plt.tight_layout()
    return fig


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    top_k: int = 15,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importances.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        title: Plot title
        top_k: Number of top features to show
        ax: Matplotlib axes
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Get top k features
    importance_col = [c for c in feature_importance_df.columns if "importance" in c.lower()][0]
    plot_df = feature_importance_df.nlargest(top_k, importance_col)

    # Create horizontal bar chart
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos, plot_df[importance_col], align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["feature"])
    ax.invert_yaxis()  # Top feature at top

    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def create_evaluation_dashboard(
    y_true: np.ndarray,
    treatment: np.ndarray,
    predicted_cate: np.ndarray,
    true_cate: Optional[np.ndarray] = None,
    propensity: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (16, 12),
) -> plt.Figure:
    """
    Create a comprehensive evaluation dashboard.

    Combines multiple visualizations into a single figure:
    - Qini curve
    - CATE distribution
    - Calibration plot (if true CATE available)
    - Propensity distribution (if available)

    Args:
        y_true: Binary outcomes
        treatment: Treatment indicators
        predicted_cate: Predicted CATE
        true_cate: Optional ground truth CATE
        propensity: Optional propensity scores
        figsize: Figure size

    Returns:
        Matplotlib Figure
    """
    from .metrics import UpliftMetrics

    n_plots = 2 + (1 if true_cate is not None else 0) + (1 if propensity is not None else 0)
    n_cols = 2
    n_rows = (n_plots + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    plot_idx = 0

    # 1. Qini Curve
    qini_x, qini_y = UpliftMetrics.compute_qini_curve(y_true, treatment, predicted_cate)
    auuc = UpliftMetrics.compute_auuc(qini_x, qini_y)
    plot_qini_curve(qini_x, qini_y, ax=axes[plot_idx], title=f"Qini Curve (AUUC: {auuc:.4f})")
    plot_idx += 1

    # 2. CATE Distribution
    plot_cate_distribution(predicted_cate, true_cate, ax=axes[plot_idx])
    plot_idx += 1

    # 3. Calibration (if true CATE available)
    if true_cate is not None:
        plot_cate_calibration(predicted_cate, true_cate, ax=axes[plot_idx])
        plot_idx += 1

    # 4. Propensity Distribution (if available)
    if propensity is not None:
        plot_propensity_distribution(propensity, treatment, ax=axes[plot_idx])
        plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Causal Model Evaluation Dashboard", fontsize=16, y=1.02)
    plt.tight_layout()
    return fig
