"""
Policy ranking and simulation.

Translates CATE predictions into business decisions using the
Counterfactual Value Estimator (CVE) logic. Generates promotional
recommendations optimized for ROI.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PolicyRanker:
    """
    Translate CATE predictions into business decisions.

    Implements Counterfactual Value Estimator (CVE) logic and generates
    actionable promotional recommendations. Ranks user-title pairs by
    expected incremental value, considering discount costs.

    The ranking algorithm:
    1. Predict CATE (uplift) for each user-title pair
    2. Apply ROI constraint: only recommend if CATE > cost_of_discount
    3. Rank by net expected value (CATE - cost)
    4. Return top-K recommendations per user

    Example:
        >>> ranker = PolicyRanker(discount_cost=0.10)
        >>> recommendations = ranker.rank(user_title_pairs, predicted_uplift, top_k=5)

    Attributes:
        discount_cost: Cost of offering a discount (as fraction of base price)
        min_expected_roi: Minimum ROI threshold for recommending
    """

    def __init__(
        self,
        discount_cost: float = 0.10,
        min_expected_roi: float = 0.0,
        base_price: float = 4.99,
    ):
        """
        Initialize policy ranker.

        Args:
            discount_cost: Cost of giving a discount (as fraction of base price)
            min_expected_roi: Minimum expected ROI to include in recommendations
            base_price: Default base price for titles
        """
        self.discount_cost = discount_cost
        self.min_expected_roi = min_expected_roi
        self.base_price = base_price

    def rank(
        self,
        user_title_pairs: pd.DataFrame,
        predicted_uplift: np.ndarray,
        top_k: int = 5,
        include_negative: bool = False,
    ) -> pd.DataFrame:
        """
        Generate top-K title recommendations per user.

        Algorithm:
            1. Calculate expected ROI: uplift * base_price - discount_cost
            2. Filter by minimum ROI threshold
            3. Rank by expected ROI within each user
            4. Return top K per user

        Args:
            user_title_pairs: DataFrame with user_id, title_id, and features
            predicted_uplift: CATE predictions for each pair
            top_k: Number of recommendations per user
            include_negative: Whether to include negative ROI items

        Returns:
            DataFrame with user_id, title_id, predicted_uplift, expected_roi, rank
        """
        predicted_uplift = np.asarray(predicted_uplift).ravel()

        df = user_title_pairs.copy()
        df["predicted_uplift"] = predicted_uplift

        # Calculate expected value
        # Uplift is in probability units, so expected revenue = uplift * base_price
        df["expected_revenue_uplift"] = df["predicted_uplift"] * self.base_price

        # Net ROI after discount cost
        df["expected_roi"] = df["expected_revenue_uplift"] - self.discount_cost

        # Apply ROI constraint (unless including negative)
        if not include_negative:
            df = df[df["expected_roi"] > self.min_expected_roi].copy()

        if len(df) == 0:
            return pd.DataFrame(columns=[
                "user_id", "title_id", "predicted_uplift", "expected_roi", "rank"
            ])

        # Rank by ROI within each user
        df = df.sort_values(
            ["user_id", "expected_roi"],
            ascending=[True, False]
        )

        # Add rank within user
        df["rank"] = df.groupby("user_id").cumcount() + 1

        # Take top K per user
        recommendations = df[df["rank"] <= top_k].copy()

        # Select output columns
        output_cols = ["user_id", "title_id", "predicted_uplift", "expected_roi", "rank"]
        if "genre" in recommendations.columns:
            output_cols.append("genre")

        return recommendations[output_cols].reset_index(drop=True)

    def rank_with_diversity(
        self,
        user_title_pairs: pd.DataFrame,
        predicted_uplift: np.ndarray,
        top_k: int = 5,
        diversity_column: str = "genre",
        max_per_category: int = 2,
    ) -> pd.DataFrame:
        """
        Generate diverse recommendations using MMR-style re-ranking.

        Ensures recommendations are diverse across categories (e.g., genres)
        while still prioritizing high uplift items.

        Args:
            user_title_pairs: DataFrame with user_id, title_id, features
            predicted_uplift: CATE predictions
            top_k: Number of recommendations per user
            diversity_column: Column to diversify over
            max_per_category: Maximum items per category

        Returns:
            DataFrame with diverse recommendations
        """
        predicted_uplift = np.asarray(predicted_uplift).ravel()

        df = user_title_pairs.copy()
        df["predicted_uplift"] = predicted_uplift
        df["expected_roi"] = df["predicted_uplift"] * self.base_price - self.discount_cost

        # Filter by ROI
        df = df[df["expected_roi"] > self.min_expected_roi].copy()

        if len(df) == 0 or diversity_column not in df.columns:
            return self.rank(user_title_pairs, predicted_uplift, top_k)

        # MMR-style selection per user
        selected = []

        for user_id, user_df in df.groupby("user_id"):
            user_recs = []
            category_counts = {}

            # Sort by ROI
            user_df = user_df.sort_values("expected_roi", ascending=False)

            for _, row in user_df.iterrows():
                category = row[diversity_column]
                count = category_counts.get(category, 0)

                if count < max_per_category:
                    user_recs.append(row)
                    category_counts[category] = count + 1

                if len(user_recs) >= top_k:
                    break

            selected.extend(user_recs)

        result = pd.DataFrame(selected)
        if len(result) > 0:
            result["rank"] = result.groupby("user_id").cumcount() + 1

        return result.reset_index(drop=True)

    def simulate_policy(
        self,
        recommendations: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        treatment_col: str = "is_treated",
        outcome_col: str = "did_rent",
    ) -> Dict:
        """
        Simulate policy performance against actual outcomes.

        Compares model-recommended policy against actual outcomes and
        random targeting to estimate incremental value.

        Args:
            recommendations: Output from rank() method
            outcomes_df: Actual outcomes with user_id, title_id, treatment, outcome
            treatment_col: Column name for treatment indicator
            outcome_col: Column name for outcome

        Returns:
            Dict with simulation metrics
        """
        # Merge recommendations with outcomes
        rec_pairs = set(zip(recommendations["user_id"], recommendations["title_id"]))

        outcomes = outcomes_df.copy()
        outcomes["would_target"] = outcomes.apply(
            lambda r: (r["user_id"], r["title_id"]) in rec_pairs,
            axis=1
        )

        # Calculate metrics for targeted vs non-targeted
        targeted = outcomes[outcomes["would_target"]]
        not_targeted = outcomes[~outcomes["would_target"]]

        # Response rates
        if len(targeted) > 0:
            targeted_treated = targeted[targeted[treatment_col] == 1]
            targeted_control = targeted[targeted[treatment_col] == 0]

            if len(targeted_treated) > 0:
                targeted_response_t = targeted_treated[outcome_col].mean()
            else:
                targeted_response_t = 0

            if len(targeted_control) > 0:
                targeted_response_c = targeted_control[outcome_col].mean()
            else:
                targeted_response_c = 0

            targeted_uplift = targeted_response_t - targeted_response_c
        else:
            targeted_response_t = 0
            targeted_response_c = 0
            targeted_uplift = 0

        # Non-targeted baseline
        if len(not_targeted) > 0:
            not_targeted_treated = not_targeted[not_targeted[treatment_col] == 1]
            not_targeted_control = not_targeted[not_targeted[treatment_col] == 0]

            if len(not_targeted_treated) > 0:
                not_targeted_response_t = not_targeted_treated[outcome_col].mean()
            else:
                not_targeted_response_t = 0

            if len(not_targeted_control) > 0:
                not_targeted_response_c = not_targeted_control[outcome_col].mean()
            else:
                not_targeted_response_c = 0

            not_targeted_uplift = not_targeted_response_t - not_targeted_response_c
        else:
            not_targeted_uplift = 0

        # Incremental value calculation
        n_targeted = len(recommendations)
        expected_incremental_conversions = n_targeted * targeted_uplift
        expected_revenue = expected_incremental_conversions * self.base_price
        expected_cost = n_targeted * self.discount_cost
        expected_profit = expected_revenue - expected_cost

        return {
            "n_recommendations": n_targeted,
            "n_users_targeted": recommendations["user_id"].nunique(),
            "avg_recommendations_per_user": n_targeted / max(1, recommendations["user_id"].nunique()),
            "targeted_response_treated": targeted_response_t,
            "targeted_response_control": targeted_response_c,
            "targeted_uplift": targeted_uplift,
            "not_targeted_uplift": not_targeted_uplift,
            "expected_incremental_conversions": expected_incremental_conversions,
            "expected_revenue": expected_revenue,
            "expected_cost": expected_cost,
            "expected_profit": expected_profit,
            "roi_percent": (expected_profit / max(1e-6, expected_cost)) * 100,
        }

    def create_campaign_report(
        self,
        recommendations: pd.DataFrame,
        title_metadata: pd.DataFrame,
        user_metadata: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """
        Create a comprehensive campaign report for business stakeholders.

        Args:
            recommendations: Output from rank()
            title_metadata: Title information
            user_metadata: User information

        Returns:
            Dict with multiple summary DataFrames
        """
        # Determine columns to merge from title_metadata
        cols_to_merge = ["title_id"]
        for col in ["genre", "director"]:
            if col not in recommendations.columns and col in title_metadata.columns:
                cols_to_merge.append(col)
        
        # Merge with metadata
        if len(cols_to_merge) > 1:
            recs = recommendations.merge(
                title_metadata[cols_to_merge],
                on="title_id",
                how="left"
            )
        else:
            recs = recommendations.copy()
        # Determine columns to merge from user_metadata
        user_cols_to_merge = ["user_id"]
        for col in ["geo_region", "device_type"]:
            if col not in recs.columns and col in user_metadata.columns:
                user_cols_to_merge.append(col)

        if len(user_cols_to_merge) > 1:
            recs = recs.merge(
                user_metadata[user_cols_to_merge],
                on="user_id",
                how="left"
            )

        # Summary by genre
        genre_summary = recs.groupby("genre").agg({
            "title_id": "count",
            "predicted_uplift": "mean",
            "expected_roi": "mean",
        }).rename(columns={
            "title_id": "n_recommendations",
            "predicted_uplift": "avg_uplift",
            "expected_roi": "avg_roi",
        }).sort_values("n_recommendations", ascending=False)

        # Summary by region
        region_summary = recs.groupby("geo_region").agg({
            "user_id": "nunique",
            "title_id": "count",
            "predicted_uplift": "mean",
            "expected_roi": "mean",
        }).rename(columns={
            "user_id": "n_users",
            "title_id": "n_recommendations",
            "predicted_uplift": "avg_uplift",
            "expected_roi": "avg_roi",
        }).sort_values("n_users", ascending=False)

        # Top titles
        top_titles = recs.groupby("title_id").agg({
            "user_id": "count",
            "predicted_uplift": "mean",
            "expected_roi": "mean",
        }).rename(columns={
            "user_id": "n_recommended_to",
            "predicted_uplift": "avg_uplift",
            "expected_roi": "avg_roi",
        }).sort_values("n_recommended_to", ascending=False).head(20)

        # Overall summary
        overall = pd.DataFrame({
            "metric": [
                "Total Recommendations",
                "Unique Users",
                "Unique Titles",
                "Average Uplift",
                "Average ROI",
                "Total Expected Value",
            ],
            "value": [
                len(recommendations),
                recommendations["user_id"].nunique(),
                recommendations["title_id"].nunique(),
                recommendations["predicted_uplift"].mean(),
                recommendations["expected_roi"].mean(),
                recommendations["expected_roi"].sum(),
            ]
        })

        return {
            "overall": overall,
            "by_genre": genre_summary.reset_index(),
            "by_region": region_summary.reset_index(),
            "top_titles": top_titles.reset_index(),
        }

    def successive_k_item_ranking(
        self,
        user_title_pairs: pd.DataFrame,
        predicted_uplift: np.ndarray,
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[int, pd.DataFrame]:
        """
        Implement successive k-item ranking heuristic.

        Generates recommendations at multiple k values for A/B testing
        different recommendation intensities.

        Args:
            user_title_pairs: DataFrame with user_id, title_id
            predicted_uplift: CATE predictions
            k_values: List of k values to generate

        Returns:
            Dict mapping k -> recommendations DataFrame
        """
        results = {}
        for k in sorted(k_values):
            results[k] = self.rank(user_title_pairs, predicted_uplift, top_k=k)
        return results
