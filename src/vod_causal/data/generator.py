"""
Synthetic data generation pipeline.

Generates a realistic relational VOD dataset with hidden causal effects
that can be used to validate causal inference models. The dataset mimics
real VOD platform logs with users, titles, promotional treatments, and outcomes.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .oracle import CausalOracle
from .schemas import InteractionOutcome, TitleMetadata, TreatmentLog, UserMetadata


class VODSyntheticData:
    """
    Generates realistic relational VOD dataset with hidden causal effects.

    Creates 4 DataFrames that mimic real VOD platform data:
    - titles_metadata: Content information (500 titles default)
    - users_metadata: Audience information (10,000 users default)
    - treatment_log: Promotional interventions (~15% treated)
    - interaction_outcomes: Target variables (rentals, revenue, watch time)

    The dataset includes a hidden causal structure (via CausalOracle) that allows
    for ground-truth validation of causal inference models. The true CATE is
    stored for each interaction but would not be available in real data.

    Example:
        >>> generator = VODSyntheticData(n_users=1000, n_titles=100)
        >>> data = generator.generate_all()
        >>> print(data['users_metadata'].shape)
        (1000, 6)

    Attributes:
        n_users: Number of users to generate
        n_titles: Number of titles to generate
        n_interactions: Number of user-title interactions
        treatment_probability: Base probability of treatment assignment
        oracle: CausalOracle instance for ground truth
        rng: NumPy random generator
    """

    # Categorical value pools
    GENRES: List[str] = [
        "Action", "Comedy", "Drama", "Sci-Fi", "Horror",
        "Documentary", "Romance", "Thriller"
    ]
    REGIONS: List[str] = ["US", "EU", "APAC", "LATAM"]
    DEVICES: List[str] = ["Mobile", "Desktop", "SmartTV", "Tablet"]
    DIRECTORS: List[str] = [
        "Director_A", "Director_B", "Director_C", "Director_D", "Director_E",
        "Director_F", "Director_G", "Director_H", "Director_I", "Director_J",
        "Director_K", "Director_L", "Director_M", "Director_N", "Director_O",
    ]

    def __init__(
        self,
        n_users: int = 10_000,
        n_titles: int = 500,
        n_interactions: int = 100_000,
        treatment_probability: float = 0.15,
        cold_start_fraction: float = 0.1,
        embedding_dim: int = 32,
        seed: int = 42,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            n_users: Number of users to generate
            n_titles: Number of titles to generate
            n_interactions: Number of user-title interactions to generate
            treatment_probability: Base probability of receiving a discount offer
            cold_start_fraction: Fraction of titles to mark as cold-start
            embedding_dim: Dimension of title embeddings
            seed: Random seed for reproducibility
        """
        self.n_users = n_users
        self.n_titles = n_titles
        self.n_interactions = n_interactions
        self.treatment_probability = treatment_probability
        self.cold_start_fraction = cold_start_fraction
        self.embedding_dim = embedding_dim
        self.seed = seed

        self.oracle = CausalOracle(seed=seed)
        self.rng = np.random.default_rng(seed)

        # Campaign IDs for the treatment log
        self.campaign_ids = [f"CAMP_{i:03d}" for i in range(10)]

    def generate_titles(self) -> pd.DataFrame:
        """
        Generate titles_metadata table.

        Creates a catalog of VOD titles with genres, directors, release years,
        and simulated embeddings. A fraction are marked as cold-start.

        Returns:
            DataFrame with columns: title_id, genre, director, release_year,
            warm_embedding, is_cold_start, base_popularity
        """
        titles_data = []

        for i in range(self.n_titles):
            title_id = f"TITLE_{i:05d}"

            # Sample categorical features with realistic distributions
            genre = self.rng.choice(self.GENRES, p=[0.2, 0.15, 0.15, 0.12, 0.08, 0.08, 0.12, 0.10])
            director = self.rng.choice(self.DIRECTORS)

            # Release year with recency bias
            release_year = int(2024 - self.rng.exponential(5))
            release_year = max(1990, min(2024, release_year))

            # Generate embedding (simulate learned latent factors)
            # Embeddings have structure based on genre
            base_embedding = self.rng.normal(0, 1, self.embedding_dim)
            genre_idx = self.GENRES.index(genre)
            base_embedding[genre_idx % self.embedding_dim] += 1.0  # Genre signal
            warm_embedding = base_embedding / np.linalg.norm(base_embedding)

            # Cold start flag (newer titles with less data)
            is_cold_start = i >= self.n_titles * (1 - self.cold_start_fraction)

            # Base popularity (power law distribution)
            base_popularity = 0.1 + 0.9 * (1 - (i / self.n_titles) ** 0.5)
            base_popularity = np.clip(base_popularity + self.rng.normal(0, 0.1), 0.05, 0.95)

            titles_data.append({
                "title_id": title_id,
                "genre": genre,
                "director": director,
                "release_year": release_year,
                "warm_embedding": warm_embedding,
                "is_cold_start": is_cold_start,
                "base_popularity": base_popularity,
            })

        return pd.DataFrame(titles_data)

    def generate_users(self) -> pd.DataFrame:
        """
        Generate users_metadata table.

        Creates a user base with demographic and behavioral features that
        influence promotional response.

        Returns:
            DataFrame with columns: user_id, subscription_tenure_months,
            geo_region, device_type, avg_daily_watch_time, price_sensitivity
        """
        users_data = []

        for i in range(self.n_users):
            user_id = f"USER_{i:06d}"

            # Subscription tenure follows exponential distribution (many new, few old)
            subscription_tenure_months = int(self.rng.exponential(18))
            subscription_tenure_months = max(1, min(120, subscription_tenure_months))

            # Geographic distribution (weighted by market size)
            geo_region = self.rng.choice(self.REGIONS, p=[0.4, 0.3, 0.2, 0.1])

            # Device distribution
            device_type = self.rng.choice(self.DEVICES, p=[0.35, 0.25, 0.30, 0.10])

            # Watch time varies by device and region
            base_watch_time = {
                "SmartTV": 90,
                "Mobile": 45,
                "Desktop": 60,
                "Tablet": 75,
            }[device_type]
            avg_daily_watch_time = max(5, base_watch_time + self.rng.normal(0, 20))

            # Price sensitivity (latent, not directly observable)
            # Correlated with region and tenure
            region_sensitivity = {"US": 0.4, "EU": 0.5, "APAC": 0.6, "LATAM": 0.7}[geo_region]
            tenure_effect = -0.1 * np.log1p(subscription_tenure_months / 12)
            price_sensitivity = np.clip(
                region_sensitivity + tenure_effect + self.rng.normal(0, 0.15),
                0.1, 0.95
            )

            users_data.append({
                "user_id": user_id,
                "subscription_tenure_months": subscription_tenure_months,
                "geo_region": geo_region,
                "device_type": device_type,
                "avg_daily_watch_time": avg_daily_watch_time,
                "price_sensitivity": price_sensitivity,
            })

        return pd.DataFrame(users_data)

    def _sample_treatment(
        self,
        user_row: pd.Series,
        title_row: pd.Series,
    ) -> Tuple[bool, float, str]:
        """
        Sample treatment assignment for a user-title pair.

        Treatment assignment is not purely random - it's influenced by
        user and item features (confounding). This makes the causal
        inference problem more realistic.

        Args:
            user_row: User metadata
            title_row: Title metadata

        Returns:
            Tuple of (is_treated, offered_price, campaign_id)
        """
        # Base probability of treatment
        base_prob = self.treatment_probability

        # Confounding: higher treatment probability for...
        # - Price-sensitive users (platform targets them)
        price_sensitivity = user_row["price_sensitivity"]
        base_prob += 0.1 * (price_sensitivity - 0.5)

        # - Newer titles (need promotion)
        if title_row["is_cold_start"]:
            base_prob += 0.1

        # - Less popular titles
        base_prob += 0.05 * (1 - title_row["base_popularity"])

        # NEW CONFOUNDING: "Peak Demand" bias
        # Simulate that high-demand users (high watch time) rarely get valid discounts
        # or get higher "discounted" prices.
        avg_watch = user_row["avg_daily_watch_time"]
        if avg_watch > 80:
             # High usage users get treated LESS often
            base_prob -= 0.15

        # Sample treatment
        is_treated = self.rng.random() < np.clip(base_prob, 0.05, 0.5)
        
        base_price = 4.99

        if is_treated:
            # Continuous pricing:
            # Random discount between 5% and 40%
            # But influenced by watch time (confounding)
            
            # Base discount
            discount_pct = self.rng.uniform(0.05, 0.40)
            
            # Confounding: High watch time users get smaller discounts
            if avg_watch > 60:
                discount_pct *= 0.7  # Reduce discount

            # Calculate price
            offered_price = base_price * (1 - discount_pct)
            
            # Round to nice numbers like X.99 or X.49
            offered_price = np.round(offered_price * 2) / 2 - 0.01
            
            campaign_id = self.rng.choice(self.campaign_ids)
        else:
            offered_price = base_price
            campaign_id = "NO_TREATMENT"

        return is_treated, float(max(0.99, offered_price)), campaign_id

    def generate_interactions(
        self,
        users_df: pd.DataFrame,
        titles_df: pd.DataFrame,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate treatment_log and interaction_outcomes tables.

        Uses the CausalOracle to determine ground truth outcomes based on
        the hidden causal structure.

        Args:
            users_df: Users metadata DataFrame
            titles_df: Titles metadata DataFrame
            start_date: Start of interaction period (default: 90 days ago)
            end_date: End of interaction period (default: now)

        Returns:
            Tuple of (treatment_log_df, outcomes_df)
        """
        if start_date is None:
            start_date = datetime(2024, 1, 1)
        if end_date is None:
            end_date = datetime(2024, 3, 31)

        # Convert to lookup-friendly format
        users_dict = users_df.set_index("user_id").to_dict("index")
        titles_dict = titles_df.set_index("title_id").to_dict("index")

        treatment_logs = []
        outcomes = []

        # Sample random user-title pairs
        user_ids = users_df["user_id"].values
        title_ids = titles_df["title_id"].values

        for _ in range(self.n_interactions):
            # Sample user and title (with popularity weighting for titles)
            user_id = self.rng.choice(user_ids)
            
            # Title selection weighted by popularity
            title_probs = titles_df["base_popularity"].values
            title_probs = title_probs / title_probs.sum()
            title_id = self.rng.choice(title_ids, p=title_probs)

            user_row = users_df[users_df["user_id"] == user_id].iloc[0]
            title_row = titles_df[titles_df["title_id"] == title_id].iloc[0]

            # Sample treatment assignment
            is_treated, offered_price, campaign_id = self._sample_treatment(
                user_row, title_row
            )

            # Generate timestamp
            time_delta = (end_date - start_date).total_seconds()
            random_seconds = self.rng.random() * time_delta
            timestamp = start_date + timedelta(seconds=random_seconds)

            # Build feature dicts for oracle
            user_features = {
                "geo_region": user_row["geo_region"],
                "tenure": user_row["subscription_tenure_months"],
                "device_type": user_row["device_type"],
                "avg_daily_watch_time": user_row["avg_daily_watch_time"],
                "price_sensitivity": user_row["price_sensitivity"],
            }
            item_features = {
                "genre": title_row["genre"],
                "popularity": title_row["base_popularity"],
                "base_price": 4.99,
            }

            # Get outcomes from oracle
            outcome = self.oracle.compute_observed_outcome(
                user_features=user_features,
                item_features=item_features,
                is_treated=is_treated,
                offered_price=offered_price,
                return_revenue=True,
            )

            # Compute watch duration
            watch_duration = self.oracle.compute_watch_duration(
                user_features, item_features, outcome["did_rent"]
            )

            # Create treatment log entry
            treatment_logs.append({
                "user_id": user_id,
                "title_id": title_id,
                "offered_price": offered_price,
                "campaign_id": campaign_id,
                "timestamp": timestamp,
            })

            # Create outcome entry
            outcomes.append({
                "user_id": user_id,
                "title_id": title_id,
                "did_rent": outcome["did_rent"],
                "revenue_generated": outcome["revenue"],
                "watch_duration_minutes": watch_duration,
                "base_price": 4.99,
                "base_price": 4.99,
                # Ground truth for validation (not available in real data!)
                "true_elasticity": outcome["true_elasticity"],
                "demand_at_price": outcome["demand_at_price"],
            })

        treatment_df = pd.DataFrame(treatment_logs)
        outcomes_df = pd.DataFrame(outcomes)

        return treatment_df, outcomes_df

    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all 4 tables and return as dictionary.

        This is the main entry point for data generation. Creates a complete
        synthetic VOD dataset ready for causal inference experimentation.

        Returns:
            Dict with keys:
                - 'titles_metadata': Content information
                - 'users_metadata': Audience information
                - 'treatment_log': Promotional interventions
                - 'interaction_outcomes': Target variables
                - 'summary': Dict with dataset statistics
        """
        # Generate entities
        titles_df = self.generate_titles()
        users_df = self.generate_users()

        # Generate interactions
        treatment_df, outcomes_df = self.generate_interactions(users_df, titles_df)

        # Compute summary statistics
        summary = {
            "n_users": len(users_df),
            "n_titles": len(titles_df),
            "n_interactions": len(outcomes_df),
            "treatment_rate": treatment_df["campaign_id"].apply(lambda x: x != "NO_TREATMENT").mean(),
            "conversion_rate": outcomes_df["did_rent"].mean(),
            "avg_revenue": outcomes_df["revenue_generated"].mean(),
            "avg_price": treatment_df["offered_price"].mean(),
            "cold_start_titles": titles_df["is_cold_start"].sum(),
        }

        return {
            "titles_metadata": titles_df,
            "users_metadata": users_df,
            "treatment_log": treatment_df,
            "interaction_outcomes": outcomes_df,
            "summary": summary,
        }

    def create_modeling_dataset(
        self,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Create a flat dataset ready for modeling.

        Merges all tables and creates a single DataFrame with all features,
        treatment indicator, and outcomes.

        Args:
            data: Output from generate_all(). If None, generates new data.

        Returns:
            DataFrame with merged features, treatment, and outcomes
        """
        if data is None:
            data = self.generate_all()

        # Merge treatment and outcomes
        df = data["treatment_log"].merge(
            data["interaction_outcomes"],
            on=["user_id", "title_id"],
            how="inner"
        )

        # Merge user features
        df = df.merge(
            data["users_metadata"],
            on="user_id",
            how="left"
        )

        # Merge title features (excluding embedding for now)
        title_cols = ["title_id", "genre", "director", "release_year", 
                      "is_cold_start", "base_popularity"]
        df = df.merge(
            data["titles_metadata"][title_cols],
            on="title_id",
            how="left"
        )

        return df
