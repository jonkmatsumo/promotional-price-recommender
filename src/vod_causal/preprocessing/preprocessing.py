"""
Feature transformation pipeline.

Transforms raw VOD logs into high-dimensional feature sets suitable for
causal inference models. Implements:
- One-hot encoding for categorical features
- Cyclical timestamp encoding for seasonality
- Cold-start embedding via metadata averaging
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureTransformer:
    """
    Transform raw VOD logs into high-dimensional feature sets.

    Implements the feature engineering pipeline described in the
    "High-Dimensional Feature Design" section, including:
    - One-hot encoding for low-cardinality categoricals (genre, region, device)
    - Cyclical timestamp encoding (sine/cosine of hour/day) for seasonality
    - Cold-start metadata averaging for new titles
    - Numerical feature standardization

    Example:
        >>> transformer = FeatureTransformer()
        >>> transformer.fit(data_dict)
        >>> X = transformer.transform(data_dict)

    Attributes:
        categorical_encoder: Fitted OneHotEncoder for categoricals
        numerical_scaler: Fitted StandardScaler for numericals
        fitted: Whether the transformer has been fitted
        feature_names_: List of output feature names
    """

    # Columns to one-hot encode
    CATEGORICAL_COLUMNS: List[str] = ["genre", "geo_region", "device_type", "director"]

    # Columns to standardize
    NUMERICAL_COLUMNS: List[str] = [
        "subscription_tenure_months",
        "avg_daily_watch_time",
        "price_sensitivity",
        "release_year",
        "base_popularity",
    ]

    def __init__(
        self,
        handle_cold_start: bool = True,
        include_embeddings: bool = False,
        embedding_dim: int = 32,
    ):
        """
        Initialize the feature transformer.

        Args:
            handle_cold_start: Whether to apply cold-start handling for titles
            include_embeddings: Whether to include title embeddings in features
            embedding_dim: Dimension of title embeddings
        """
        self.handle_cold_start = handle_cold_start
        self.include_embeddings = include_embeddings
        self.embedding_dim = embedding_dim

        self.categorical_encoder: Optional[OneHotEncoder] = None
        self.numerical_scaler: Optional[StandardScaler] = None
        self.fitted = False
        self.feature_names_: List[str] = []

        # For cold-start handling: store average embeddings by genre-director
        self._embedding_averages: Dict[Tuple[str, str], np.ndarray] = {}

    def fit(
        self,
        data: Dict[str, pd.DataFrame],
        modeling_df: Optional[pd.DataFrame] = None,
    ) -> "FeatureTransformer":
        """
        Fit encoders and scalers on training data.

        Args:
            data: Dict with 'titles_metadata', 'users_metadata', etc.
            modeling_df: Pre-merged modeling DataFrame (optional)

        Returns:
            self for method chaining
        """
        # Create modeling DataFrame if not provided
        if modeling_df is None:
            modeling_df = self._create_modeling_df(data)

        # Fit categorical encoder
        cat_cols_present = [c for c in self.CATEGORICAL_COLUMNS if c in modeling_df.columns]
        if cat_cols_present:
            self.categorical_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore",
                drop=None,  # Keep all categories for interpretability
            )
            self.categorical_encoder.fit(modeling_df[cat_cols_present])

        # Fit numerical scaler
        num_cols_present = [c for c in self.NUMERICAL_COLUMNS if c in modeling_df.columns]
        if num_cols_present:
            self.numerical_scaler = StandardScaler()
            self.numerical_scaler.fit(modeling_df[num_cols_present])

        # Compute cold-start embedding averages
        if self.handle_cold_start and "titles_metadata" in data:
            self._compute_embedding_averages(data["titles_metadata"])

        # Build feature names
        self._build_feature_names(cat_cols_present, num_cols_present)

        self.fitted = True
        return self

    def transform(
        self,
        data: Dict[str, pd.DataFrame],
        modeling_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Transform raw data into feature matrix.

        Args:
            data: Dict with 'titles_metadata', 'users_metadata', etc.
            modeling_df: Pre-merged modeling DataFrame (optional)

        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            raise RuntimeError("FeatureTransformer must be fitted before transform")

        # Create modeling DataFrame if not provided
        if modeling_df is None:
            modeling_df = self._create_modeling_df(data)

        feature_parts = []

        # Transform categorical features
        cat_cols_present = [c for c in self.CATEGORICAL_COLUMNS if c in modeling_df.columns]
        if cat_cols_present and self.categorical_encoder is not None:
            cat_encoded = self.categorical_encoder.transform(modeling_df[cat_cols_present])
            cat_names = self.categorical_encoder.get_feature_names_out(cat_cols_present)
            cat_df = pd.DataFrame(cat_encoded, columns=cat_names, index=modeling_df.index)
            feature_parts.append(cat_df)

        # Transform numerical features
        num_cols_present = [c for c in self.NUMERICAL_COLUMNS if c in modeling_df.columns]
        if num_cols_present and self.numerical_scaler is not None:
            num_scaled = self.numerical_scaler.transform(modeling_df[num_cols_present])
            num_df = pd.DataFrame(
                num_scaled,
                columns=[f"{c}_scaled" for c in num_cols_present],
                index=modeling_df.index
            )
            feature_parts.append(num_df)

        # Extract cyclical timestamp features
        if "timestamp" in modeling_df.columns:
            time_features = self.encode_timestamp_cyclical(modeling_df["timestamp"])
            time_features.index = modeling_df.index
            feature_parts.append(time_features)

        # Handle embeddings if requested
        if self.include_embeddings and "titles_metadata" in data:
            embedding_df = self._extract_embeddings(modeling_df, data["titles_metadata"])
            embedding_df.index = modeling_df.index
            feature_parts.append(embedding_df)

        # Combine all features
        if feature_parts:
            X = pd.concat(feature_parts, axis=1)
        else:
            X = pd.DataFrame(index=modeling_df.index)

        return X

    def fit_transform(
        self,
        data: Dict[str, pd.DataFrame],
        modeling_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, modeling_df).transform(data, modeling_df)

    def encode_timestamp_cyclical(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Convert timestamps into cyclical features.

        Captures seasonal patterns by encoding time components as sine/cosine
        pairs, which preserves the cyclical nature (e.g., hour 23 is close to hour 0).

        Args:
            timestamps: Series of datetime values

        Returns:
            DataFrame with sin/cos features for hour-of-day and day-of-week
        """
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(timestamps):
            timestamps = pd.to_datetime(timestamps)

        hour = timestamps.dt.hour
        day = timestamps.dt.dayofweek
        month = timestamps.dt.month

        return pd.DataFrame({
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * day / 7),
            "day_cos": np.cos(2 * np.pi * day / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
        })

    def handle_cold_start_embeddings(
        self,
        titles_df: pd.DataFrame,
        embeddings_col: str = "warm_embedding",
    ) -> pd.DataFrame:
        """
        Metadata Averaging for cold-start titles.

        For titles with is_cold_start=True, generate their embedding by
        averaging embeddings of titles with the same genre and director.

        Args:
            titles_df: Titles metadata DataFrame
            embeddings_col: Column name containing embeddings

        Returns:
            DataFrame with updated embeddings for cold-start titles
        """
        df = titles_df.copy()

        if not self._embedding_averages:
            self._compute_embedding_averages(df)

        for idx, row in df[df["is_cold_start"]].iterrows():
            key = (row["genre"], row["director"])
            if key in self._embedding_averages:
                df.at[idx, embeddings_col] = self._embedding_averages[key]
            elif (row["genre"], None) in self._embedding_averages:
                # Fall back to genre-only average
                df.at[idx, embeddings_col] = self._embedding_averages[(row["genre"], None)]

        return df

    def _compute_embedding_averages(self, titles_df: pd.DataFrame) -> None:
        """Compute average embeddings by genre-director for cold-start handling."""
        # Filter to warm titles only
        warm_titles = titles_df[~titles_df["is_cold_start"]]

        # Compute genre-director averages
        for (genre, director), group in warm_titles.groupby(["genre", "director"]):
            embeddings = np.stack(group["warm_embedding"].values)
            self._embedding_averages[(genre, director)] = embeddings.mean(axis=0)

        # Compute genre-only averages as fallback
        for genre, group in warm_titles.groupby("genre"):
            embeddings = np.stack(group["warm_embedding"].values)
            self._embedding_averages[(genre, None)] = embeddings.mean(axis=0)

    def _create_modeling_df(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a merged DataFrame from the data dictionary."""
        # Start with treatment log if available
        if "treatment_log" in data and "interaction_outcomes" in data:
            df = data["treatment_log"].merge(
                data["interaction_outcomes"],
                on=["user_id", "title_id"],
                how="inner"
            )
        elif "interaction_outcomes" in data:
            df = data["interaction_outcomes"].copy()
        else:
            raise ValueError("Data must contain 'interaction_outcomes'")

        # Merge user features
        if "users_metadata" in data:
            df = df.merge(data["users_metadata"], on="user_id", how="left")

        # Merge title features
        if "titles_metadata" in data:
            title_cols = [c for c in data["titles_metadata"].columns
                         if c != "warm_embedding"]
            df = df.merge(data["titles_metadata"][title_cols], on="title_id", how="left")

        return df

    def _extract_embeddings(
        self,
        modeling_df: pd.DataFrame,
        titles_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Extract and optionally handle cold-start for embeddings."""
        # Handle cold-start
        if self.handle_cold_start:
            titles_df = self.handle_cold_start_embeddings(titles_df)

        # Create title_id to embedding mapping
        embedding_map = dict(zip(titles_df["title_id"], titles_df["warm_embedding"]))

        # Extract embeddings for each row
        embeddings = modeling_df["title_id"].map(embedding_map)

        # Stack into matrix
        embedding_matrix = np.stack(embeddings.values)

        # Create DataFrame
        cols = [f"embedding_{i}" for i in range(embedding_matrix.shape[1])]
        return pd.DataFrame(embedding_matrix, columns=cols)

    def _build_feature_names(
        self,
        cat_cols: List[str],
        num_cols: List[str],
    ) -> None:
        """Build list of feature names after transformation."""
        self.feature_names_ = []

        # Categorical feature names
        if self.categorical_encoder is not None:
            self.feature_names_.extend(
                self.categorical_encoder.get_feature_names_out(cat_cols).tolist()
            )

        # Numerical feature names
        self.feature_names_.extend([f"{c}_scaled" for c in num_cols])

        # Timestamp features
        self.feature_names_.extend([
            "hour_sin", "hour_cos",
            "day_sin", "day_cos",
            "month_sin", "month_cos",
        ])

        # Embedding features
        if self.include_embeddings:
            self.feature_names_.extend([f"embedding_{i}" for i in range(self.embedding_dim)])

    def get_feature_names(self) -> List[str]:
        """Return list of feature names after transformation."""
        return self.feature_names_.copy()
