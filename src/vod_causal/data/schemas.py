"""
Data schema definitions for VOD entities.

Defines the core data structures for:
- TitleMetadata: Content/movie information
- UserMetadata: Audience/subscriber information
- TreatmentLog: Promotional intervention records
- InteractionOutcome: Target variables (rentals, revenue, watch time)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class TitleMetadata:
    """
    Content metadata for VOD titles.

    Represents a movie or show in the catalog with both explicit features
    (genre, director, release year) and learned embeddings for recommendation.

    Attributes:
        title_id: Unique identifier for the title
        genre: Primary genre category (Action, Comedy, Drama, Sci-Fi, etc.)
        director: Director name (categorical)
        release_year: Year of release
        warm_embedding: Learned latent factor vector (e.g., from matrix factorization)
        is_cold_start: Flag for titles with <5 interactions (need special handling)
        base_popularity: Intrinsic popularity score (0-1)
    """

    title_id: str
    genre: str
    director: str
    release_year: int
    warm_embedding: np.ndarray = field(default_factory=lambda: np.zeros(32))
    is_cold_start: bool = False
    base_popularity: float = 0.5

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.release_year < 1900 or self.release_year > 2030:
            raise ValueError(f"Invalid release_year: {self.release_year}")
        if not 0 <= self.base_popularity <= 1:
            raise ValueError(f"base_popularity must be in [0, 1]: {self.base_popularity}")


@dataclass
class UserMetadata:
    """
    Audience metadata for VOD users.

    Represents a subscriber with demographic and behavioral features
    that influence their response to promotions.

    Attributes:
        user_id: Unique identifier for the user
        subscription_tenure_months: How long they've been subscribed
        geo_region: Geographic region (US, EU, APAC, LATAM)
        device_type: Primary device (Mobile, Desktop, SmartTV, Tablet)
        avg_daily_watch_time: Average daily viewing in minutes
        price_sensitivity: Latent price sensitivity score (0-1, higher = more sensitive)
    """

    user_id: str
    subscription_tenure_months: int
    geo_region: str
    device_type: str
    avg_daily_watch_time: float
    price_sensitivity: float = 0.5

    # Valid values for categorical fields
    VALID_REGIONS: List[str] = field(
        default_factory=lambda: ["US", "EU", "APAC", "LATAM"], repr=False
    )
    VALID_DEVICES: List[str] = field(
        default_factory=lambda: ["Mobile", "Desktop", "SmartTV", "Tablet"], repr=False
    )

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.subscription_tenure_months < 0:
            raise ValueError(
                f"subscription_tenure_months must be >= 0: {self.subscription_tenure_months}"
            )
        if self.avg_daily_watch_time < 0:
            raise ValueError(
                f"avg_daily_watch_time must be >= 0: {self.avg_daily_watch_time}"
            )
        if not 0 <= self.price_sensitivity <= 1:
            raise ValueError(
                f"price_sensitivity must be in [0, 1]: {self.price_sensitivity}"
            )


@dataclass
class TreatmentLog:
    """
    The intervention/promotion record.

    Represents when a user was shown a promotional discount for a title.
    This is the "treatment" variable in our causal model.

    Attributes:
        user_id: User who received the promotion
        title_id: Title that was promoted
        discount_level: Discount percentage (0.0, 0.1, 0.2, 0.3)
        campaign_id: Identifier for the marketing campaign
        timestamp: When the promotion was shown
    """

    user_id: str
    title_id: str
    discount_level: float
    campaign_id: str
    timestamp: datetime

    # Valid discount levels
    VALID_DISCOUNT_LEVELS: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3], repr=False
    )

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.discount_level < 0 or self.discount_level > 1:
            raise ValueError(f"discount_level must be in [0, 1]: {self.discount_level}")

    @property
    def is_treated(self) -> bool:
        """Return True if this is an actual treatment (discount > 0)."""
        return self.discount_level > 0


@dataclass
class InteractionOutcome:
    """
    The target variables we aim to predict.

    Represents the observed outcome after a user was exposed to a title
    (with or without a promotional discount).

    Attributes:
        user_id: User identifier
        title_id: Title identifier
        did_rent: Binary outcome - did the user rent/purchase?
        revenue_generated: Actual revenue (after any discount)
        watch_duration_minutes: How long they watched (engagement metric)
        base_price: Original price before any discount
    """

    user_id: str
    title_id: str
    did_rent: bool
    revenue_generated: float
    watch_duration_minutes: float
    base_price: float = 4.99

    def __post_init__(self):
        """Validate fields after initialization."""
        if self.revenue_generated < 0:
            raise ValueError(f"revenue_generated must be >= 0: {self.revenue_generated}")
        if self.watch_duration_minutes < 0:
            raise ValueError(
                f"watch_duration_minutes must be >= 0: {self.watch_duration_minutes}"
            )
        if self.base_price < 0:
            raise ValueError(f"base_price must be >= 0: {self.base_price}")

    @property
    def conversion_value(self) -> float:
        """Return 1.0 if rented, 0.0 otherwise (for binary outcome modeling)."""
        return 1.0 if self.did_rent else 0.0


# Type aliases for collections
TitleCatalog = List[TitleMetadata]
UserBase = List[UserMetadata]
TreatmentHistory = List[TreatmentLog]
OutcomeHistory = List[InteractionOutcome]
