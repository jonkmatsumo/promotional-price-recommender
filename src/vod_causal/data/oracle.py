"""
Oracle simulation for ground truth causal effects.

This module provides the "ground truth" for our synthetic data, allowing us to
calculate the actual counterfactual outcomes - something impossible with real data
but necessary for validating causal inference models.

The CausalOracle encodes hidden causal relationships between user features,
item features, and treatment effects that our models will attempt to discover.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class CausalOracle:
    """
    Oracle that generates ground truth causal effects.

    This allows validation of our causal models by computing true counterfactuals
    (impossible with real data). The oracle encodes complex interaction effects
    that represent realistic promotional response patterns in VOD.

    The hidden causal structure includes:
    - Genre-specific treatment effects (Sci-Fi fans respond more to promotions)
    - Geographic heterogeneity (price sensitivity varies by region)
    - Tenure interactions (long-term subscribers respond differently)
    - Device-based modifiers (mobile users have different conversion patterns)

    Attributes:
        rng: NumPy random generator for reproducibility
        noise_scale: Standard deviation of outcome noise
    """

    # Hidden causal effect parameters (the "truth" models try to discover)
    _GENRE_EFFECTS: Dict[str, float] = {
        "Sci-Fi": 0.15,
        "Action": 0.12,
        "Comedy": 0.08,
        "Drama": 0.06,
        "Horror": 0.10,
        "Documentary": 0.04,
        "Romance": 0.07,
        "Thriller": 0.11,
    }

    _REGION_EFFECTS: Dict[str, float] = {
        "US": 0.10,
        "EU": 0.08,
        "APAC": 0.12,
        "LATAM": 0.14,
    }

    _DEVICE_EFFECTS: Dict[str, float] = {
        "Mobile": 0.05,
        "Desktop": 0.03,
        "SmartTV": 0.08,
        "Tablet": 0.04,
    }

    def __init__(self, seed: int = 42, noise_scale: float = 0.1):
        """
        Initialize the causal oracle.

        Args:
            seed: Random seed for reproducibility
            noise_scale: Standard deviation of Gaussian noise added to outcomes
        """
        self.rng = np.random.default_rng(seed)
        self.noise_scale = noise_scale

    def calculate_true_lift(
        self,
        user_features: Dict,
        item_features: Dict,
        discount_level: float = 0.1,
    ) -> float:
        """
        Calculate the true treatment effect (uplift) for a user-item pair.

        The true lift is a deterministic function of user and item features,
        modulated by the discount level. This represents the hidden causal
        effect that our models attempt to estimate.

        **Causal Structure (Hidden from Models):**
        - Base lift from promotion
        - Genre-specific bonus (e.g., Sci-Fi fans respond more)
        - Region-specific bonus (price sensitivity varies)
        - Tenure modifier (log relationship with subscription length)
        - Device modifier
        - Discount level amplifier

        Args:
            user_features: Dict with keys: geo_region, tenure, device_type, price_sensitivity
            item_features: Dict with keys: genre, popularity
            discount_level: The discount being offered (0.0 to 0.3)

        Returns:
            The true conditional average treatment effect (CATE) in probability units
        """
        # Base lift that everyone gets from seeing a promotion
        base_lift = 0.05

        # Genre-specific effect
        genre = item_features.get("genre", "Drama")
        genre_bonus = self._GENRE_EFFECTS.get(genre, 0.05)

        # Geographic effect (price sensitivity varies by region)
        region = user_features.get("geo_region", "US")
        region_bonus = self._REGION_EFFECTS.get(region, 0.05)

        # Tenure interaction: longer-tenured users are less responsive to promotions
        # (they know what they want), but there's a sweet spot
        tenure = user_features.get("tenure", 12)
        tenure_modifier = 0.02 * np.log1p(tenure) - 0.005 * np.log1p(max(0, tenure - 24))

        # Device effect
        device = user_features.get("device_type", "Desktop")
        device_bonus = self._DEVICE_EFFECTS.get(device, 0.03)

        # Price sensitivity amplifier (more sensitive users respond more to discounts)
        price_sensitivity = user_features.get("price_sensitivity", 0.5)
        sensitivity_amplifier = 0.5 + price_sensitivity  # Range: [0.5, 1.5]

        # Discount level amplifier (higher discounts = higher lift, diminishing returns)
        discount_amplifier = 1.0 + 2.0 * np.sqrt(discount_level)

        # Combine all effects
        raw_lift = (
            base_lift
            + genre_bonus
            + region_bonus
            + tenure_modifier
            + device_bonus
        )

        # Apply amplifiers
        total_lift = raw_lift * sensitivity_amplifier * discount_amplifier

        # Clip to reasonable range
        return np.clip(total_lift, 0.0, 0.5)

    def _compute_base_probability(
        self,
        user_features: Dict,
        item_features: Dict,
    ) -> float:
        """
        Compute baseline conversion probability (without treatment).

        This represents the natural propensity to rent based on user-item match,
        independent of any promotional effect.

        Args:
            user_features: User characteristics
            item_features: Item characteristics

        Returns:
            Base log-odds of conversion
        """
        # User affinity based on watch time (more engaged users convert more)
        watch_time = user_features.get("avg_daily_watch_time", 60)
        user_affinity = 0.5 * np.log1p(watch_time / 60)

        # Item popularity effect
        popularity = item_features.get("popularity", 0.5)
        item_effect = 1.0 * popularity

        # User-item interaction: simulate some genre preferences by region
        genre = item_features.get("genre", "Drama")
        region = user_features.get("geo_region", "US")
        
        # Hidden preference patterns
        preference_matrix = {
            ("US", "Action"): 0.3,
            ("US", "Sci-Fi"): 0.25,
            ("EU", "Drama"): 0.3,
            ("EU", "Documentary"): 0.25,
            ("APAC", "Action"): 0.35,
            ("APAC", "Horror"): 0.2,
            ("LATAM", "Comedy"): 0.3,
            ("LATAM", "Romance"): 0.25,
        }
        preference_bonus = preference_matrix.get((region, genre), 0.1)

        # Combine into base log-odds
        base_logit = -1.5 + user_affinity + item_effect + preference_bonus

        return base_logit

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Apply sigmoid function with numerical stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def compute_potential_outcomes(
        self,
        user_features: Dict,
        item_features: Dict,
        discount_level: float = 0.1,
    ) -> Tuple[float, float]:
        """
        Compute Y(0) and Y(1) - the potential outcomes.

        This is the fundamental causal inference quantity: what would happen
        under control vs treatment. In real data, we only observe one of these;
        the oracle lets us see both for validation.

        Args:
            user_features: User characteristics
            item_features: Item characteristics
            discount_level: Discount level for treatment arm

        Returns:
            Tuple of (y0, y1):
                y0: Control outcome (probability of conversion without discount)
                y1: Treatment outcome (probability of conversion with discount)
        """
        # Base probability (in log-odds space)
        base_logit = self._compute_base_probability(user_features, item_features)

        # Add noise (same noise for both potential outcomes - individual heterogeneity)
        noise = self.rng.normal(0, self.noise_scale)

        # Y(0): Control outcome - no treatment effect
        y0 = self._sigmoid(base_logit + noise)

        # Y(1): Treatment outcome - includes true lift (converted to log-odds)
        true_lift = self.calculate_true_lift(user_features, item_features, discount_level)
        # Convert lift to log-odds addition (approximate)
        lift_logit = np.log((y0 + true_lift) / (1 - y0 - true_lift + 1e-10) + 1e-10) - np.log(
            y0 / (1 - y0 + 1e-10) + 1e-10
        )
        y1 = self._sigmoid(base_logit + lift_logit + noise)

        # Ensure valid probabilities
        y0 = np.clip(y0, 0.001, 0.999)
        y1 = np.clip(y1, 0.001, 0.999)

        return float(y0), float(y1)

    def compute_observed_outcome(
        self,
        user_features: Dict,
        item_features: Dict,
        is_treated: bool,
        discount_level: float = 0.1,
        return_revenue: bool = True,
    ) -> Dict:
        """
        Compute the observed outcome based on treatment assignment.

        This simulates what we actually observe in data: only the outcome
        under the assigned treatment condition.

        Args:
            user_features: User characteristics
            item_features: Item characteristics
            is_treated: Whether this observation received treatment
            discount_level: Discount level (only used if is_treated=True)
            return_revenue: If True, also compute revenue

        Returns:
            Dict with:
                - did_rent: Binary conversion outcome
                - conversion_prob: Underlying probability
                - revenue: Revenue generated (if return_revenue=True)
                - true_cate: The true causal effect (for validation)
                - y0, y1: Potential outcomes (for validation)
        """
        # Compute potential outcomes
        y0, y1 = self.compute_potential_outcomes(
            user_features, item_features, discount_level
        )

        # Select observed probability based on treatment
        if is_treated:
            conversion_prob = y1
        else:
            conversion_prob = y0

        # Simulate binary outcome
        did_rent = self.rng.random() < conversion_prob

        # Compute true CATE (for validation only - not observable in real world)
        true_cate = self.calculate_true_lift(user_features, item_features, discount_level)

        result = {
            "did_rent": did_rent,
            "conversion_prob": conversion_prob,
            "true_cate": true_cate,
            "y0": y0,
            "y1": y1,
        }

        if return_revenue:
            base_price = item_features.get("base_price", 4.99)
            if did_rent:
                if is_treated:
                    revenue = base_price * (1 - discount_level)
                else:
                    revenue = base_price
            else:
                revenue = 0.0
            result["revenue"] = revenue

        return result

    def compute_watch_duration(
        self,
        user_features: Dict,
        item_features: Dict,
        did_rent: bool,
    ) -> float:
        """
        Compute watch duration in minutes.

        Args:
            user_features: User characteristics
            item_features: Item characteristics
            did_rent: Whether the user rented the title

        Returns:
            Watch duration in minutes (0 if not rented)
        """
        if not did_rent:
            return 0.0

        # Base duration influenced by user's typical watch patterns
        avg_watch = user_features.get("avg_daily_watch_time", 60)
        base_duration = 30 + 0.5 * avg_watch

        # Genre affects completion rate
        genre = item_features.get("genre", "Drama")
        genre_duration_mult = {
            "Documentary": 0.7,
            "Action": 1.1,
            "Comedy": 0.9,
            "Drama": 1.0,
            "Sci-Fi": 1.05,
            "Horror": 0.85,
            "Romance": 0.95,
            "Thriller": 1.0,
        }
        mult = genre_duration_mult.get(genre, 1.0)

        # Add noise
        duration = base_duration * mult * (0.8 + 0.4 * self.rng.random())

        return max(0, duration)
