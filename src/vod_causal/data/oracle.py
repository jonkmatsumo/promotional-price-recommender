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

    def calculate_elasticity(
        self,
        user_features: Dict,
        item_features: Dict,
    ) -> float:
        """
        Calculate the price elasticity ($\epsilon$) for a user-item pair.
        
        Elasticity defines how demand changes with price:
        $Demand(P) = Demand(P_0) * (1 + \epsilon * (P - P_0))$
        
        Note: We model this as a negative slope coefficient (semi-elasticity-like).
        A more negative value means MORE sensitive to price increases.
        
        Args:
            user_features: Dict with keys: geo_region, tenure, device_type, price_sensitivity
            item_features: Dict with keys: genre, popularity
            
        Returns:
            Elasticity coefficient (negative value).
            e.g., -0.2 means 1$ increase -> 20% drop in probability.
        """
        # Base elasticity (negative)
        base_elasticity = -0.15

        # Feature effects
        # 1. User sensitivity (latent)
        sensitivity = user_features.get("price_sensitivity", 0.5)
        # High sensitivity -> more negative elasticity
        sens_effect = -0.2 * sensitivity 

        # 2. Tenure
        # Long tenure (> 12 mo) -> Less sensitive (closer to 0)
        tenure = user_features.get("tenure", 0)
        tenure_effect = 0.05 * np.log1p(tenure / 12)

        # 3. Peak Demand (Proxy via Watch Time)
        # High watch time -> Less sensitive
        watch_time = user_features.get("avg_daily_watch_time", 60)
        usage_effect = 0.05 * (watch_time / 100)

        # 4. Content Popularity
        # Popular content -> Less sensitive
        popularity = item_features.get("popularity", 0.5)
        pop_effect = 0.05 * popularity

        # Combine
        total_elasticity = (
            base_elasticity + 
            sens_effect + 
            tenure_effect + 
            usage_effect + 
            pop_effect
        )
        
        # Ensure it's always negative (Price up -> Demand down)
        # Cap at -0.01 (min sensitivity) and -0.8 (max sensitivity)
        return np.clip(total_elasticity, -0.8, -0.01)

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

    def compute_demand(
        self,
        user_features: Dict,
        item_features: Dict,
        price: float,
    ) -> float:
        """
        Compute demand probability at a specific price.
        
        Args:
            user_features: User characteristics
            item_features: Item characteristics
            price: Offered price
            
        Returns:
            Probability of purchase (0-1)
        """
        base_price = item_features.get("base_price", 4.99)
        
        # Base probability at standard price
        base_logit = self._compute_base_probability(user_features, item_features)
        base_prob = self._sigmoid(base_logit)
        
        # Calculate elasticity
        elasticity = self.calculate_elasticity(user_features, item_features)
        
        # Demand Curve Formula (Linear Log-Odds approximation or Linear Probability)
        # Let's use a robust linear probability adjustment clipped to [0,1]
        # P(p) = P(base) + \epsilon * (p - base_price)
        
        price_delta = price - base_price
        
        # Effect on probability
        prob_delta = elasticity * price_delta
        
        # New probability
        final_prob = np.clip(base_prob + prob_delta, 0.01, 0.99)
        
        return final_prob, elasticity

    def compute_observed_outcome(
        self,
        user_features: Dict,
        item_features: Dict,
        is_treated: bool,
        offered_price: float,
        return_revenue: bool = True,
    ) -> Dict:
        """
        Compute the observed outcome based on offered price.
        """
        # Compute demand probability at offered price
        true_prob, true_elasticity = self.compute_demand(
            user_features, item_features, offered_price
        )
        
        # Add random noise to decision
        # We do this by sampling from the probability
        did_rent = self.rng.random() < true_prob

        result = {
            "did_rent": did_rent,
            "demand_at_price": true_prob,
            "true_elasticity": true_elasticity,
        }

        if return_revenue:
            base_price = item_features.get("base_price", 4.99)
            if did_rent:
                revenue = offered_price
            else:
                revenue = 0.0
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
