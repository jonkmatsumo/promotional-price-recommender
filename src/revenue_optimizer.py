"""
Revenue Optimization Solver.

This module uses the estimated price elasticity from the DML model to optimize
pricing for individual users. It finds the price point that maximizes expected revenue:
    
    Expected Revenue = Price * Probability(Purchase | Price, UserFeatures)
    
Where Probability is estimated using the DML model's elasticity coefficient:
    P(p) = P(base) + elasticity * (p - base_price)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any

from vod_causal.models.dml import DMLWithEconML


def get_price_candidates(
    min_price: float = 2.99, 
    max_price: float = 14.99, 
    step: float = 0.50
) -> np.ndarray:
    """Generate candidate price points."""
    return np.arange(min_price, max_price + 0.01, step)


def optimize_price(
    user_features: pd.DataFrame,
    dml_model: DMLWithEconML,
    base_price: float = 4.99,
    base_demand: float = 0.15,
    min_price: float = 2.99,
    max_price: float = 14.99,
) -> Tuple[float, float, float]:
    """
    Find the optimal price for a specific user to maximize revenue.
    
    Args:
        user_features: DataFrame containing feature vector for a SINGLE user.
                       Must match the format used to train the DML model.
        dml_model: Trained DMLWithEconML model instance.
        base_price: The reference price used during training/baseline.
        base_demand: The baseline probability of purchase at base_price.
                     Ideally, this comes from a separate propensity model (nuisance model).
        min_price: Minimum allowed price.
        max_price: Maximum allowed price.
        
    Returns:
        Tuple of (optimal_price, expected_revenue, predicted_demand)
    """
    # 1. Estimate Elasticity (theta) for this user
    # theta represents the change in probability per unit change in price
    elasticity = dml_model.effect(user_features)[0]
    
    # 2. Define Demand Curve Function
    # P(p) = P_base + theta * (p - p_base)
    # We clip probability to [0.01, 1.0] to be realistic
    def predict_demand(price):
        delta_p = price - base_price
        prob = base_demand + elasticity * delta_p
        return max(0.01, min(0.99, prob))

    # 3. Grid Search for Optimum
    # We use a grid search because the revenue function might not be perfectly convex
    # if we add complex constraints, but mainly because it's fast enough for 1D.
    candidates = get_price_candidates(min_price, max_price)
    
    best_price = base_price
    max_revenue = -1.0
    best_demand = 0.0
    
    for log_price in candidates:
        # Round to nice "99" endings for realism (optional, but good for retail)
        # e.g. 2.99, 3.49, 3.99...
        # For this solver, we stick to the raw candidate
        
        demand = predict_demand(log_price)
        expected_revenue = log_price * demand
        
        if expected_revenue > max_revenue:
            max_revenue = expected_revenue
            best_price = log_price
            best_demand = demand
            
    return float(best_price), float(max_revenue), float(best_demand)


def bulk_optimize(
    users_df: pd.DataFrame,
    dml_model: DMLWithEconML,
    base_probas: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Optimize prices for a batch of users.
    
    Args:
        users_df: DataFrame of user features.
        dml_model: Fitted DML model.
        base_probas: Optional array of baseline purchase probabilities (at default price)
                     from a separate Outcome Prediction Model. 
                     If None, assumes constant baseline (not recommended for production).
    
    Returns:
        DataFrame with columns [user_id, optimal_price, predicted_revenue]
    """
    # Get elasticities for all users
    elasticities = dml_model.effect(users_df)
    
    results = []
    candidates = get_price_candidates()
    
    # Vectorized optimization could be done, but loop is clearer for prototype
    for i in range(len(users_df)):
        user_idx = users_df.index[i]
        theta = elasticities[i]
        
        # Use provided baseline prob or default
        p_base = base_probas[i] if base_probas is not None else 0.15
        
        # Vectorized search over candidates for this user
        # Demand = p_base + theta * (Price - 4.99)
        demands = p_base + theta * (candidates - 4.99)
        demands = np.clip(demands, 0.01, 0.99)
        
        revenues = candidates * demands
        
        best_idx = np.argmax(revenues)
        best_price = candidates[best_idx]
        best_rev = revenues[best_idx]
        
        results.append({
            "user_index": user_idx,
            "optimal_price": best_price,
            "expected_revenue": best_rev,
            "elasticity": theta
        })
        
    return pd.DataFrame(results)
