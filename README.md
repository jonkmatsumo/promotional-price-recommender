# Causal Uplift & Revenue Optimization for VOD

A modular Python-based prototype implementing causal inference methods (X-Learner and Double Machine Learning) for Video-on-Demand promotional price optimization.

## Overview

This project provides tools to identify which users will respond positively to promotional discounts, enabling:

- **Targeting Optimization**: Focus marketing spend on "persuadable" users
- **Revenue Maximization**: Avoid discounting users who would convert anyway
- **Personalized Pricing**: Understand heterogeneous treatment effects across user segments

## Key Features

- **Synthetic Data Generation**: Realistic VOD dataset with hidden causal effects for model validation
- **Feature Engineering**: One-hot encoding, cyclical timestamps, cold-start handling
- **Causal Models**:
  - **X-Learner**: Meta-learner for CATE estimation with treatment imbalance
  - **Double Machine Learning (DML)**: Price elasticity estimation with continuous treatments
- **Evaluation**: Qini curves, AUUC, calibration plots, policy simulation
- **Production Integration**: EconML wrappers for enterprise deployment

## Installation

```bash
# Clone the repository
git clone https://github.com/jonkmatsumo/promotional-price-recommender.git
cd promotional-price-recommender

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from vod_causal.data import VODSyntheticData
from vod_causal.preprocessing import FeatureTransformer
from vod_causal.models import XLearner
from vod_causal.evaluation import PolicyRanker, UpliftMetrics

# Generate synthetic data
generator = VODSyntheticData(n_users=10_000, n_titles=500)
data = generator.generate_all()

# Create modeling dataset
modeling_df = generator.create_modeling_dataset(data)

# Feature engineering
transformer = FeatureTransformer()
X = transformer.fit_transform(data, modeling_df)
y = modeling_df['did_rent'].astype(int)
treatment = modeling_df['is_treated'].astype(int)

# Train X-Learner
xlearner = XLearner()
xlearner.fit(X, y, treatment)

# Predict CATE
predicted_cate = xlearner.predict(X)

# Evaluate
qini_x, qini_y = UpliftMetrics.compute_qini_curve(y, treatment, predicted_cate)
auuc = UpliftMetrics.compute_auuc(qini_x, qini_y)
print(f"AUUC: {auuc:.4f}")

# Generate recommendations
ranker = PolicyRanker(discount_cost=0.50)
recommendations = ranker.rank(modeling_df[['user_id', 'title_id']], predicted_cate, top_k=5)
```

## Project Structure

```
promotional-price-recommender/
├── src/vod_causal/
│   ├── data/
│   │   ├── schemas.py      # Data class definitions
│   │   ├── oracle.py       # Ground truth causal effects
│   │   └── generator.py    # Synthetic data generation
│   ├── preprocessing/
│   │   ├── preprocessing.py # Feature transformations
│   │   └── propensity.py   # Propensity scoring
│   ├── models/
│   │   ├── base_learners.py # XGBoost response models
│   │   ├── xlearner.py     # X-Learner meta-learner
│   │   └── dml.py          # Double Machine Learning
│   └── evaluation/
│       ├── metrics.py      # Qini, AUUC, etc.
│       ├── visualization.py # Plotting utilities
│       └── policy_ranker.py # Recommendation engine
├── notebooks/
│   └── causal_model.ipynb  # Interactive demo
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Methodology

### X-Learner Algorithm

The X-Learner is particularly effective for VOD because promotional campaigns typically have treatment imbalance (few titles on sale):

1. **Stage 1**: Fit response models μ₀, μ₁ on control/treatment groups
2. **Stage 2**: Impute counterfactual outcomes
3. **Stage 3**: Train CATE models τ₀, τ₁ on imputed effects
4. **Stage 4**: Combine with propensity weighting: τ(x) = e(x)·τ₀(x) + (1-e(x))·τ₁(x)

### Double Machine Learning

For continuous treatments (variable discount levels), DML provides valid causal inference:

1. Residualize treatment: R_T = T - f̂(X)
2. Residualize outcome: R_Y = Y - ĥ(X)
3. Regress R_Y ~ R_T to get elasticity θ

## Notebooks

- `notebooks/causal_model.ipynb`: Complete walkthrough demonstrating data generation, model training, evaluation, and policy simulation

## Dependencies

- Python 3.10+
- pandas, numpy, scikit-learn
- xgboost
- econml (Microsoft's causal ML library)
- matplotlib, seaborn
- jupyter

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black src/ tests/
ruff check src/ tests/
```

## References

- [EconML Documentation](https://econml.azurewebsites.net/)
- [Causal Inference in Statistics: A Primer](https://www.amazon.com/Causal-Inference-Statistics-Judea-Pearl/dp/1119186846)
- [Machine Learning for Causal Inference](https://arxiv.org/abs/2002.01163)

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.