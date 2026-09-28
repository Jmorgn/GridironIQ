# GridironIQ Architecture

## Phase 1

```text
nflverse GitHub releases
        |
        v
src/data/download_nfl_data.py
        |
        v
data/raw/
        |
        v
SQL / Python feature engineering
        |
        v
QB_MODEL_DATASET
        |
        v
Linear Regression baseline
        |
        +--> Random Forest
        |
        +--> Gradient Boosting
        |
        v
Model evaluation
        |
        v
Azure Machine Learning
```

## Data Leakage Rule

A row predicting a player's Week N performance may only use information from games completed before Week N.

For example, a Week 8 prediction may use Weeks 1-7 but never Week 8 or later.

## Evaluation Strategy

The initial time-aware split will be:

- 2021-2024: training
- 2025: validation
- 2026: live/test season

Primary metric: Mean Absolute Error (MAE).

Secondary metrics: RMSE and R-squared.
