# GridironIQ

NFL fantasy football analytics and machine learning platform using SQL, Python, Azure Machine Learning, and historical NFL data.

## Project Goal

GridironIQ will turn historical NFL data into weekly fantasy-football predictions and matchup-aware lineup recommendations.

The first milestone is a quarterback regression model that predicts custom fantasy points using only information that would have been available before kickoff.

## Initial Architecture

```text
nflverse data
     |
     v
Python ingestion
     |
     v
data/raw/
     |
     v
SQL + Python feature engineering
     |
     v
data/processed/
     |
     v
Regression models
     |
     v
Azure Machine Learning
     |
     v
Weekly predictions / fantasy decision engine
```

## Phase 1

1. Download weekly player and team statistics for 2021-2026.
2. Build custom fantasy scoring.
3. Create QB rolling features without data leakage.
4. Train a baseline linear regression model.
5. Compare it with tree-based regression models.
6. Evaluate predictions with MAE, RMSE, and R-squared.
7. Hold out 2026 as the live/test season.

## Repository Structure

```text
GridironIQ/
├── data/
│   ├── raw/
│   └── processed/
├── sql/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── azure/
├── docs/
├── requirements.txt
└── README.md
```

## Data

Initial data comes from the nflverse project:

- Weekly player statistics
- Weekly team statistics
- Player metadata

Raw downloaded data is intentionally excluded from Git. The ingestion script reproduces it locally.

## First Model

The first target will be:

```text
ACTUAL_FANTASY_POINTS
```

for quarterbacks.

Example features will include:

- previous-week fantasy points
- rolling 3-game and 5-game fantasy averages
- completions and attempts
- passing yards and passing touchdowns
- interceptions
- rushing yards and rushing touchdowns
- opponent passing yards allowed
- opponent passing touchdowns allowed
- opponent sacks and interceptions
- home/away status

All predictive features must be calculated from games that happened before the game being predicted.

## Long-Term Roadmap

Later phases will add RB/WR/TE models, weather, stadium type, travel, rest, injuries, depth charts, defensive personnel changes, betting markets, Next Gen Stats, uncertainty ranges, player correlation, and matchup-level win-probability recommendations.
