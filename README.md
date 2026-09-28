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

## Phase 1 — QB Model v1

Completed milestones:

1. Download weekly player/team stats, snap counts, schedules, depth charts, and supporting metadata.
2. Reconstruct the league's custom fantasy scoring from weekly nflverse data.
3. Build leakage-safe rolling QB and opponent-defense features.
4. Add role/snap-share, home/away, rest, venue/weather, betting-market, and pregame depth-chart context.
5. Compare Linear Regression, Random Forest, and Gradient Boosting with walk-forward validation.
6. Hold 2026 out of model selection as the live/demo season.

### QB Model v1 Benchmark

The selected model is a Random Forest evaluated with walk-forward validation:

| Fold | Train | Test | MAE |
| --- | --- | --- | ---: |
| 1 | 2021-2022 | 2023 | 8.560 |
| 2 | 2021-2023 | 2024 | 8.534 |
| 3 | 2021-2024 | 2025 | 8.316 |
| **Average** | — | **2023-2025** | **8.470** |

The simple Last-3 fantasy-points baseline averaged **9.639 MAE**, so QB Model v1 reduced MAE by **12.1%** across the three unseen-season folds.

The current 2026 live/demo snapshot produced **8.187 MAE on 109 QB-games** after fitting the selected model on 2021-2025. That number is not used for model selection and will change as the season grows.

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
- Snap counts
- Schedules and game context
- Depth charts
- Injury/practice reports
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
- rest and neutral-site context
- roof, surface, temperature, and wind
- pregame spread and game total
- QB depth-chart rank / QB1 status

All predictive features must be calculated from games that happened before the game being predicted.

## Phase 2 — Availability and Personnel

The first injury/practice experiment did **not** improve the active-QB model: walk-forward MAE moved from **8.472** to **8.478**. Those features are therefore not part of the current v1 model.

That result is expected to be limited by the current training table, which only contains QBs who recorded game activity. The next v2 step is a pregame candidate table built from depth charts so the model can learn zero-participation cases such as ruled-out or inactive QBs.

## Long-Term Roadmap

Later phases will add RB/WR/TE models, travel distance, defensive personnel changes, supporting-cast availability, Next Gen Stats, uncertainty ranges, player correlation, and matchup-level win-probability recommendations.
