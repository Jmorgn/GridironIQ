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

That result is expected to be limited by the current training table, which only contains QBs who recorded game activity. An initial depth-chart candidate audit exposed an important data-quality issue: older weekly depth charts also contain entries during team bye weeks. Those rows must not become zero-point fantasy training examples.

The candidate audit now filters depth-chart rows against the actual regular-season schedule before measuring zero activity. After removing bye-week contamination, the 2021-2025 audit found **6,716 historical QB candidate rows**. Among **2,702 QB1 candidate rows**, **12.1% recorded zero QB activity**.

A deeper validation checked those 327 zero-activity QB1 rows: only **3.4%** still showed offensive snaps, while **98.8%** had another QB active for the same team/week. That supports using these rows as real availability/replacement examples for the v2 pregame candidate dataset rather than treating them as simple join failures.

### QB Model v2

QB v2 uses a two-stage architecture:

1. A Random Forest classifier estimates the probability that each listed QB receives a meaningful role (at least 50% of offensive snaps).
2. A Random Forest regressor predicts fantasy points conditional on a meaningful role.
3. The production gate selects the **highest role-probability QB on each team/week** and assigns that QB the conditional fantasy projection; other listed QBs receive zero.

The role classifier averaged **0.937 ROC AUC**, **0.870 accuracy**, **0.818 precision**, and **0.869 recall** in walk-forward validation across 2023-2025.

| Candidate scoring strategy | All-candidate MAE | Listed-QB1 MAE | Actual-role MAE | Played-QB MAE |
| --- | ---: | ---: | ---: | ---: |
| **Top role per team** | **5.321** | 9.915 | 9.690 | **8.512** |
| Listed QB1 only | 5.425 | 10.009 | 9.883 | 8.666 |
| 50% hard role gate | 5.927 | 9.995 | 9.778 | 8.755 |
| Direct candidate regression | 6.567 | **9.631** | 9.712 | 8.680 |
| Soft probability-weighted two-stage | 7.224 | 9.706 | **9.425** | 8.618 |

The production top-role-per-team gate is used because it performed best on the full pregame candidate pool and on QBs who actually played, while also enforcing the normal one-primary-QB-per-team structure. The table also preserves the tradeoffs: direct regression was best on the listed-QB1 subset, and soft probability weighting was best on the actual meaningful-role subset.

The v1 **8.470 MAE** and v2 **5.321 all-candidate MAE** are not directly comparable because v1 evaluates only QBs who recorded game activity, while v2 includes the entire pregame depth-chart candidate pool, including inactive and zero-point rows.

## Weekly Workflow

After the QB v2 models have been trained once, the normal weekly workflow is a single command:

```cmd
py run_weekly.py
```

That command:

1. refreshes the live 2026 nflverse files and schedule/results data;
2. rebuilds the processed historical tables;
3. rebuilds the QB v2 pregame candidate dataset;
4. loads the saved QB v2 model bundle and generates the earliest upcoming week's rankings.

The main weekly output is:

```text
data/processed/qb_v2_weekly_rankings.csv
```

All future QB candidates and their role probabilities are also saved to:

```text
data/processed/qb_v2_all_future_candidates.csv
```

The weekly rankings file also includes `key_positives` and `key_negatives` columns. These are descriptive context signals built from recent fantasy form, betting environment, opponent pass-defense trends, pass rush, rest, home/away status, weather, injury status, and role confidence. They are intentionally labeled as context signals rather than exact Random Forest feature-attribution values.

The command-line report prints the top 10 QBs with a short explanation of why GridironIQ likes or dislikes the matchup.

Use `py run_weekly.py --retrain` only when intentionally refitting the official QB v2 models. Normal weekly refreshes do not need to retrain the 2021-2025 model.

## Long-Term Roadmap

Later phases will add RB/WR/TE models, travel distance, defensive personnel changes, supporting-cast availability, Next Gen Stats, uncertainty ranges, player correlation, and matchup-level win-probability recommendations.
