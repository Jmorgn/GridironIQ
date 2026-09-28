"""Test QB game-context feature combinations with walk-forward validation.

This isolates whether rest, environment, and betting-market context each add
value on top of the current football feature set.

2026 is intentionally excluded from model selection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_model_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_context_combination_results.csv"

TARGET = "actual_fantasy_points"

FOLDS = [
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
]

CORE = [
    "previous_fp",
    "avg_fp_last_3",
    "avg_fp_last_5",
    "avg_completions_last_3",
    "avg_completions_last_5",
    "avg_attempts_last_3",
    "avg_attempts_last_5",
    "avg_pass_yards_last_3",
    "avg_pass_yards_last_5",
    "avg_pass_tds_last_3",
    "avg_pass_tds_last_5",
    "avg_interceptions_last_3",
    "avg_interceptions_last_5",
    "avg_rush_yards_last_3",
    "avg_rush_yards_last_5",
    "avg_rush_tds_last_3",
    "avg_rush_tds_last_5",
    "opp_avg_pass_yards_allowed_last_3",
    "opp_avg_pass_yards_allowed_last_5",
    "opp_avg_pass_tds_allowed_last_3",
    "opp_avg_pass_tds_allowed_last_5",
    "opp_avg_def_interceptions_last_3",
    "opp_avg_def_interceptions_last_5",
    "opp_avg_def_sacks_last_3",
    "opp_avg_def_sacks_last_5",
    "previous_offense_pct",
    "avg_offense_pct_last_3",
    "avg_offense_pct_last_5",
    "start_like_games_last_3",
    "start_like_games_last_5",
    "home_away",
    "team",
    "opponent",
]

REST = [
    "team_rest",
    "opponent_rest",
    "rest_advantage",
    "neutral_site",
]

ENVIRONMENT = [
    "roof",
    "surface",
    "game_temp",
    "game_wind",
]

MARKET = [
    "team_spread_line",
    "game_total_line",
]

CATEGORICAL = {"team", "opponent", "home_away", "roof", "surface"}


def present(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def make_pipeline(features: list[str]) -> Pipeline:
    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c not in CATEGORICAL]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(transformers=transformers),
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=3,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[
        df["season"].between(2021, 2025) & df[TARGET].notna()
    ].copy()

    core = present(df, CORE)
    rest = present(df, REST)
    environment = present(df, ENVIRONMENT)
    market = present(df, MARKET)

    experiments = [
        ("Core", core),
        ("Core + Rest", core + rest),
        ("Core + Environment", core + environment),
        ("Core + Market", core + market),
        ("Core + Rest + Environment", core + rest + environment),
        ("Core + Rest + Market", core + rest + market),
        ("Core + Environment + Market", core + environment + market),
        ("All Context", core + rest + environment + market),
    ]

    rows = []

    print("GRIDIRONIQ QB CONTEXT COMBINATION TEST")
    print("=" * 68)
    print("Random Forest | 2026 excluded | Lower MAE is better.\n")

    for name, features in experiments:
        fold_maes = []

        for train_start, train_end, test_season in FOLDS:
            train = df[df["season"].between(train_start, train_end)]
            test = df[df["season"].eq(test_season)]

            model = make_pipeline(features)
            model.fit(train[features], train[TARGET])
            predictions = model.predict(test[features])
            mae = mean_absolute_error(test[TARGET], predictions)
            fold_maes.append(mae)

            rows.append(
                {
                    "experiment": name,
                    "test_season": test_season,
                    "feature_count": len(features),
                    "mae": mae,
                }
            )

        print(
            f"{name:<35} "
            f"features={len(features):>2}  "
            f"2023={fold_maes[0]:.3f}  "
            f"2024={fold_maes[1]:.3f}  "
            f"2025={fold_maes[2]:.3f}  "
            f"AVG={sum(fold_maes) / len(fold_maes):.3f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_FILE, index=False)

    summary = (
        results.groupby(["experiment", "feature_count"], as_index=False)
        .agg(average_mae=("mae", "mean"))
        .sort_values("average_mae")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(
        summary.to_string(
            index=False,
            formatters={"average_mae": "{:.3f}".format},
        )
    )

    print(
        f"\nBest combination: {summary.iloc[0]['experiment']} "
        f"(MAE {summary.iloc[0]['average_mae']:.3f})"
    )
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
