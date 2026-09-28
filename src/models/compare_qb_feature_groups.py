"""Compare QB feature groups with walk-forward validation.

Purpose:
Figure out which TYPES of information are actually helping the Random Forest
instead of adding features blindly.

Every experiment uses the same time-aware folds:
- train 2021-2022 -> test 2023
- train 2021-2023 -> test 2024
- train 2021-2024 -> test 2025

2026 remains excluded from model selection.
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
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_feature_group_results.csv"

TARGET = "actual_fantasy_points"

FOLDS = [
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
]

QB_HISTORY = [
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
]

OPPONENT_DEFENSE = [
    "opp_avg_pass_yards_allowed_last_3",
    "opp_avg_pass_yards_allowed_last_5",
    "opp_avg_pass_tds_allowed_last_3",
    "opp_avg_pass_tds_allowed_last_5",
    "opp_avg_def_interceptions_last_3",
    "opp_avg_def_interceptions_last_5",
    "opp_avg_def_sacks_last_3",
    "opp_avg_def_sacks_last_5",
]

ROLE = [
    "previous_offense_pct",
    "avg_offense_pct_last_3",
    "avg_offense_pct_last_5",
    "start_like_games_last_3",
    "start_like_games_last_5",
]

HOME_AWAY = ["home_away"]
TEAM_IDENTITIES = ["team", "opponent"]

REST_CONTEXT = [
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

MARKET_CONTEXT = [
    "team_spread_line",
    "game_total_line",
]


def available(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in df.columns]


def make_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
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

    qb = available(df, QB_HISTORY)
    opp = available(df, OPPONENT_DEFENSE)
    role = available(df, ROLE)
    home = available(df, HOME_AWAY)
    identities = available(df, TEAM_IDENTITIES)
    rest = available(df, REST_CONTEXT)
    environment = available(df, ENVIRONMENT)
    market = available(df, MARKET_CONTEXT)

    current = qb + opp + role + home + identities

    experiments = [
        ("1. QB history only", qb),
        ("2. + opponent defense", qb + opp),
        ("3. + role / snap share", qb + opp + role),
        ("4. + home / away", qb + opp + role + home),
        ("5. + team / opponent IDs", current),
        ("6. + rest / neutral site", current + rest),
        ("7. + environment", current + rest + environment),
        ("8. + betting market", current + rest + environment + market),
    ]

    results = []

    print("GRIDIRONIQ QB FEATURE-GROUP TEST")
    print("=" * 55)
    print("Model: Random Forest")
    print("2026 is excluded.")
    print("Lower MAE is better.\n")

    for experiment_name, features in experiments:
        categorical = [
            c for c in features
            if c in {"team", "opponent", "home_away", "roof", "surface"}
        ]
        numeric = [c for c in features if c not in categorical]

        fold_maes = []

        for train_start, train_end, test_season in FOLDS:
            train = df[df["season"].between(train_start, train_end)]
            test = df[df["season"].eq(test_season)]

            pipeline = make_pipeline(numeric, categorical)
            pipeline.fit(train[features], train[TARGET])
            predictions = pipeline.predict(test[features])

            mae = mean_absolute_error(test[TARGET], predictions)
            fold_maes.append(mae)

            results.append(
                {
                    "experiment": experiment_name,
                    "test_season": test_season,
                    "feature_count": len(features),
                    "mae": mae,
                }
            )

        average_mae = sum(fold_maes) / len(fold_maes)
        print(
            f"{experiment_name:<27} "
            f"features={len(features):>2}  "
            f"2023={fold_maes[0]:.3f}  "
            f"2024={fold_maes[1]:.3f}  "
            f"2025={fold_maes[2]:.3f}  "
            f"AVG={average_mae:.3f}"
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)

    summary = (
        results_df.groupby(
            ["experiment", "feature_count"], as_index=False
        )
        .agg(average_mae=("mae", "mean"))
        .sort_values("average_mae")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(
        summary.to_string(
            index=False,
            formatters={"average_mae": "{:.3f}".format},
        )
    )

    best = summary.iloc[0]
    print(
        f"\nBest feature set: {best['experiment']} "
        f"(MAE {best['average_mae']:.3f})"
    )
    print(f"Saved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
