"""Test whether pregame QB depth-chart status improves GridironIQ.

Compares the current best all-context Random Forest against the same model
plus depth-chart QB rank / QB1 status.

2026 stays excluded from model selection.
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
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_depth_chart_results.csv"

TARGET = "actual_fantasy_points"

FOLDS = [
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
]

IDENTIFIERS = {
    "player_id",
    "player_name",
    "season",
    "week",
    TARGET,
    "depth_chart_qb_rank",
    "listed_qb1",
}

CATEGORICAL = {"team", "opponent", "home_away", "roof", "surface"}
DEPTH_FEATURES = ["depth_chart_qb_rank", "listed_qb1"]


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

    depth = [c for c in DEPTH_FEATURES if c in df.columns]
    if len(depth) != len(DEPTH_FEATURES):
        missing = sorted(set(DEPTH_FEATURES) - set(depth))
        raise RuntimeError(
            "Depth-chart features are missing from the model table: "
            + ", ".join(missing)
        )

    base_features = [
        c for c in df.columns
        if c not in IDENTIFIERS
    ]

    experiments = [
        ("All Context", base_features),
        ("All Context + Depth Chart", base_features + depth),
    ]

    rows = []

    print("GRIDIRONIQ QB DEPTH-CHART TEST")
    print("=" * 62)
    print("Random Forest | 2026 excluded | Lower MAE is better.\n")

    coverage = df["depth_chart_qb_rank"].notna().mean() * 100
    starter_rate = df["listed_qb1"].dropna().mean() * 100
    print(f"Depth-chart coverage: {coverage:.1f}% of historical QB rows")
    print(f"QB1 rate among matched rows: {starter_rate:.1f}%\n")

    for name, features in experiments:
        fold_maes = []

        for train_start, train_end, test_season in FOLDS:
            train = df[df["season"].between(train_start, train_end)]
            test = df[df["season"].eq(test_season)]

            model = make_pipeline(features)
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            mae = mean_absolute_error(test[TARGET], pred)
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
            f"{name:<30} "
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

    print("\n" + "=" * 62)
    print("SUMMARY")
    print("=" * 62)
    print(
        summary.to_string(
            index=False,
            formatters={"average_mae": "{:.3f}".format},
        )
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
