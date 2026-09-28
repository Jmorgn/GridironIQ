"""Test whether pregame QB injury-report features improve GridironIQ.

Compares the current v1 feature set (all context + depth chart) against the
same model plus official injury/practice-report signals.

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
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_injury_feature_results.csv"

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
}

INJURY_FEATURES = [
    "on_injury_report",
    "injury_questionable",
    "injury_doubtful",
    "injury_out",
    "practice_dnp",
    "practice_limited",
    "practice_full",
    "injury_status_score",
]

CATEGORICAL = {"team", "opponent", "home_away", "roof", "surface"}


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

    missing = [c for c in INJURY_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Injury features are missing from the model table: "
            + ", ".join(missing)
        )

    base_features = [
        c for c in df.columns
        if c not in IDENTIFIERS and c not in INJURY_FEATURES
    ]
    with_injuries = base_features + INJURY_FEATURES

    experiments = [
        ("GridironIQ v1", base_features),
        ("v1 + Injury / Practice", with_injuries),
    ]

    injury_rate = df["on_injury_report"].mean() * 100
    questionable_rate = df.loc[
        df["on_injury_report"].eq(1), "injury_questionable"
    ].mean() * 100
    limited_rate = df.loc[
        df["on_injury_report"].eq(1), "practice_limited"
    ].mean() * 100
    dnp_rate = df.loc[
        df["on_injury_report"].eq(1), "practice_dnp"
    ].mean() * 100

    print("GRIDIRONIQ QB INJURY-FEATURE TEST")
    print("=" * 66)
    print("Random Forest | 2026 excluded | Lower MAE is better.\n")
    print(f"QB rows appearing on injury report: {injury_rate:.1f}%")
    print(f"Questionable among reported rows:   {questionable_rate:.1f}%")
    print(f"Limited practice among reported:    {limited_rate:.1f}%")
    print(f"DNP practice among reported:        {dnp_rate:.1f}%\n")

    rows = []

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
            f"{name:<28} "
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

    print("\n" + "=" * 66)
    print("SUMMARY")
    print("=" * 66)
    print(
        summary.to_string(
            index=False,
            formatters={"average_mae": "{:.3f}".format},
        )
    )

    print(
        "\nImportant: the current historical training table contains QBs who "
        "recorded game activity. This test measures whether injury information "
        "helps predict players who still played; a future candidate-table "
        "upgrade will also model zero-participation / ruled-out QBs."
    )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
