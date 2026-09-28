"""Walk-forward validation for GridironIQ QB fantasy models.

This tests the model the same way we would actually use it:
train only on past seasons, then predict the next unseen season.

Folds:
- Train 2021-2022 -> test 2023
- Train 2021-2023 -> test 2024
- Train 2021-2024 -> test 2025

2026 is intentionally excluded from model selection so it can remain our
live/demo season.
"""

from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_model_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_walk_forward_results.csv"

TARGET = "actual_fantasy_points"

IDENTIFIER_COLUMNS = {
    "player_id",
    "player_name",
    "season",
    "week",
}

CATEGORICAL_FEATURES = ["team", "opponent", "home_away"]

FOLDS = [
    {"train_start": 2021, "train_end": 2022, "test_season": 2023},
    {"train_start": 2021, "train_end": 2023, "test_season": 2024},
    {"train_start": 2021, "train_end": 2024, "test_season": 2025},
]


def evaluate(
    model_name: str,
    test_season: int,
    train_rows: int,
    test_rows: int,
    y_true: pd.Series,
    y_pred,
) -> dict[str, float | int | str]:
    return {
        "model": model_name,
        "test_season": test_season,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def make_preprocessor(numeric_features: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def model_specs():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            loss="squared_error",
            random_state=42,
        ),
    }


def main() -> None:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[df[TARGET].notna()].copy()

    # Keep 2026 out of model-selection experiments.
    historical = df[df["season"].between(2021, 2025)].copy()

    feature_columns = [
        column
        for column in historical.columns
        if column not in IDENTIFIER_COLUMNS | {TARGET}
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    all_results: list[dict[str, float | int | str]] = []

    print("GRIDIRONIQ QB WALK-FORWARD VALIDATION")
    print("=" * 45)
    print("2026 is excluded from model selection.")
    print(f"Features used: {len(feature_columns)}")

    for fold in FOLDS:
        train_start = fold["train_start"]
        train_end = fold["train_end"]
        test_season = fold["test_season"]

        train = historical[
            historical["season"].between(train_start, train_end)
        ].copy()
        test = historical[historical["season"].eq(test_season)].copy()

        X_train = train[feature_columns]
        y_train = train[TARGET]
        X_test = test[feature_columns]
        y_test = test[TARGET]

        print(
            f"\nFold: train {train_start}-{train_end} -> test {test_season}"
        )
        print(f"Training rows: {len(train):,}")
        print(f"Test rows:     {len(test):,}")

        # Simple baselines.
        mean_pred = [y_train.mean()] * len(test)
        all_results.append(
            evaluate(
                "Training Mean Baseline",
                test_season,
                len(train),
                len(test),
                y_test,
                mean_pred,
            )
        )

        last3_pred = test["avg_fp_last_3"].fillna(y_train.mean())
        all_results.append(
            evaluate(
                "Last-3 FP Baseline",
                test_season,
                len(train),
                len(test),
                y_test,
                last3_pred,
            )
        )

        for name, estimator in model_specs().items():
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", make_preprocessor(numeric_features)),
                    ("model", estimator),
                ]
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            result = evaluate(
                name,
                test_season,
                len(train),
                len(test),
                y_test,
                predictions,
            )
            all_results.append(result)

            print(
                f"{name:<20} "
                f"MAE={result['mae']:.3f}  "
                f"RMSE={result['rmse']:.3f}  "
                f"R2={result['r2']:.3f}"
            )

    results = pd.DataFrame(all_results)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 45)
    print("AVERAGE ACROSS 2023-2025")
    print("=" * 45)

    summary = (
        results.groupby("model", as_index=False)
        .agg(
            average_mae=("mae", "mean"),
            average_rmse=("rmse", "mean"),
            average_r2=("r2", "mean"),
        )
        .sort_values("average_mae")
        .reset_index(drop=True)
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "average_mae": "{:.3f}".format,
                "average_rmse": "{:.3f}".format,
                "average_r2": "{:.3f}".format,
            },
        )
    )

    best_ml = summary[
        summary["model"].isin(
            ["Linear Regression", "Random Forest", "Gradient Boosting"]
        )
    ].iloc[0]

    print(
        f"\nBest ML model by walk-forward MAE: "
        f"{best_ml['model']} ({best_ml['average_mae']:.3f})"
    )

    baseline = summary[
        summary["model"].eq("Last-3 FP Baseline")
    ].iloc[0]

    improvement = (
        (baseline["average_mae"] - best_ml["average_mae"])
        / baseline["average_mae"]
        * 100
    )

    print(
        f"Improvement vs Last-3 baseline: {improvement:.1f}%"
    )
    print(f"\nSaved fold-by-fold results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
