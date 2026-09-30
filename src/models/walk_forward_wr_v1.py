"""Walk-forward validation for the GridironIQ WR v1 regression model.

WR v1 evaluates only wide receivers who recorded fantasy-relevant activity.
The goal is to establish a leakage-safe active-game benchmark before building
the full pregame candidate / route-and-target-role architecture.

2026 is excluded from model selection.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    ROOT / "data" / "processed" / "wr_model_dataset.csv"
)
OUTPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "wr_v1_walk_forward_results.csv"
)

TARGET = "actual_fantasy_points"
IDENTIFIERS = {
    "player_id",
    "player_name",
    "season",
    "week",
    TARGET,
}
CATEGORICAL = {
    "team",
    "opponent",
    "home_away",
    "roof",
    "surface",
}

FOLDS = [
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
]


def make_preprocessor(
    features: list[str],
    *,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    categorical = [
        c for c in features if c in CATEGORICAL
    ]
    numeric = [
        c for c in features if c not in CATEGORICAL
    ]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=numeric_steps),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent",
                            ),
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
                categorical,
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


def make_models(
    features: list[str],
) -> dict[str, Pipeline]:
    return {
        "Linear Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(
                        features,
                        scale_numeric=True,
                    ),
                ),
                ("model", LinearRegression()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(features),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        min_samples_leaf=3,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                (
                    "preprocessor",
                    make_preprocessor(features),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def metrics(
    y_true,
    y_pred,
) -> tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(
        mean_squared_error(y_true, y_pred)
    )
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    test_season: int,
) -> list[dict]:
    rows = []

    baselines = {
        "Mean baseline": np.repeat(
            train[TARGET].mean(),
            len(test),
        ),
        "Last-3 baseline": (
            pd.to_numeric(
                test["avg_fp_last_3"],
                errors="coerce",
            )
            .fillna(train[TARGET].mean())
            .to_numpy()
        ),
    }

    for name, pred in baselines.items():
        mae, rmse, r2 = metrics(
            test[TARGET],
            pred,
        )
        rows.append(
            {
                "test_season": test_season,
                "model": name,
                "rows": len(test),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

    for name, model in make_models(features).items():
        model.fit(
            train[features],
            train[TARGET],
        )
        pred = np.maximum(
            model.predict(test[features]),
            0.0,
        )
        mae, rmse, r2 = metrics(
            test[TARGET],
            pred,
        )

        rows.append(
            {
                "test_season": test_season,
                "model": name,
                "rows": len(test),
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
            }
        )

    print(f"\nTest season {test_season}")
    for row in rows:
        print(
            f"{row['model']:<20} "
            f"MAE={row['mae']:.3f}  "
            f"RMSE={row['rmse']:.3f}  "
            f"R²={row['r2']:.3f}"
        )

    return rows


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing wr_model_dataset.csv. "
            "Run build_wr_model_dataset.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    historical = df[
        df["season"].between(2021, 2025)
        & df[TARGET].notna()
    ].copy()

    features = [
        column
        for column in historical.columns
        if column not in IDENTIFIERS
    ]

    print("GRIDIRONIQ WR V1 WALK-FORWARD VALIDATION")
    print("=" * 72)
    print(
        "Population: WRs with fantasy-relevant game activity"
    )
    print("2026 excluded from model selection.")
    print(
        f"Historical WR-games: {len(historical):,}"
    )
    print(
        f"Pregame features:    {len(features)}"
    )

    all_rows = []

    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(
                train_start,
                train_end,
            )
        ].copy()
        test = historical[
            historical["season"].eq(test_season)
        ].copy()

        print(
            f"\nFold: train {train_start}-{train_end} "
            f"-> test {test_season}"
        )
        print(
            f"Train rows: {len(train):,} | "
            f"Test rows: {len(test):,}"
        )

        all_rows.extend(
            evaluate_fold(
                train,
                test,
                features,
                test_season,
            )
        )

    results = pd.DataFrame(all_rows)
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 72)
    print("AVERAGE RESULTS, 2023-2025")
    print("=" * 72)

    summary = (
        results.groupby(
            "model",
            as_index=False,
        )
        .agg(
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            r2=("r2", "mean"),
        )
        .sort_values("mae")
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "mae": "{:.3f}".format,
                "rmse": "{:.3f}".format,
                "r2": "{:.3f}".format,
            },
        )
    )

    best = summary.iloc[0]
    last3 = summary[
        summary["model"].eq(
            "Last-3 baseline"
        )
    ].iloc[0]

    improvement = (
        (last3["mae"] - best["mae"])
        / last3["mae"]
        * 100
    )

    print(
        f"\nBest by walk-forward MAE: "
        f"{best['model']} ({best['mae']:.3f})"
    )
    print(
        "Improvement vs Last-3 baseline: "
        f"{improvement:.1f}%"
    )

    live = df[
        df["season"].eq(2026)
        & df[TARGET].notna()
    ].copy()

    if len(live):
        print(
            "\n2026 live/demo rows currently available: "
            f"{len(live):,}"
        )
        print(
            "2026 is NOT used to choose the model. "
            "We will evaluate it separately after model selection."
        )

    print(f"\nSaved results to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
