"""Train the official GridironIQ QB model after walk-forward model selection.

Random Forest was selected using walk-forward validation across 2023-2025.
This script then fits the selected model on all completed historical seasons
(2021-2025) and uses 2026 only as the current live/demo check.

The model predicts actual_fantasy_points from features available before the
game being predicted.
"""

from __future__ import annotations

from pathlib import Path
import math

import joblib
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
MODEL_DIR = ROOT / "models"

TARGET = "actual_fantasy_points"

IDENTIFIER_COLUMNS = {
    "player_id",
    "player_name",
    "season",
    "week",
}

CATEGORICAL_FEATURES = ["team", "opponent", "home_away", "roof", "surface"]


def evaluate(name: str, y_true: pd.Series, y_pred) -> dict[str, float | str]:
    return {
        "model": name,
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


def main() -> None:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df = df[df[TARGET].notna()].copy()

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS | {TARGET}
    ]

    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_FEATURES
    ]

    train = df[df["season"].between(2021, 2025)].copy()
    live = df[df["season"].eq(2026)].copy()

    X_train = train[feature_columns]
    y_train = train[TARGET]

    print("GRIDIRONIQ OFFICIAL QB MODEL")
    print("=" * 42)
    print("Selected model: Random Forest")
    print("Selection method: walk-forward validation, 2023-2025")
    print(f"Training rows (2021-2025): {len(train):,}")
    print(f"2026 live rows:            {len(live):,}")
    print(f"Model features:            {len(feature_columns)}")

    model = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features)),
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

    print("\nTraining official Random Forest on 2021-2025...")
    model.fit(X_train, y_train)

    if len(live) > 0:
        X_live = live[feature_columns]
        y_live = live[TARGET]

        live_predictions = model.predict(X_live)
        live_result = evaluate("Random Forest", y_live, live_predictions)

        last3_live = live["avg_fp_last_3"].fillna(y_train.mean())
        last3_result = evaluate("Last-3 FP Baseline", y_live, last3_live)

        print("\n2026 LIVE CHECK")
        print(
            pd.DataFrame([last3_result, live_result]).to_string(
                index=False,
                formatters={
                    "mae": "{:.3f}".format,
                    "rmse": "{:.3f}".format,
                    "r2": "{:.3f}".format,
                },
            )
        )
        print(
            "\nNote: 2026 is a live/demo season, not part of model selection."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "qb_random_forest.joblib"
    best_model_path = MODEL_DIR / "qb_best_model.joblib"

    joblib.dump(model, model_path)
    joblib.dump(model, best_model_path)

    print(f"\nSaved official model to: {model_path}")
    print(f"Best-model alias:        {best_model_path}")


if __name__ == "__main__":
    main()
