"""Train the first GridironIQ quarterback regression models.

Time-aware split:
- 2021-2024: training
- 2025: validation
- 2026: live/test

The model predicts actual_fantasy_points from features that were created using
only games completed before the game being predicted.
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

    train = df[df["season"].between(2021, 2024)].copy()
    validation = df[df["season"].eq(2025)].copy()
    test = df[df["season"].eq(2026)].copy()

    X_train = train[feature_columns]
    y_train = train[TARGET]

    X_validation = validation[feature_columns]
    y_validation = validation[TARGET]

    X_test = test[feature_columns]
    y_test = test[TARGET]

    print("Time-aware dataset split")
    print(f"Training rows (2021-2024): {len(train):,}")
    print(f"Validation rows (2025):    {len(validation):,}")
    print(f"2026 live/test rows:       {len(test):,}")
    print(f"Model features:            {len(feature_columns)}")

    # Two intentionally simple baselines.
    validation_results = []

    global_mean_prediction = [y_train.mean()] * len(validation)
    validation_results.append(
        evaluate("Training Mean Baseline", y_validation, global_mean_prediction)
    )

    last3_validation = validation["avg_fp_last_3"].fillna(y_train.mean())
    validation_results.append(
        evaluate("Last-3 FP Baseline", y_validation, last3_validation)
    )

    model_specs = {
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

    trained_models = {}

    for name, estimator in model_specs.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", make_preprocessor(numeric_features)),
                ("model", estimator),
            ]
        )

        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        validation_predictions = pipeline.predict(X_validation)
        result = evaluate(name, y_validation, validation_predictions)
        validation_results.append(result)
        trained_models[name] = pipeline

    results_df = (
        pd.DataFrame(validation_results)
        .sort_values("mae")
        .reset_index(drop=True)
    )

    print("\n2025 VALIDATION RESULTS")
    print(results_df.to_string(index=False, formatters={
        "mae": "{:.3f}".format,
        "rmse": "{:.3f}".format,
        "r2": "{:.3f}".format,
    }))

    ml_results = results_df[
        results_df["model"].isin(model_specs.keys())
    ]
    best_name = ml_results.iloc[0]["model"]
    best_model = trained_models[best_name]

    print(f"\nBest ML model by 2025 MAE: {best_name}")

    if len(test) > 0:
        test_predictions = best_model.predict(X_test)
        test_result = evaluate(best_name, y_test, test_predictions)

        last3_test = test["avg_fp_last_3"].fillna(y_train.mean())
        last3_test_result = evaluate("Last-3 FP Baseline", y_test, last3_test)

        print("\n2026 LIVE/TEST CHECK")
        print(
            pd.DataFrame([last3_test_result, test_result]).to_string(
                index=False,
                formatters={
                    "mae": "{:.3f}".format,
                    "rmse": "{:.3f}".format,
                    "r2": "{:.3f}".format,
                },
            )
        )
        print(
            "\nNote: 2026 is an incomplete live season, so these test metrics "
            "will move as new games are added."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = best_name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"qb_{safe_name}.joblib"
    best_model_path = MODEL_DIR / "qb_best_model.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(best_model, best_model_path)

    print(f"\nSaved selected model to: {model_path}")
    print(f"Best-model alias:         {best_model_path}")


if __name__ == "__main__":
    main()
