"""Analyze the validation-selected GridironIQ QB model.

Outputs:
- feature importances for the saved Random Forest model
- 2026 predictions versus actual results
- data/processed/qb_2026_predictions.csv
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_model_dataset.csv"
MODEL_FILE = ROOT / "models" / "qb_random_forest.joblib"
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_2026_predictions.csv"

TARGET = "actual_fantasy_points"

IDENTIFIER_COLUMNS = {
    "player_id",
    "player_name",
    "season",
    "week",
}

CATEGORICAL_FEATURES = ["team", "opponent"]


def main() -> None:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    model = joblib.load(MODEL_FILE)

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS | {TARGET}
    ]

    test = df[df["season"].eq(2026) & df[TARGET].notna()].copy()
    test["predicted_fantasy_points"] = model.predict(test[feature_columns])
    test["prediction_error"] = (
        test["predicted_fantasy_points"] - test[TARGET]
    )
    test["absolute_error"] = test["prediction_error"].abs()

    output_columns = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "predicted_fantasy_points",
        TARGET,
        "prediction_error",
        "absolute_error",
    ]

    test[output_columns].sort_values(
        ["week", "predicted_fantasy_points"],
        ascending=[True, False],
    ).to_csv(OUTPUT_FILE, index=False)

    print(f"2026 rows analyzed: {len(test):,}")
    print(f"2026 MAE: {mean_absolute_error(test[TARGET], test['predicted_fantasy_points']):.3f}")

    print("\nBEST 10 PREDICTIONS (smallest error)")
    print(
        test.nsmallest(10, "absolute_error")[output_columns]
        .to_string(index=False)
    )

    print("\nWORST 10 PREDICTIONS (largest error)")
    print(
        test.nlargest(10, "absolute_error")[output_columns]
        .to_string(index=False)
    )

    preprocessor = model.named_steps["preprocessor"]
    random_forest = model.named_steps["model"]

    transformed_feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\nTOP 20 MODEL FEATURES")
    print(importance.head(20).to_string(index=False, formatters={
        "importance": "{:.4f}".format,
    }))

    print(f"\nSaved 2026 predictions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
