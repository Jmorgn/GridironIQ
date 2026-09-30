"""Train the official GridironIQ RB v2 two-stage model.

Model selection is based on 2023-2025 walk-forward validation.

Selected production architecture:
1. Random Forest classifier estimates P(RB receives >=35% offensive snaps).
2. Gradient Boosting regressor predicts fantasy points conditional on that role.
3. Official fantasy projection is the soft expected value:
       P(35%+ snaps) * conditional fantasy points

Why this architecture:
- The snap_35 soft model produced the lowest all-candidate walk-forward MAE
  (4.167) across 2023-2025.
- RBs are not forced into one-player-per-team gates because committees are real.
- 2026 is held out of model selection and used only for live/future predictions.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from walk_forward_rb_v2 import (
    CATEGORICAL,
    IDENTIFIERS,
    TARGET,
    make_regressor,
    make_role_classifier,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "rb_v2_candidate_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "rb_v2_official_predictions.csv"
)

ROLE_TARGET = "snap_35_role"


def main() -> None:
    df = pd.read_csv(DATA_FILE, low_memory=False)

    historical = df[
        df["season"].between(2021, 2025)
        & df["game_completed"].eq(1)
        & df[TARGET].notna()
        & df[ROLE_TARGET].notna()
    ].copy()

    features = [
        column
        for column in historical.columns
        if column not in IDENTIFIERS
    ]

    role_model = make_role_classifier(features)
    conditional_model = make_regressor(features)

    print("GRIDIRONIQ OFFICIAL RB V2")
    print("=" * 68)
    print("Selection method: walk-forward validation, 2023-2025")
    print("Role target: >=35% offensive snaps")
    print(
        "Production projection: P(35%+ snaps) x "
        "conditional fantasy points"
    )
    print("Walk-forward all-candidate MAE: 4.167")
    print(f"Historical candidate rows: {len(historical):,}")
    print(f"Pregame features:          {len(features)}")

    print("\nTraining RB role classifier on 2021-2025...")
    role_model.fit(
        historical[features],
        historical[ROLE_TARGET].astype(int),
    )

    role_rows = historical[
        historical[ROLE_TARGET].eq(1)
    ].copy()

    print(
        "Training conditional fantasy regressor on "
        f"{len(role_rows):,} 35%+ snap-role rows..."
    )
    conditional_model.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    role_path = MODEL_DIR / "rb_v2_role_classifier.joblib"
    points_path = MODEL_DIR / "rb_v2_conditional_regressor.joblib"
    bundle_path = MODEL_DIR / "rb_v2_bundle.joblib"

    joblib.dump(role_model, role_path)
    joblib.dump(conditional_model, points_path)
    joblib.dump(
        {
            "version": "rb_v2",
            "production_method": "soft_expected_points",
            "role_target": ROLE_TARGET,
            "role_definition": "offense_pct >= 0.35",
            "fantasy_target": TARGET,
            "features": features,
            "categorical_features": sorted(CATEGORICAL),
            "role_classifier": role_model,
            "conditional_regressor": conditional_model,
        },
        bundle_path,
    )

    future = df[
        df["season"].eq(2026)
        & df["game_completed"].eq(0)
    ].copy()

    if future.empty:
        print("\nNo future 2026 RB candidate rows are currently available.")
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
    else:
        X = future[features]
        probability = role_model.predict_proba(X)[:, 1]
        conditional_points = np.maximum(
            conditional_model.predict(X),
            0.0,
        )
        projection = probability * conditional_points

        predictions = future[
            [
                "player_id",
                "player_name",
                "season",
                "week",
                "team",
                "opponent",
                "depth_chart_rb_rank",
                "listed_rb1",
            ]
        ].copy()

        predictions["rb_role_probability"] = probability
        predictions["conditional_fantasy_points"] = conditional_points
        predictions["gridironiq_v2_fantasy_points"] = projection

        predictions = predictions.sort_values(
            [
                "week",
                "gridironiq_v2_fantasy_points",
                "rb_role_probability",
            ],
            ascending=[True, False, False],
        )

        predictions.to_csv(OUTPUT_FILE, index=False)

        upcoming_week = int(
            pd.to_numeric(
                predictions["week"],
                errors="coerce",
            ).dropna().min()
        )

        current = predictions[
            predictions["week"].eq(upcoming_week)
        ].copy()

        print(
            f"\nCURRENT OFFICIAL RB V2 PROJECTIONS — "
            f"WEEK {upcoming_week}"
        )
        print("=" * 68)
        print(
            current.head(40).to_string(
                index=False,
                formatters={
                    "rb_role_probability": "{:.1%}".format,
                    "conditional_fantasy_points": "{:.2f}".format,
                    "gridironiq_v2_fantasy_points": "{:.2f}".format,
                },
            )
        )

    print(f"\nSaved RB role classifier:   {role_path}")
    print(f"Saved RB fantasy regressor: {points_path}")
    print(f"Saved RB v2 model bundle:   {bundle_path}")
    print(f"Saved official predictions: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
