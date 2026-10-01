"""Train the official GridironIQ TE v2 two-stage model.

Walk-forward selection (2023-2025): the 50% snap-share Random Forest
classifier and conditional Gradient Boosting regressor achieved the
lowest all-candidate MAE (2.603). The official point projection is
P(50%+ offensive snaps) * conditional fantasy points.

The 15% target-share model was close (2.621 all-candidate MAE), while
the 35% snap-share model was marginally stronger on a fixed projected
top-12 cohort (5.870 vs 5.879 MAE). We follow the original all-candidate
selection metric, not a post-hoc change of objective.

Snap share includes blocking; it is not a measure of receiving routes.
Residual-based 80% historical intervals use 2023-2025 walk-forward
predictions from the same production formula. 2026 is prediction-only.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uncertainty import (
    apply_residual_intervals,
    build_residual_calibration,
    calibration_summary,
)
from walk_forward_te_v2 import (
    CATEGORICAL,
    FOLDS,
    IDENTIFIERS,
    TARGET,
    make_regressor,
    make_role_classifier,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "te_v2_candidate_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "te_v2_official_predictions.csv"
)

ROLE_TARGET = "snap_50_role"


def build_uncertainty_calibration(
    historical: pd.DataFrame,
    features: list[str],
) -> dict:
    """Calibrate TE intervals from exact walk-forward production residuals."""
    actual_parts = []
    prediction_parts = []
    probability_parts = []

    print(
        "\nCalibrating TE uncertainty from "
        "2023-2025 walk-forward residuals..."
    )

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

        role_model = make_role_classifier(features)
        role_model.fit(
            train[features],
            train[ROLE_TARGET].astype(int),
        )
        probability = role_model.predict_proba(
            test[features]
        )[:, 1]

        role_rows = train[
            train[ROLE_TARGET].eq(1)
        ].copy()
        points_model = make_regressor(features)
        points_model.fit(
            role_rows[features],
            role_rows[TARGET],
        )

        conditional_points = np.maximum(
            points_model.predict(test[features]),
            0.0,
        )
        production_points = (
            probability * conditional_points
        )

        actual_parts.append(
            test[TARGET].to_numpy(dtype=float)
        )
        prediction_parts.append(production_points)
        probability_parts.append(probability)

    actual = np.concatenate(actual_parts)
    prediction = np.concatenate(prediction_parts)
    probability = np.concatenate(probability_parts)

    return build_residual_calibration(
        actual,
        prediction,
        probability,
        nominal_coverage=0.80,
        min_bucket_rows=100,
    )


def main() -> None:
    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

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

    print("GRIDIRONIQ OFFICIAL TE V2")
    print("=" * 72)
    print(
        "Selection method: walk-forward validation, "
        "2023-2025"
    )
    print("Role target: >=50% offensive snaps")
    print(
        "Production projection: P(50%+ snaps) x "
        "conditional fantasy points"
    )
    print("Walk-forward all-candidate MAE: 2.603")
    print(
        "Uncertainty: historical walk-forward "
        "80% residual interval"
    )
    print(
        f"Historical candidate rows: {len(historical):,}"
    )
    print(
        f"Pregame features:          {len(features)}"
    )

    uncertainty = build_uncertainty_calibration(
        historical,
        features,
    )

    print("Uncertainty calibration:")
    print(
        f"  {calibration_summary(uncertainty)}"
    )
    for name, bucket in uncertainty["buckets"].items():
        if bucket.get("uses_fallback"):
            source_rows = bucket.get(
                "source_rows",
                0,
            )
            sample_text = (
                f"source n={source_rows:,}; "
                "using all-sample fallback "
                f"n={bucket['n']:,}"
            )
        else:
            sample_text = (
                f"n={bucket['n']:,}"
            )

        print(
            f"  {name:<6} {sample_text} | "
            f"coverage="
            f"{bucket['empirical_coverage']:.1%} | "
            f"offsets="
            f"{bucket['lower_residual']:+.2f}/"
            f"{bucket['upper_residual']:+.2f}"
        )

    print(
        "\nTraining TE role classifier "
        "on 2021-2025..."
    )
    role_model.fit(
        historical[features],
        historical[ROLE_TARGET].astype(int),
    )

    role_rows = historical[
        historical[ROLE_TARGET].eq(1)
    ].copy()

    print(
        "Training conditional fantasy regressor on "
        f"{len(role_rows):,} 50%+ snap-role rows..."
    )
    conditional_model.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    role_path = (
        MODEL_DIR / "te_v2_role_classifier.joblib"
    )
    points_path = (
        MODEL_DIR / "te_v2_conditional_regressor.joblib"
    )
    bundle_path = (
        MODEL_DIR / "te_v2_bundle.joblib"
    )

    joblib.dump(
        role_model,
        role_path,
    )
    joblib.dump(
        conditional_model,
        points_path,
    )
    joblib.dump(
        {
            "version": "te_v2",
            "production_method": "soft_expected_points",
            "role_target": ROLE_TARGET,
            "role_definition": "offense_pct >= 0.50",
            "fantasy_target": TARGET,
            "features": features,
            "categorical_features": sorted(
                CATEGORICAL
            ),
            "role_classifier": role_model,
            "conditional_regressor": conditional_model,
            "uncertainty": uncertainty,
        },
        bundle_path,
    )

    future = df[
        df["season"].eq(2026)
        & df["game_completed"].eq(0)
    ].copy()

    if future.empty:
        print(
            "\nNo future 2026 TE candidate rows "
            "are currently available."
        )
        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
        )
    else:
        X = future[features]
        probability = role_model.predict_proba(
            X
        )[:, 1]
        conditional_points = np.maximum(
            conditional_model.predict(X),
            0.0,
        )
        projection = (
            probability * conditional_points
        )
        low, high = apply_residual_intervals(
            projection,
            probability,
            uncertainty,
        )

        predictions = future[
            [
                "player_id",
                "player_name",
                "season",
                "week",
                "team",
                "opponent",
                "depth_chart_te_rank",
                "listed_te1",
            ]
        ].copy()

        predictions[
            "te_role_probability"
        ] = probability
        predictions[
            "conditional_fantasy_points"
        ] = conditional_points
        predictions[
            "gridironiq_v2_fantasy_points"
        ] = projection
        predictions[
            "prediction_low_80"
        ] = low
        predictions[
            "prediction_high_80"
        ] = high

        predictions = predictions.sort_values(
            [
                "week",
                "gridironiq_v2_fantasy_points",
                "te_role_probability",
            ],
            ascending=[
                True,
                False,
                False,
            ],
        )

        predictions.to_csv(
            OUTPUT_FILE,
            index=False,
        )

        upcoming_week = int(
            pd.to_numeric(
                predictions["week"],
                errors="coerce",
            )
            .dropna()
            .min()
        )

        current = predictions[
            predictions["week"].eq(
                upcoming_week
            )
        ].copy()

        print(
            "\nCURRENT OFFICIAL TE V2 "
            f"PROJECTIONS — WEEK {upcoming_week}"
        )
        print("=" * 72)
        print(
            current.head(60).to_string(
                index=False,
                formatters={
                    "te_role_probability": "{:.1%}".format,
                    "conditional_fantasy_points": "{:.2f}".format,
                    "gridironiq_v2_fantasy_points": "{:.2f}".format,
                    "prediction_low_80": "{:.2f}".format,
                    "prediction_high_80": "{:.2f}".format,
                },
            )
        )

    print(
        f"\nSaved TE role classifier:   {role_path}"
    )
    print(
        f"Saved TE fantasy regressor: {points_path}"
    )
    print(
        f"Saved TE v2 model bundle:   {bundle_path}"
    )
    print(
        f"Saved official predictions: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
