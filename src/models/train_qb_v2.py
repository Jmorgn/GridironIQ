"""Train the official GridironIQ QB v2 two-stage model.

Model selection is based on 2023-2025 walk-forward validation.

Stage 1:
    Random Forest classifier estimates P(QB receives >=50% offensive snaps).

Stage 2:
    Random Forest regressor predicts fantasy points conditional on a
    starter-level role.

Production gate:
    For each team/week, only the QB with the highest role probability receives
    the conditional fantasy projection. Other listed QBs receive zero.

Uncertainty:
    An empirical 80% interval is calibrated from out-of-season 2023-2025
    residuals for the QBs selected by the same top-role-per-team rule.

2026 is used only for current live/future predictions.
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
from walk_forward_qb_v2 import (
    CATEGORICAL,
    FOLDS,
    IDENTIFIERS,
    ROLE_TARGET,
    TARGET,
    make_regressor,
    make_role_classifier,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_v2_candidate_dataset.csv"
MODEL_DIR = ROOT / "models"
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_v2_official_predictions.csv"


def select_top_role_per_team(
    frame: pd.DataFrame,
    role_probability: np.ndarray,
) -> np.ndarray:
    helper = frame[["season", "week", "team"]].copy()
    helper["_role_probability"] = role_probability
    helper["_row_index"] = np.arange(len(helper))

    winners = (
        helper.sort_values(
            ["season", "week", "team", "_role_probability"],
            ascending=[True, True, True, False],
        )
        .drop_duplicates(["season", "week", "team"])
        ["_row_index"]
        .to_numpy()
    )

    selected = np.zeros(len(frame), dtype=int)
    selected[winners] = 1
    return selected


def build_uncertainty_calibration(
    historical: pd.DataFrame,
    features: list[str],
) -> dict:
    """Calibrate intervals from walk-forward-selected QB residuals."""
    actual_parts = []
    prediction_parts = []
    probability_parts = []

    print("\nCalibrating QB uncertainty from 2023-2025 walk-forward residuals...")

    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(train_start, train_end)
        ].copy()
        test = historical[
            historical["season"].eq(test_season)
        ].copy()

        role_model = make_role_classifier(features)
        role_model.fit(
            train[features],
            train[ROLE_TARGET].astype(int),
        )

        role_probability = role_model.predict_proba(
            test[features]
        )[:, 1]

        role_rows = train[train[ROLE_TARGET].eq(1)].copy()
        points_model = make_regressor(features)
        points_model.fit(
            role_rows[features],
            role_rows[TARGET],
        )

        conditional_points = np.maximum(
            points_model.predict(test[features]),
            0.0,
        )

        selected = select_top_role_per_team(
            test,
            role_probability,
        ).astype(bool)

        actual_parts.append(
            test.loc[selected, TARGET].to_numpy(dtype=float)
        )
        prediction_parts.append(
            conditional_points[selected]
        )
        probability_parts.append(
            role_probability[selected]
        )

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

    print("GRIDIRONIQ OFFICIAL QB V2")
    print("=" * 68)
    print("Selection method: walk-forward validation, 2023-2025")
    print("Production gate: highest role probability per team/week")
    print("Uncertainty: historical walk-forward 80% residual interval")
    print(f"Historical candidate rows: {len(historical):,}")
    print(f"Pregame features:          {len(features)}")

    uncertainty = build_uncertainty_calibration(
        historical,
        features,
    )
    print("Uncertainty calibration:")
    print(f"  {calibration_summary(uncertainty)}")
    for name, bucket in uncertainty["buckets"].items():
        if bucket.get("uses_fallback"):
            source_rows = bucket.get("source_rows", 0)
            sample_text = (
                f"source n={source_rows:,}; using all-sample fallback "
                f"n={bucket['n']:,}"
            )
        else:
            sample_text = f"n={bucket['n']:,}"

        print(
            f"  {name:<6} {sample_text} | "
            f"coverage={bucket['empirical_coverage']:.1%} | "
            f"offsets={bucket['lower_residual']:+.2f}/"
            f"{bucket['upper_residual']:+.2f}"
        )

    print("\nTraining role classifier on 2021-2025...")
    role_model.fit(
        historical[features],
        historical[ROLE_TARGET].astype(int),
    )

    role_rows = historical[historical[ROLE_TARGET].eq(1)].copy()
    print(
        "Training conditional fantasy regressor on "
        f"{len(role_rows):,} starter-level-role rows..."
    )
    conditional_model.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    role_path = MODEL_DIR / "qb_v2_role_classifier.joblib"
    points_path = MODEL_DIR / "qb_v2_conditional_regressor.joblib"
    bundle_path = MODEL_DIR / "qb_v2_bundle.joblib"

    joblib.dump(role_model, role_path)
    joblib.dump(conditional_model, points_path)
    joblib.dump(
        {
            "version": "qb_v2",
            "production_gate": "top_role_probability_per_team_week",
            "role_target": ROLE_TARGET,
            "fantasy_target": TARGET,
            "features": features,
            "categorical_features": sorted(CATEGORICAL),
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
        print("\nNo future 2026 candidate rows are currently available.")
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False)
    else:
        X_future = future[features]

        role_probability = role_model.predict_proba(X_future)[:, 1]
        conditional_points = np.maximum(
            conditional_model.predict(X_future),
            0.0,
        )
        selected = select_top_role_per_team(
            future,
            role_probability,
        )

        production_points = np.where(
            selected == 1,
            conditional_points,
            0.0,
        )

        predictions = future[
            [
                "player_id",
                "player_name",
                "season",
                "week",
                "team",
                "opponent",
                "depth_chart_qb_rank",
                "listed_qb1",
            ]
        ].copy()

        predictions["meaningful_role_probability"] = role_probability
        predictions["selected_team_qb"] = selected
        predictions["conditional_fantasy_points"] = conditional_points
        predictions["soft_expected_fantasy_points"] = (
            role_probability * conditional_points
        )
        predictions["gridironiq_v2_fantasy_points"] = production_points

        predictions["prediction_low_80"] = np.nan
        predictions["prediction_high_80"] = np.nan

        selected_mask = selected.astype(bool)
        if selected_mask.any():
            low, high = apply_residual_intervals(
                production_points[selected_mask],
                role_probability[selected_mask],
                uncertainty,
            )
            predictions.loc[
                selected_mask,
                "prediction_low_80",
            ] = low
            predictions.loc[
                selected_mask,
                "prediction_high_80",
            ] = high

        predictions = predictions.sort_values(
            [
                "week",
                "gridironiq_v2_fantasy_points",
                "meaningful_role_probability",
            ],
            ascending=[True, False, False],
        )

        predictions.to_csv(OUTPUT_FILE, index=False)

        print("\nCURRENT OFFICIAL V2 FUTURE PROJECTIONS")
        print("=" * 68)
        print(
            predictions[
                predictions["selected_team_qb"].eq(1)
            ].to_string(
                index=False,
                formatters={
                    "meaningful_role_probability": "{:.1%}".format,
                    "conditional_fantasy_points": "{:.2f}".format,
                    "soft_expected_fantasy_points": "{:.2f}".format,
                    "gridironiq_v2_fantasy_points": "{:.2f}".format,
                    "prediction_low_80": "{:.2f}".format,
                    "prediction_high_80": "{:.2f}".format,
                },
            )
        )

    print(f"\nSaved role classifier:      {role_path}")
    print(f"Saved fantasy regressor:    {points_path}")
    print(f"Saved v2 model bundle:      {bundle_path}")
    print(f"Saved official predictions: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
