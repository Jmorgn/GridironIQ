"""Train the official GridironIQ Kicker v2 direct-regression model.

Production selection from 2023-2025 walk-forward experiments:
- Gradient Boosting
- "All no IDs" feature group
- Direct fantasy-point regression (no activity/role gate)

Average walk-forward MAE was 3.831. Team/opponent identity did not
improve the average MAE, so production excludes IDs for robustness.
The active-kicker two-stage model also underperformed direct GB.

Uncertainty is an empirical 80% interval from exact out-of-fold
2023-2025 residuals. Kicker has no separate role model, so the
shared residual utility receives a constant confidence of 1.0.
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
from walk_forward_k_v2 import (
    FOLDS,
    TARGET,
    FEATURE_GROUPS,
    make_regressor,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "k_v1_candidate_dataset.csv"
MODEL_DIR = ROOT / "models"
BUNDLE_FILE = MODEL_DIR / "k_v2_bundle.joblib"
OUTPUT_FILE = ROOT / "data" / "processed" / "k_v2_official_predictions.csv"

FEATURES = FEATURE_GROUPS["All no IDs"]
WALK_FORWARD_MAE = 3.831


def build_uncertainty(
    historical: pd.DataFrame,
) -> dict:
    actual_parts = []
    pred_parts = []
    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(
                train_start, train_end
            )
        ].copy()
        test = historical[
            historical["season"].eq(test_season)
        ].copy()
        model = make_regressor(
            "Gradient Boosting", FEATURES
        )
        model.fit(train[FEATURES], train[TARGET])
        pred = np.maximum(
            model.predict(test[FEATURES]), 0.0
        )
        actual_parts.append(
            test[TARGET].to_numpy(dtype=float)
        )
        pred_parts.append(pred)

    actual = np.concatenate(actual_parts)
    prediction = np.concatenate(pred_parts)
    # Direct K regression has no role probability. A constant of 1
    # uses the shared high-confidence residual bucket consistently.
    confidence = np.ones(len(actual), dtype=float)
    return build_residual_calibration(
        actual,
        prediction,
        confidence,
        nominal_coverage=0.80,
        min_bucket_rows=100,
    )


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run the K candidate builder first."
        )
    df = pd.read_csv(
        DATA_FILE, low_memory=False,
        dtype={"player_id": str},
    )
    missing = [
        col for col in [
            *FEATURES, TARGET,
            "season", "week", "game_completed",
        ] if col not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "K v2 dataset missing required fields: "
            + ", ".join(missing)
        )

    historical = df[
        df["season"].between(2021, 2025)
        & df["game_completed"].eq(1)
        & df[TARGET].notna()
    ].copy()
    if historical.empty:
        raise RuntimeError("No historical Kicker candidates.")

    print("GRIDIRONIQ OFFICIAL KICKER V2")
    print("=" * 78)
    print("Selection: 2023-2025 walk-forward validation")
    print("Architecture: direct Gradient Boosting")
    print("Feature group: All no IDs")
    print(
        f"Walk-forward all-candidate MAE: "
        f"{WALK_FORWARD_MAE:.3f}"
    )
    print(
        "No K role/activity gate: direct regression performed better."
    )
    print(
        "Team/opponent identity excluded: no meaningful average gain."
    )
    print(f"Historical candidates: {len(historical):,}")
    print(f"Production features:   {len(FEATURES)}")

    uncertainty = build_uncertainty(historical)
    print("Uncertainty calibration:")
    print(f"  {calibration_summary(uncertainty)}")

    model = make_regressor(
        "Gradient Boosting", FEATURES
    )
    model.fit(
        historical[FEATURES],
        historical[TARGET],
    )

    MODEL_DIR.mkdir(
        parents=True, exist_ok=True
    )
    bundle = {
        "version": "k_v2",
        "production_method": "direct_gradient_boosting",
        "fantasy_target": TARGET,
        "feature_group": "All no IDs",
        "features": FEATURES,
        "regressor": model,
        "uncertainty": uncertainty,
        "walk_forward_mae": WALK_FORWARD_MAE,
        "has_role_model": False,
    }
    joblib.dump(bundle, BUNDLE_FILE)

    future = df[
        df["season"].eq(2026)
        & df["game_completed"].eq(0)
    ].copy()
    if future.empty:
        pd.DataFrame().to_csv(
            OUTPUT_FILE, index=False
        )
        print("No future 2026 Kicker candidates currently available.")
    else:
        projection = np.maximum(
            model.predict(future[FEATURES]), 0.0
        )
        confidence = np.ones(len(future))
        low, high = apply_residual_intervals(
            projection, confidence, uncertainty
        )
        keep = [
            "player_id", "player_name",
            "season", "week", "team", "opponent",
            "depth_chart_k_rank", "listed_k1",
        ]
        output = future[keep].copy()
        output["gridironiq_v2_fantasy_points"] = projection
        output["prediction_low_80"] = low
        output["prediction_high_80"] = high
        output = output.sort_values(
            ["week", "gridironiq_v2_fantasy_points"],
            ascending=[True, False],
        )
        output.to_csv(OUTPUT_FILE, index=False)
        week = int(output["week"].min())
        print(f"\nCURRENT OFFICIAL K V2 — WEEK {week}")
        print("=" * 78)
        print(
            output[
                output["week"].eq(week)
            ].head(32).to_string(
                index=False,
                formatters={
                    "gridironiq_v2_fantasy_points":
                    "{:.2f}".format,
                    "prediction_low_80": "{:.2f}".format,
                    "prediction_high_80": "{:.2f}".format,
                },
            )
        )

    print(f"\nSaved K bundle: {BUNDLE_FILE}")
    print(f"Saved K predictions: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
