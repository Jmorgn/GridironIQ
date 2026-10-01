"""Experiment: model WR fantasy production on both sides of the 65% snap threshold.

Current production WR v2 assumes zero fantasy points below the 65% snap role:
    P(role) * E(FP | role)

This experiment tests a more complete mixture of experts:
    P(role) * E(FP | role) + (1-P(role)) * E(FP | no role)

It also evaluates direct candidate Gradient Boosting as a control.
All models use exactly the same 2023-2025 walk-forward folds, pregame
features, and historical candidate population as the official WR v2 model.

This is research only: it does NOT replace or retrain the saved production
WR model, change rankings, or reuse 2026 outcomes for model selection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from walk_forward_wr_v2 import (
    DATA_FILE,
    FOLDS,
    IDENTIFIERS,
    TARGET,
    make_regressor,
    make_role_classifier,
    regression_metrics,
    subset_mae,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "processed"
RESULTS_FILE = OUTPUT_DIR / "wr_two_conditional_results.csv"
PREDICTIONS_FILE = OUTPUT_DIR / "wr_two_conditional_oof_predictions.csv"

ROLE_TARGET = "snap_65_role"
PRODUCTION_NAME = "Current: zero below 65%"
MIXTURE_NAME = "Proposed: two conditionals"
DIRECT_NAME = "Control: direct GB"
MODEL_ORDER = [
    PRODUCTION_NAME,
    MIXTURE_NAME,
    DIRECT_NAME,
]


def mae_for(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    mask: pd.Series,
) -> float:
    """Evaluate the same fixed pregame/test cohort for every model."""
    return subset_mae(frame, predictions, mask)


def make_result_row(
    *,
    test: pd.DataFrame,
    prediction: np.ndarray,
    model_name: str,
    test_season: int,
    top24_mask: pd.Series,
) -> dict:
    all_mae, all_rmse = regression_metrics(
        test[TARGET],
        prediction,
    )
    role = test[ROLE_TARGET].eq(1)
    nonrole = ~role
    played = test["played_any_snap"].eq(1)

    # Positive bias means the model overpredicts this group on average.
    nonrole_bias = float(
        np.mean(
            prediction[nonrole.to_numpy()]
            - test.loc[nonrole, TARGET].to_numpy(dtype=float)
        )
    ) if nonrole.any() else float("nan")

    return {
        "test_season": test_season,
        "model": model_name,
        "rows": len(test),
        "mae_all": all_mae,
        "rmse_all": all_rmse,
        "mae_wr1": mae_for(
            test, prediction, test["listed_wr1"].eq(1)
        ),
        "mae_wr2": mae_for(
            test, prediction, test["depth_chart_wr_rank"].eq(2)
        ),
        "mae_wr3": mae_for(
            test, prediction, test["depth_chart_wr_rank"].eq(3)
        ),
        "mae_played": mae_for(test, prediction, played),
        "mae_role": mae_for(test, prediction, role),
        "mae_nonrole": mae_for(test, prediction, nonrole),
        "mae_nonrole_played": mae_for(
            test, prediction, nonrole & played
        ),
        # One fixed cohort selected from the CURRENT model's pregame
        # projection for each season/week. This avoids giving competing
        # models different subsets when comparing lineup-relevant errors.
        "mae_baseline_top24": mae_for(
            test, prediction, top24_mask
        ),
        "nonrole_mean_signed_error": nonrole_bias,
    }


def run_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    test_season: int,
) -> tuple[list[dict], pd.DataFrame]:
    print(
        f"\nFold: train 2021-{test_season - 1} -> "
        f"test {test_season}"
    )
    print(
        f"Train: {len(train):,} | Test: {len(test):,}"
    )

    high_train = train[
        train[ROLE_TARGET].eq(1)
    ].copy()
    low_train = train[
        train[ROLE_TARGET].eq(0)
    ].copy()
    print(
        f"Role >=65%: {len(high_train):,} train rows | "
        f"Below 65%: {len(low_train):,} train rows"
    )

    if high_train.empty or low_train.empty:
        raise RuntimeError(
            "Both role and nonrole rows are required "
            "to fit the mixture model."
        )

    role_model = make_role_classifier(features)
    role_model.fit(
        train[features],
        train[ROLE_TARGET].astype(int),
    )
    probability = role_model.predict_proba(
        test[features]
    )[:, 1]

    high_model = make_regressor(features)
    high_model.fit(
        high_train[features],
        high_train[TARGET],
    )
    high_points = np.maximum(
        high_model.predict(test[features]),
        0.0,
    )

    low_model = make_regressor(features)
    low_model.fit(
        low_train[features],
        low_train[TARGET],
    )
    low_points = np.maximum(
        low_model.predict(test[features]),
        0.0,
    )

    direct_model = make_regressor(features)
    direct_model.fit(
        train[features],
        train[TARGET],
    )
    direct_points = np.maximum(
        direct_model.predict(test[features]),
        0.0,
    )

    current_points = probability * high_points
    mixture_points = (
        probability * high_points
        + (1.0 - probability) * low_points
    )

    # "Top 24" uses only the current model's pregame projections.
    # It is fixed across all three models, so error differences are fair.
    ranking_frame = test[
        ["season", "week"]
    ].copy()
    ranking_frame["_baseline_projection"] = current_points
    top24_mask = (
        ranking_frame.groupby(["season", "week"])[
            "_baseline_projection"
        ]
        .rank(method="first", ascending=False)
        .le(24)
    )

    predictions_by_model = {
        PRODUCTION_NAME: current_points,
        MIXTURE_NAME: mixture_points,
        DIRECT_NAME: direct_points,
    }

    rows = []
    for name in MODEL_ORDER:
        row = make_result_row(
            test=test,
            prediction=predictions_by_model[name],
            model_name=name,
            test_season=test_season,
            top24_mask=top24_mask,
        )
        rows.append(row)
        print(
            f"  {name:<28} "
            f"ALL={row['mae_all']:.3f} "
            f"WR1={row['mae_wr1']:.3f} "
            f"PLAYED={row['mae_played']:.3f} "
            f"BELOW65={row['mae_nonrole']:.3f} "
            f"PLAYED-BELOW65={row['mae_nonrole_played']:.3f} "
            f"TOP24={row['mae_baseline_top24']:.3f}"
        )

    prediction_rows = test[
        [
            "player_id",
            "player_name",
            "season",
            "week",
            "team",
            "depth_chart_wr_rank",
            "listed_wr1",
            "played_any_snap",
            ROLE_TARGET,
            TARGET,
        ]
    ].copy()

    prediction_rows["probability_65_snap_role"] = probability
    prediction_rows["conditional_fp_65_plus"] = high_points
    prediction_rows["conditional_fp_below_65"] = low_points
    prediction_rows["current_projection"] = current_points
    prediction_rows["mixture_projection"] = mixture_points
    prediction_rows["direct_projection"] = direct_points
    prediction_rows["mixture_minus_current"] = (
        mixture_points - current_points
    )
    prediction_rows["current_top24"] = (
        top24_mask.astype(int)
    )

    return rows, prediction_rows


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing wr_v2_candidate_dataset.csv. "
            "Run build_wr_v2_candidate_dataset.py first."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    historical = (
        df[
            df["season"].between(2021, 2025)
            & df["game_completed"].eq(1)
            & df[TARGET].notna()
            & df[ROLE_TARGET].notna()
        ]
        .copy()
        .reset_index(drop=True)
    )

    features = [
        column
        for column in historical.columns
        if column not in IDENTIFIERS
    ]

    print("GRIDIRONIQ WR TWO-CONDITIONAL EXPERIMENT")
    print("=" * 88)
    print(
        "Current: P(65%+ snaps) x "
        "E(FP | 65%+ snaps)"
    )
    print(
        "Proposed: P(65%+ snaps) x "
        "E(FP | 65%+ snaps) + "
        "P(below 65%) x E(FP | below 65%)"
    )
    print(
        "Control: direct candidate Gradient Boosting"
    )
    print("2026 excluded from model selection.")
    print(
        f"Historical candidate rows: {len(historical):,}"
    )
    print(f"Pregame features: {len(features)}")

    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []

    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(
                train_start, train_end
            )
        ].copy().reset_index(drop=True)
        test = historical[
            historical["season"].eq(test_season)
        ].copy().reset_index(drop=True)

        fold_rows, prediction_rows = run_fold(
            train,
            test,
            features,
            test_season,
        )
        all_rows.extend(fold_rows)
        all_predictions.append(prediction_rows)

    results = pd.DataFrame(all_rows)
    predictions = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)
    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    metric_columns = [
        "mae_all",
        "mae_wr1",
        "mae_wr2",
        "mae_wr3",
        "mae_played",
        "mae_role",
        "mae_nonrole",
        "mae_nonrole_played",
        "mae_baseline_top24",
        "nonrole_mean_signed_error",
    ]
    summary = (
        results.groupby("model", as_index=False)[
            metric_columns
        ]
        .mean()
    )
    summary["model"] = pd.Categorical(
        summary["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )
    summary = summary.sort_values("model")

    print("\n" + "=" * 88)
    print(
        "AVERAGE WALK-FORWARD RESULTS, 2023-2025"
    )
    print("=" * 88)
    print(
        summary.to_string(
            index=False,
            formatters={
                column: "{:.3f}".format
                for column in metric_columns
            },
        )
    )

    print("\nCHANGE VS CURRENT PRODUCTION MODEL")
    baseline = summary.loc[
        summary["model"].eq(
            PRODUCTION_NAME
        )
    ].iloc[0]
    mixture = summary.loc[
        summary["model"].eq(
            MIXTURE_NAME
        )
    ].iloc[0]

    for metric in metric_columns:
        diff = (
            mixture[metric]
            - baseline[metric]
        )
        print(
            f"  {metric:<30} "
            f"{diff:+.3f}"
        )

    print(
        "\nInterpretation: negative MAE differences "
        "mean the proposed model made smaller errors. "
        "For signed error, closer to zero is better."
    )
    print(
        "TOP24 refers to the same 24 receivers per "
        "season/week, selected using the current "
        "model's pregame projection."
    )
    print(
        "\nThis experiment does NOT replace "
        "the production WR bundle, rankings, "
        "uncertainty intervals, or comparison tool."
    )
    print(f"\nSaved fold results: {RESULTS_FILE}")
    print(
        f"Saved per-player out-of-fold predictions: "
        f"{PREDICTIONS_FILE}"
    )


if __name__ == "__main__":
    main()
