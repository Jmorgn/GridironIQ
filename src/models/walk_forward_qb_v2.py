"""Walk-forward validation for GridironIQ v2 two-stage QB prediction.

Stage 1:
    Predict whether a listed QB will receive a meaningful role
    (>= 50% offensive snaps).

Stage 2:
    For meaningful-role QB rows, predict fantasy points.

Final expected fantasy points:
    P(meaningful role) * predicted fantasy points if meaningful role

A direct one-stage Random Forest trained on all candidate rows is included as a
control. 2026 is excluded from model selection.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_v2_candidate_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "qb_v2_walk_forward_results.csv"
PREDICTION_FILE = ROOT / "data" / "processed" / "qb_v2_future_predictions.csv"

TARGET = "actual_fantasy_points"
ROLE_TARGET = "start_like_role"

IDENTIFIERS = {
    "player_id",
    "player_name",
    "season",
    "week",
    "game_completed",
    TARGET,
    "played_any_snap",
    ROLE_TARGET,
    "offense_snaps",
    "offense_pct",
}

CATEGORICAL = {"team", "opponent", "home_away", "roof", "surface"}

FOLDS = [
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [c for c in features if c in CATEGORICAL]
    numeric = [c for c in features if c not in CATEGORICAL]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
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

    return ColumnTransformer(transformers=transformers)


def make_role_classifier(features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(features)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_regressor(features: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(features)),
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
    )


def regression_metrics(y_true, y_pred) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse


def subset_mae(
    test: pd.DataFrame,
    predictions: np.ndarray,
    mask: pd.Series,
) -> float:
    if not mask.any():
        return float("nan")
    return mean_absolute_error(
        test.loc[mask, TARGET],
        predictions[mask.to_numpy()],
    )


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    test_season: int,
) -> list[dict]:
    X_train = train[features]
    X_test = test[features]

    role_model = make_role_classifier(features)
    role_model.fit(X_train, train[ROLE_TARGET].astype(int))

    role_probability = role_model.predict_proba(X_test)[:, 1]
    role_pred = (role_probability >= 0.50).astype(int)

    role_rows = train[train[ROLE_TARGET].eq(1)].copy()
    conditional_model = make_regressor(features)
    conditional_model.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    conditional_points = np.maximum(
        conditional_model.predict(X_test),
        0.0,
    )
    expected_points = role_probability * conditional_points

    direct_model = make_regressor(features)
    direct_model.fit(X_train, train[TARGET])
    direct_points = np.maximum(direct_model.predict(X_test), 0.0)

    rows = []

    for name, pred in [
        ("Two-stage expected points", expected_points),
        ("Direct candidate regressor", direct_points),
    ]:
        mae, rmse = regression_metrics(test[TARGET], pred)

        rows.append(
            {
                "test_season": test_season,
                "model": name,
                "rows": len(test),
                "mae_all_candidates": mae,
                "rmse_all_candidates": rmse,
                "mae_listed_qb1": subset_mae(
                    test,
                    pred,
                    test["listed_qb1"].eq(1),
                ),
                "mae_actual_role": subset_mae(
                    test,
                    pred,
                    test[ROLE_TARGET].eq(1),
                ),
                "mae_actual_played": subset_mae(
                    test,
                    pred,
                    test["played_any_snap"].eq(1),
                ),
            }
        )

    classifier_row = {
        "test_season": test_season,
        "model": "Role classifier",
        "rows": len(test),
        "role_accuracy": accuracy_score(
            test[ROLE_TARGET].astype(int),
            role_pred,
        ),
        "role_precision": precision_score(
            test[ROLE_TARGET].astype(int),
            role_pred,
            zero_division=0,
        ),
        "role_recall": recall_score(
            test[ROLE_TARGET].astype(int),
            role_pred,
            zero_division=0,
        ),
        "role_auc": roc_auc_score(
            test[ROLE_TARGET].astype(int),
            role_probability,
        ),
        "role_brier": brier_score_loss(
            test[ROLE_TARGET].astype(int),
            role_probability,
        ),
    }
    rows.append(classifier_row)

    print(
        f"\n{test_season} role classifier | "
        f"AUC={classifier_row['role_auc']:.3f}  "
        f"Brier={classifier_row['role_brier']:.3f}  "
        f"Precision={classifier_row['role_precision']:.3f}  "
        f"Recall={classifier_row['role_recall']:.3f}"
    )

    for row in rows[:2]:
        print(
            f"{row['model']:<28} "
            f"ALL={row['mae_all_candidates']:.3f}  "
            f"QB1={row['mae_listed_qb1']:.3f}  "
            f"ROLE={row['mae_actual_role']:.3f}  "
            f"PLAYED={row['mae_actual_played']:.3f}"
        )

    return rows


def fit_and_predict_future(
    historical: pd.DataFrame,
    future: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    role_model = make_role_classifier(features)
    role_model.fit(
        historical[features],
        historical[ROLE_TARGET].astype(int),
    )

    role_rows = historical[historical[ROLE_TARGET].eq(1)].copy()
    conditional_model = make_regressor(features)
    conditional_model.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    direct_model = make_regressor(features)
    direct_model.fit(historical[features], historical[TARGET])

    out = future[
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

    if len(out) == 0:
        return out

    X = future[features]

    out["meaningful_role_probability"] = (
        role_model.predict_proba(X)[:, 1]
    )
    out["conditional_fantasy_points"] = np.maximum(
        conditional_model.predict(X),
        0.0,
    )
    out["expected_fantasy_points"] = (
        out["meaningful_role_probability"]
        * out["conditional_fantasy_points"]
    )
    out["direct_fantasy_points"] = np.maximum(
        direct_model.predict(X),
        0.0,
    )

    return out.sort_values(
        ["week", "expected_fantasy_points"],
        ascending=[True, False],
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
        c
        for c in historical.columns
        if c not in IDENTIFIERS
    ]

    print("GRIDIRONIQ QB V2 WALK-FORWARD VALIDATION")
    print("=" * 72)
    print("2026 excluded from model selection.")
    print(f"Historical candidate rows: {len(historical):,}")
    print(f"Pregame features: {len(features)}")
    print(
        "Two-stage formula: P(meaningful role) x "
        "fantasy points if meaningful role"
    )

    all_rows = []

    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(train_start, train_end)
        ].copy()
        test = historical[
            historical["season"].eq(test_season)
        ].copy()

        print(
            f"\nFold: train {train_start}-{train_end} "
            f"-> test {test_season}"
        )
        print(
            f"Training candidates: {len(train):,} | "
            f"Test candidates: {len(test):,}"
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    reg = results[results["model"].ne("Role classifier")].copy()

    print("\n" + "=" * 72)
    print("AVERAGE REGRESSION RESULTS, 2023-2025")
    print("=" * 72)

    summary = (
        reg.groupby("model", as_index=False)
        .agg(
            all_candidate_mae=("mae_all_candidates", "mean"),
            qb1_mae=("mae_listed_qb1", "mean"),
            actual_role_mae=("mae_actual_role", "mean"),
            actual_played_mae=("mae_actual_played", "mean"),
        )
        .sort_values("all_candidate_mae")
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "all_candidate_mae": "{:.3f}".format,
                "qb1_mae": "{:.3f}".format,
                "actual_role_mae": "{:.3f}".format,
                "actual_played_mae": "{:.3f}".format,
            },
        )
    )

    cls = results[results["model"].eq("Role classifier")]

    print("\n" + "=" * 72)
    print("AVERAGE ROLE-CLASSIFIER RESULTS, 2023-2025")
    print("=" * 72)
    print(f"AUC:       {cls['role_auc'].mean():.3f}")
    print(f"Brier:     {cls['role_brier'].mean():.3f}")
    print(f"Precision: {cls['role_precision'].mean():.3f}")
    print(f"Recall:    {cls['role_recall'].mean():.3f}")
    print(f"Accuracy:  {cls['role_accuracy'].mean():.3f}")

    future = df[
        df["season"].eq(2026)
        & df["game_completed"].eq(0)
    ].copy()

    future_predictions = fit_and_predict_future(
        historical,
        future,
        features,
    )
    future_predictions.to_csv(PREDICTION_FILE, index=False)

    if len(future_predictions):
        print("\n" + "=" * 72)
        print("CURRENT 2026 FUTURE CANDIDATE PREDICTIONS")
        print("=" * 72)
        print(
            future_predictions.to_string(
                index=False,
                formatters={
                    "meaningful_role_probability": "{:.1%}".format,
                    "conditional_fantasy_points": "{:.2f}".format,
                    "expected_fantasy_points": "{:.2f}".format,
                    "direct_fantasy_points": "{:.2f}".format,
                },
            )
        )

    print(f"\nSaved validation results to: {OUTPUT_FILE}")
    print(f"Saved future predictions to:  {PREDICTION_FILE}")
    print(
        "\nImportant: v1's 8.470 MAE was measured on QBs who recorded game "
        "activity. v2's all-candidate MAE includes inactive/zero-point rows, "
        "so those two headline MAEs are not an apples-to-apples comparison."
    )


if __name__ == "__main__":
    main()
