"""Walk-forward evaluation of GridironIQ TE v2 candidate modeling.

Eight receiving, snap-share, and combined participation definitions are
compared with a role Random Forest and conditional Gradient Boosting.
Evaluate soft expected points, a 50% hard gate, and a direct candidate
Gradient Boosting control.

Besides overall errors, report TE1/TE2/TE3, played TEs, actual 3+
targets, actual double-digit scoring, and the same pregame top-12 TEs
chosen by the direct control within each test week.

Use only 2021-2025 history, testing 2023-2025 walk-forward; reserve 2026.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
)
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
DATA_FILE = (
    ROOT / "data" / "processed" / "te_v2_candidate_dataset.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "te_v2_walk_forward_results.csv"
)

TARGET = "actual_fantasy_points"
ROLE_TARGETS = [
    "snap_35_role",
    "snap_50_role",
    "target_1_role",
    "target_3_role",
    "target_5_role",
    "target_share_15_role",
    "snap35_or_target3_role",
    "snap50_or_target3_role",
]

OUTCOMES = {
    TARGET,
    "recorded_activity",
    "played_any_snap",
    *ROLE_TARGETS,
    "targets",
    "receptions",
    "rec_yards",
    "rec_tds",
    "carries",
    "rush_yards",
    "rush_tds",
    "opportunities",
    "air_yards",
    "yac",
    "rec_first_downs",
    "target_share",
    "reception_share",
    "receiving_yard_share",
    "yards_per_target",
    "catch_rate",
    "offense_snaps",
    "offense_pct",
}

IDENTIFIERS = {
    "player_id",
    "player_name",
    "season",
    "week",
    "game_completed",
    *OUTCOMES,
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
) -> ColumnTransformer:
    categorical = [
        c for c in features if c in CATEGORICAL
    ]
    numeric = [
        c for c in features if c not in CATEGORICAL
    ]

    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
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
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
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


def make_role_classifier(
    features: list[str],
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                make_preprocessor(features),
            ),
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


def make_regressor(
    features: list[str],
) -> Pipeline:
    return Pipeline(
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
    )


def regression_metrics(
    y_true,
    y_pred,
) -> tuple[float, float]:
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )
    rmse = math.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )
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


def regression_row(
    *,
    test: pd.DataFrame,
    predictions: np.ndarray,
    role_target: str,
    model_name: str,
    test_season: int,
    direct_top12: pd.Series,
) -> dict:
    mae, rmse = regression_metrics(
        test[TARGET],
        predictions,
    )
    return {
        "test_season": test_season,
        "role_target": role_target,
        "model": model_name,
        "rows": len(test),
        "mae_all_candidates": mae,
        "rmse_all_candidates": rmse,
        "mae_te1": subset_mae(
            test,
            predictions,
            test["listed_te1"].eq(1),
        ),
        "mae_wr2": subset_mae(
            test,
            predictions,
            test["depth_chart_te_rank"].eq(2),
        ),
        "mae_wr3": subset_mae(
            test,
            predictions,
            test["depth_chart_te_rank"].eq(3),
        ),
        "mae_played": subset_mae(
            test,
            predictions,
            test["played_any_snap"].eq(1),
        ),
        "mae_actual_role": subset_mae(
            test,
            predictions,
            test[role_target].eq(1),
        ),
        "mae_target3": subset_mae(
            test,
            predictions,
            test["target_3_role"].eq(1),
        ),
        "mae_10plus_actual": subset_mae(
            test,
            predictions,
            test[TARGET].ge(10),
        ),
        "mae_direct_top12": subset_mae(
            test,
            predictions,
            direct_top12,
        ),
    }


def evaluate_role_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    role_target: str,
    test_season: int,
    direct_points: np.ndarray,
    direct_top12: pd.Series,
) -> list[dict]:
    X_train = train[features]
    X_test = test[features]

    classifier = make_role_classifier(features)
    classifier.fit(
        X_train,
        train[role_target].astype(int),
    )

    probability = classifier.predict_proba(
        X_test
    )[:, 1]
    role_pred = (
        probability >= 0.50
    ).astype(int)

    role_rows = train[
        train[role_target].eq(1)
    ].copy()

    conditional = make_regressor(features)
    conditional.fit(
        role_rows[features],
        role_rows[TARGET],
    )

    conditional_points = np.maximum(
        conditional.predict(X_test),
        0.0,
    )

    soft_points = (
        probability * conditional_points
    )
    hard_points = np.where(
        probability >= 0.50,
        conditional_points,
        0.0,
    )

    rows = [
        regression_row(
            test=test,
            predictions=soft_points,
            role_target=role_target,
            model_name="Soft expected points",
            test_season=test_season,
            direct_top12=direct_top12,
        ),
        regression_row(
            test=test,
            predictions=hard_points,
            role_target=role_target,
            model_name="50% hard role gate",
            test_season=test_season,
            direct_top12=direct_top12,
        ),
        regression_row(
            test=test,
            predictions=direct_points,
            role_target=role_target,
            model_name="Direct candidate GB",
            test_season=test_season,
            direct_top12=direct_top12,
        ),
    ]

    classifier_row = {
        "test_season": test_season,
        "role_target": role_target,
        "model": "Role classifier",
        "rows": len(test),
        "role_accuracy": accuracy_score(
            test[role_target].astype(int),
            role_pred,
        ),
        "role_precision": precision_score(
            test[role_target].astype(int),
            role_pred,
            zero_division=0,
        ),
        "role_recall": recall_score(
            test[role_target].astype(int),
            role_pred,
            zero_division=0,
        ),
        "role_auc": roc_auc_score(
            test[role_target].astype(int),
            probability,
        ),
        "role_brier": brier_score_loss(
            test[role_target].astype(int),
            probability,
        ),
    }
    rows.append(classifier_row)

    print(
        f"  {role_target:<25} classifier | "
        f"AUC={classifier_row['role_auc']:.3f} "
        f"Brier={classifier_row['role_brier']:.3f} "
        f"P={classifier_row['role_precision']:.3f} "
        f"R={classifier_row['role_recall']:.3f}"
    )

    for row in rows[:3]:
        print(
            f"    {row['model']:<22} "
            f"ALL={row['mae_all_candidates']:.3f} "
            f"TE1={row['mae_te1']:.3f} "
            f"TE2={row['mae_wr2']:.3f} "
            f"TE3={row['mae_wr3']:.3f} "
            f"PLAYED={row['mae_played']:.3f} "
            f"ROLE={row['mae_actual_role']:.3f} "
            f"TGT3={row['mae_target3']:.3f} "
            f"TOP12={row['mae_direct_top12']:.3f}"
        )

    return rows


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing te_v2_candidate_dataset.csv. "
            "Run build_te_v2_candidate_dataset.py first."
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )

    historical = df[
        df["season"].between(2021, 2025)
        & df["game_completed"].eq(1)
        & df[TARGET].notna()
    ].copy()

    features = [
        column
        for column in historical.columns
        if column not in IDENTIFIERS
    ]

    print(
        "GRIDIRONIQ TE V2 WALK-FORWARD VALIDATION"
    )
    print("=" * 96)
    print("2026 excluded from model selection.")
    print(
        f"Historical candidates: {len(historical):,}"
    )
    print(
        f"Pregame features:       {len(features)}"
    )
    print(
        "Testing snap participation, target volume, target share, "
        "and combined role definitions."
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
            historical["season"].eq(
                test_season
            )
        ].copy()

        print(
            f"\nFold: train {train_start}-{train_end} "
            f"-> test {test_season}"
        )
        print(
            f"Train candidates: {len(train):,} | "
            f"Test candidates: {len(test):,}"
        )

        direct = make_regressor(features)
        direct.fit(
            train[features],
            train[TARGET],
        )
        direct_points = np.maximum(
            direct.predict(test[features]),
            0.0,
        )

        # Fixed weekly pregame top 12 chosen by the direct control,
        # not by each competing model's own predictions.
        ranking = test[["season", "week"]].copy()
        ranking["_direct_points"] = direct_points
        direct_top12 = (
            ranking.groupby(["season", "week"])[
                "_direct_points"
            ]
            .rank(method="first", ascending=False)
            .le(12)
        )

        for role_target in ROLE_TARGETS:
            all_rows.extend(
                evaluate_role_target(
                    train,
                    test,
                    features,
                    role_target,
                    test_season,
                    direct_points,
                    direct_top12,
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

    regression = results[
        results["model"].ne(
            "Role classifier"
        )
    ].copy()

    print("\n" + "=" * 96)
    print(
        "AVERAGE REGRESSION RESULTS, 2023-2025"
    )
    print("=" * 96)

    reg_summary = (
        regression.groupby(
            ["role_target", "model"],
            as_index=False,
        )
        .agg(
            all_mae=(
                "mae_all_candidates",
                "mean",
            ),
            te1_mae=("mae_te1", "mean"),
            wr2_mae=("mae_wr2", "mean"),
            wr3_mae=("mae_wr3", "mean"),
            played_mae=("mae_played", "mean"),
            role_mae=(
                "mae_actual_role",
                "mean",
            ),
            target3_mae=("mae_target3", "mean"),
            fp10_mae=("mae_10plus_actual", "mean"),
            top12_mae=("mae_direct_top12", "mean"),
        )
        .sort_values("all_mae")
    )

    print(
        reg_summary.to_string(
            index=False,
            formatters={
                "all_mae": "{:.3f}".format,
                "te1_mae": "{:.3f}".format,
                "wr2_mae": "{:.3f}".format,
                "wr3_mae": "{:.3f}".format,
                "played_mae": "{:.3f}".format,
                "role_mae": "{:.3f}".format,
                "target3_mae": "{:.3f}".format,
                "fp10_mae": "{:.3f}".format,
                "top12_mae": "{:.3f}".format,
            },
        )
    )

    classifiers = results[
        results["model"].eq(
            "Role classifier"
        )
    ].copy()

    print("\n" + "=" * 96)
    print(
        "AVERAGE ROLE-CLASSIFIER RESULTS, 2023-2025"
    )
    print("=" * 96)

    cls_summary = (
        classifiers.groupby(
            "role_target",
            as_index=False,
        )
        .agg(
            auc=("role_auc", "mean"),
            brier=("role_brier", "mean"),
            precision=("role_precision", "mean"),
            recall=("role_recall", "mean"),
            accuracy=("role_accuracy", "mean"),
        )
        .sort_values(
            "auc",
            ascending=False,
        )
    )

    print(
        cls_summary.to_string(
            index=False,
            formatters={
                "auc": "{:.3f}".format,
                "brier": "{:.3f}".format,
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "accuracy": "{:.3f}".format,
            },
        )
    )

    print(
        f"\nSaved results to: {OUTPUT_FILE}"
    )
    print(
        "\nDo not compare TE v2 all-candidate MAE directly with "
        "TE v1's 3.867 MAE. TE v1 evaluates active fantasy-relevant "
        "TE-games; TE v2 evaluates the full pregame depth-chart pool."
    )


if __name__ == "__main__":
    main()
