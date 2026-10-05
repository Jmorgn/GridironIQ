"""GridironIQ Kicker v2 walk-forward architecture experiments.

Purpose: determine whether Kicker v1's weak/unstable edge can be improved
before production. Predeclared experiments:
1) Gradient Boosting feature-group ablation:
   Player -> +Team -> +Opponent -> +Market -> +Environment -> +IDs.
2) On the full feature set, compare Ridge, Random Forest, Gradient
   Boosting, equal-weight ensemble, and a two-stage active-kicker model:
   P(any FG/PAT attempt) * E(fantasy points | active).

The fantasy target uses the user's finalized Kicker scoring rules.
2026 is excluded from selection. Historical 2021-24 depth charts do
not have independently verifiable publication timestamps; interpret
results with that limitation.

Run after:
    py src\features\build_k_v1_candidate_dataset.py
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
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "k_v1_candidate_dataset.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "k_v2_walk_forward_results.csv"

TARGET = "actual_fantasy_points"
ROLE_TARGET = "actual_active_kicker"
FOLDS = (
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
)

PLAYER_METRICS = [
    "fp", "fg_att", "fg_made", "pat_att", "pat_made",
    "fg_40plus_made", "fg_50plus_made",
    "long_misses", "kick_attempts", "active_kicker",
]
TEAM_METRICS = [
    "kicker_fp", "fg_att", "fg_made", "pat_att",
    "pat_made", "total_yards", "offensive_tds",
]
OPP_METRICS = [
    "fg_att_allowed", "pat_att_allowed", "yards_allowed",
]

PLAYER_FEATURES = [
    "depth_chart_k_rank", "listed_k1", "prior_chart_games",
    *[f"previous_{metric}" for metric in PLAYER_METRICS],
    *[
        f"avg_{metric}_last_{window}"
        for metric in PLAYER_METRICS
        for window in (3, 5)
    ],
]
TEAM_FEATURES = [
    *[f"team_prev_{metric}" for metric in TEAM_METRICS],
    *[
        f"team_avg_{metric}_last_{window}"
        for metric in TEAM_METRICS
        for window in (3, 5)
    ],
]
OPP_FEATURES = [
    *[f"opp_prev_{metric}" for metric in OPP_METRICS],
    *[
        f"opp_avg_{metric}_last_{window}"
        for metric in OPP_METRICS
        for window in (3, 5)
    ],
]
MARKET_FEATURES = [
    "team_rest", "opponent_rest", "rest_advantage",
    "neutral_site", "team_spread_line", "game_total_line",
    "team_implied_points", "opponent_implied_points",
]
ENVIRONMENT_FEATURES = [
    "home_away", "roof", "surface",
]
IDENTITY_FEATURES = ["team", "opponent"]

FEATURE_GROUPS = {
    "Player": PLAYER_FEATURES,
    "Player+Team": PLAYER_FEATURES + TEAM_FEATURES,
    "Player+Team+Opponent": (
        PLAYER_FEATURES + TEAM_FEATURES + OPP_FEATURES
    ),
    "Player+Team+Opponent+Market": (
        PLAYER_FEATURES + TEAM_FEATURES
        + OPP_FEATURES + MARKET_FEATURES
    ),
    "All no IDs": (
        PLAYER_FEATURES + TEAM_FEATURES + OPP_FEATURES
        + MARKET_FEATURES + ENVIRONMENT_FEATURES
    ),
    "All + IDs": (
        PLAYER_FEATURES + TEAM_FEATURES + OPP_FEATURES
        + MARKET_FEATURES + ENVIRONMENT_FEATURES
        + IDENTITY_FEATURES
    ),
}
FULL_FEATURES = FEATURE_GROUPS["All + IDs"]


def make_preprocessor(features: list[str]) -> ColumnTransformer:
    categorical = [
        name for name in features
        if name in {
            "team", "opponent", "home_away", "roof", "surface"
        }
    ]
    numeric = [
        name for name in features
        if name not in categorical
    ]
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True,
                    ),
                ),
            ]),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline(steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="UNKNOWN",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers=transformers)


def make_regressor(
    model: str,
    features: list[str],
) -> Pipeline:
    if model == "Ridge":
        estimator = Ridge(alpha=10.0)
    elif model == "Random Forest":
        estimator = RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=3,
            max_features=0.85,
            random_state=42,
            n_jobs=-1,
        )
    elif model == "Gradient Boosting":
        estimator = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown regressor: {model}")
    return Pipeline(steps=[
        ("preprocessor", make_preprocessor(features)),
        ("model", estimator),
    ])


def make_role_classifier(features: list[str]) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", make_preprocessor(features)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=4,
                max_features=0.75,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])


def fixed_top12_mask(test: pd.DataFrame) -> pd.Series:
    selection = pd.to_numeric(
        test["avg_fp_last_3"], errors="coerce"
    ).fillna(
        pd.to_numeric(
            test["team_avg_kicker_fp_last_3"],
            errors="coerce",
        )
    )
    fallback = pd.to_numeric(
        test["team_avg_kicker_fp_last_5"],
        errors="coerce",
    )
    selection = selection.fillna(fallback).fillna(0.0)
    ranking = test[["season", "week"]].copy()
    ranking["selection"] = selection
    return (
        ranking.groupby(["season", "week"])["selection"]
        .rank(method="first", ascending=False)
        .le(12)
    )


def subset_mae(
    test: pd.DataFrame,
    prediction: np.ndarray,
    mask: pd.Series,
) -> float:
    valid = mask.fillna(False).to_numpy(dtype=bool)
    if not valid.any():
        return float("nan")
    return float(mean_absolute_error(
        test.loc[valid, TARGET],
        np.asarray(prediction, dtype=float)[valid],
    ))


def score_row(
    *,
    test: pd.DataFrame,
    prediction: np.ndarray,
    fixed_top12: pd.Series,
    test_season: int,
    experiment: str,
    architecture: str,
    feature_group: str,
    role_probability: np.ndarray | None = None,
) -> dict:
    prediction = np.maximum(
        np.asarray(prediction, dtype=float), 0.0
    )
    actual = test[TARGET].to_numpy(dtype=float)
    row = {
        "test_season": test_season,
        "experiment": experiment,
        "architecture": architecture,
        "feature_group": feature_group,
        "test_rows": len(test),
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": math.sqrt(
            float(mean_squared_error(actual, prediction))
        ),
        "r2": float(r2_score(actual, prediction)),
        "k1_mae": subset_mae(
            test, prediction, test["listed_k1"].eq(1)
        ),
        "active_mae": subset_mae(
            test, prediction, test[ROLE_TARGET].eq(1)
        ),
        "fixed_top12_mae": subset_mae(
            test, prediction, fixed_top12
        ),
        "role_auc": np.nan,
        "role_brier": np.nan,
        "role_prevalence": float(test[ROLE_TARGET].mean()),
    }
    if role_probability is not None:
        role_actual = test[ROLE_TARGET].astype(int).to_numpy()
        role_probability = np.asarray(
            role_probability, dtype=float
        )
        row["role_brier"] = float(
            brier_score_loss(
                role_actual, role_probability
            )
        )
        if np.unique(role_actual).size == 2:
            row["role_auc"] = float(
                roc_auc_score(
                    role_actual, role_probability
                )
            )
    return row


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Rebuild K v1 candidates first."
        )
    data = pd.read_csv(
        DATA_FILE, low_memory=False,
        dtype={"player_id": str},
    )
    required = [
        *FULL_FEATURES,
        "season", "week", "game_completed",
        TARGET, ROLE_TARGET, "listed_k1",
        "avg_fp_last_3",
        "team_avg_kicker_fp_last_3",
        "team_avg_kicker_fp_last_5",
    ]
    missing = [name for name in required if name not in data]
    if missing:
        raise RuntimeError(
            "K v2 experiment requires rebuilt dataset with: "
            + ", ".join(missing)
        )
    historical = data[
        data["season"].between(2021, 2025)
        & data["game_completed"].eq(1)
        & data[TARGET].notna()
        & data[ROLE_TARGET].notna()
    ].copy()

    print("GRIDIRONIQ K V2 WALK-FORWARD EXPERIMENTS")
    print("=" * 100)
    print(
        "Goal: require a stable, meaningful improvement before "
        "promoting a Kicker model."
    )
    print("2026 excluded from model selection.")
    print(f"Historical candidates: {len(historical):,}")
    print(
        f"Active-kicker prevalence: "
        f"{historical[ROLE_TARGET].mean():.1%}"
    )
    print(
        "2021-24 depth-chart publication timestamps remain "
        "historically unverifiable."
    )

    rows = []
    for train_start, train_end, test_season in FOLDS:
        train = historical[
            historical["season"].between(
                train_start, train_end
            )
        ].copy()
        test = historical[
            historical["season"].eq(test_season)
        ].copy()
        fixed_top12 = fixed_top12_mask(test)
        print(
            f"\nTrain {train_start}-{train_end} -> "
            f"test {test_season} "
            f"({len(train):,}/{len(test):,})"
        )

        # Feature ablation: keep architecture constant.
        for group_name, features in FEATURE_GROUPS.items():
            model = make_regressor(
                "Gradient Boosting", features
            )
            model.fit(train[features], train[TARGET])
            pred = model.predict(test[features])
            row = score_row(
                test=test,
                prediction=pred,
                fixed_top12=fixed_top12,
                test_season=test_season,
                experiment="feature_ablation",
                architecture="Gradient Boosting",
                feature_group=group_name,
            )
            rows.append(row)
            print(
                f"  GB {group_name:<30} "
                f"MAE={row['mae']:.3f} "
                f"K1={row['k1_mae']:.3f} "
                f"Active={row['active_mae']:.3f} "
                f"Top12={row['fixed_top12_mae']:.3f}"
            )

        # Architecture comparison uses one predeclared feature set.
        direct_predictions = {}
        for name in [
            "Ridge", "Random Forest", "Gradient Boosting"
        ]:
            model = make_regressor(name, FULL_FEATURES)
            model.fit(
                train[FULL_FEATURES], train[TARGET]
            )
            pred = np.maximum(
                model.predict(test[FULL_FEATURES]), 0.0
            )
            direct_predictions[name] = pred
            row = score_row(
                test=test,
                prediction=pred,
                fixed_top12=fixed_top12,
                test_season=test_season,
                experiment="architecture",
                architecture=name,
                feature_group="All + IDs",
            )
            rows.append(row)

        ensemble = np.mean(
            np.column_stack([
                direct_predictions["Ridge"],
                direct_predictions["Random Forest"],
                direct_predictions["Gradient Boosting"],
            ]),
            axis=1,
        )
        rows.append(score_row(
            test=test,
            prediction=ensemble,
            fixed_top12=fixed_top12,
            test_season=test_season,
            experiment="architecture",
            architecture="Equal ensemble",
            feature_group="All + IDs",
        ))

        role_model = make_role_classifier(FULL_FEATURES)
        role_model.fit(
            train[FULL_FEATURES],
            train[ROLE_TARGET].astype(int),
        )
        role_probability = role_model.predict_proba(
            test[FULL_FEATURES]
        )[:, 1]
        active_train = train[
            train[ROLE_TARGET].eq(1)
        ].copy()
        conditional = make_regressor(
            "Gradient Boosting", FULL_FEATURES
        )
        conditional.fit(
            active_train[FULL_FEATURES],
            active_train[TARGET],
        )
        conditional_points = np.maximum(
            conditional.predict(test[FULL_FEATURES]),
            0.0,
        )
        soft = role_probability * conditional_points
        gated_row = score_row(
            test=test,
            prediction=soft,
            fixed_top12=fixed_top12,
            test_season=test_season,
            experiment="architecture",
            architecture="Active-prob x conditional GB",
            feature_group="All + IDs",
            role_probability=role_probability,
        )
        rows.append(gated_row)
        print(
            "  Architecture full-feature results: "
            + " | ".join(
                f"{name}={score_row(test=test, prediction=pred, fixed_top12=fixed_top12, test_season=test_season, experiment='print', architecture=name, feature_group='All + IDs')['mae']:.3f}"
                for name, pred in [
                    ("Ridge", direct_predictions["Ridge"]),
                    ("RF", direct_predictions["Random Forest"]),
                    ("GB", direct_predictions["Gradient Boosting"]),
                    ("Ensemble", ensemble),
                    ("ActiveGate", soft),
                ]
            )
        )
        print(
            f"  ActiveGate role AUC={gated_row['role_auc']:.3f} "
            f"Brier={gated_row['role_brier']:.3f}"
        )

    results = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 100)
    print("FEATURE ABLATION — 2023-2025 AVERAGE")
    print("=" * 100)
    feature_summary = (
        results[
            results["experiment"].eq("feature_ablation")
        ]
        .groupby(
            ["architecture", "feature_group"],
            as_index=False,
        )
        .agg(
            mae=("mae", "mean"),
            k1_mae=("k1_mae", "mean"),
            active_mae=("active_mae", "mean"),
            top12_mae=("fixed_top12_mae", "mean"),
        )
        .sort_values("mae")
    )
    print(
        feature_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )

    print("\n" + "=" * 100)
    print("ARCHITECTURES — 2023-2025 AVERAGE")
    print("=" * 100)
    architecture_summary = (
        results[
            results["experiment"].eq("architecture")
        ]
        .groupby(
            ["architecture", "feature_group"],
            as_index=False,
        )
        .agg(
            mae=("mae", "mean"),
            k1_mae=("k1_mae", "mean"),
            active_mae=("active_mae", "mean"),
            top12_mae=("fixed_top12_mae", "mean"),
            role_auc=("role_auc", "mean"),
            role_brier=("role_brier", "mean"),
        )
        .sort_values("mae")
    )
    print(
        architecture_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )
    print(f"\nSaved results: {OUTPUT_FILE}")
    print(
        "Do not promote Kicker based on a tiny average edge. "
        "Prefer an architecture that improves meaningfully and "
        "does not depend on a single test season."
    )


if __name__ == "__main__":
    main()
