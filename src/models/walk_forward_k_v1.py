"""Leakage-aware GridironIQ Kicker v1 walk-forward benchmark.

Only screenshot-confirmed scoring components are modeled: made FG by
distance, short-FG misses, and made PATs. Missed/blocked PAT penalties
are not yet confirmed. Scores are PROVISIONAL research targets and
must not be called official Yahoo fantasy projections.

The fixed pregame top-12 cohort uses a simple player-last-3/team-last-3
ranking, NOT the model being evaluated. Train through the preceding
seasons and test 2023, 2024 and 2025. Never use 2026 to pick a model.
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    ROOT / "data" / "processed" / "k_v1_candidate_dataset.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed" / "k_v1_walk_forward_results.csv"
)

TARGET = "actual_confirmed_component_points"
FOLDS = (
    (2021, 2022, 2023),
    (2021, 2023, 2024),
    (2021, 2024, 2025),
)
CATEGORICAL = [
    "team", "opponent", "home_away", "roof", "surface",
]
PREGAME_CONTEXT = [
    "depth_chart_k_rank", "listed_k1",
    "team_rest", "opponent_rest",
    "rest_advantage", "neutral_site",
    # Historical temperature/wind in nflverse games.csv can
    # reflect the played-game weather, not a verified forecast;
    # exclude them until prekickoff values are available.
    "team_spread_line", "game_total_line",
    "prior_chart_games",
]
PLAYER_METRICS = [
    "fp", "fg_att", "fg_made", "pat_att",
    "pat_made", "fg_40plus_made", "fg_50plus_made",
    "long_misses", "kick_attempts", "active_kicker",
]
TEAM_METRICS = [
    "kicker_fp", "fg_att", "fg_made",
    "pat_att", "pat_made", "total_yards",
    "offensive_tds",
]
OPP_METRICS = [
    "fg_att_allowed", "pat_att_allowed", "yards_allowed",
]
FEATURES = [
    *CATEGORICAL,
    *PREGAME_CONTEXT,
    *[f"previous_{metric}" for metric in PLAYER_METRICS],
    *[
        f"avg_{metric}_last_{n}"
        for metric in PLAYER_METRICS
        for n in (3, 5)
    ],
    *[f"team_prev_{metric}" for metric in TEAM_METRICS],
    *[
        f"team_avg_{metric}_last_{n}"
        for metric in TEAM_METRICS
        for n in (3, 5)
    ],
    *[f"opp_prev_{metric}" for metric in OPP_METRICS],
    *[
        f"opp_avg_{metric}_last_{n}"
        for metric in OPP_METRICS
        for n in (3, 5)
    ],
]


def preprocessor() -> ColumnTransformer:
    numeric = [
        name for name in FEATURES
        if name not in CATEGORICAL
    ]
    return ColumnTransformer(transformers=[
        (
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
        ),
        (
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
            CATEGORICAL,
        ),
    ])


def models() -> dict[str, Pipeline]:
    return {
        "Ridge Regression": Pipeline(steps=[
            ("preprocessor", preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "Random Forest": Pipeline(steps=[
            ("preprocessor", preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=3,
                max_features=0.85,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Gradient Boosting": Pipeline(steps=[
            ("preprocessor", preprocessor()),
            ("model", GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=5,
                random_state=42,
            )),
        ]),
    }


def subset_mae(
    test: pd.DataFrame,
    pred: np.ndarray,
    mask: pd.Series,
) -> float:
    valid = mask.fillna(False).to_numpy(dtype=bool)
    if not valid.any():
        return float("nan")
    return float(mean_absolute_error(
        test.loc[valid, TARGET],
        np.asarray(pred, dtype=float)[valid],
    ))


def metrics(
    test: pd.DataFrame,
    pred: np.ndarray,
    fixed_top12: pd.Series,
    model: str,
    test_season: int,
) -> dict:
    actual = test[TARGET].to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "test_season": test_season,
        "model": model,
        "test_rows": len(test),
        "mae": float(mean_absolute_error(actual, pred)),
        "rmse": math.sqrt(
            float(mean_squared_error(actual, pred))
        ),
        "r2": float(r2_score(actual, pred)),
        "listed_k1_mae": subset_mae(
            test, pred, test["listed_k1"].eq(1)
        ),
        "active_kicker_mae": subset_mae(
            test, pred, test["actual_active_kicker"].eq(1)
        ),
        "fixed_top12_mae": subset_mae(
            test, pred, fixed_top12
        ),
        "unconfirmed_pat_event_rows": int(
            test["has_unconfirmed_pat_event"].eq(1).sum()
        ),
    }


def baseline_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, np.ndarray]:
    mean = train[TARGET].mean()
    personal = pd.to_numeric(
        test["avg_fp_last_3"], errors="coerce"
    )
    team = pd.to_numeric(
        test["team_avg_kicker_fp_last_3"],
        errors="coerce",
    )
    return {
        "Historical mean": np.full(len(test), mean),
        "Player last-3": (
            personal.fillna(team)
            .fillna(mean).to_numpy(dtype=float)
        ),
        "Team K last-3": (
            team.fillna(personal)
            .fillna(mean).to_numpy(dtype=float)
        ),
    }


def fixed_top12_mask(
    test: pd.DataFrame,
    player_baseline: np.ndarray,
) -> pd.Series:
    ranking = test[["season", "week"]].copy()
    ranking["selection_projection"] = player_baseline
    return (
        ranking.groupby(["season", "week"])[
            "selection_projection"
        ]
        .rank(method="first", ascending=False)
        .le(12)
    )


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run "
            "py src\\features\\build_k_v1_candidate_dataset.py first."
        )
    data = pd.read_csv(
        DATA_FILE, low_memory=False,
        dtype={"player_id": str},
    )
    missing = [
        c for c in [
            *FEATURES, "season", "week",
            "game_completed", TARGET,
            "listed_k1", "actual_active_kicker",
            "has_unconfirmed_pat_event",
        ] if c not in data.columns
    ]
    if missing:
        raise RuntimeError(
            f"K v1 dataset missing required columns: {missing}"
        )
    history = data[
        data["season"].between(2021, 2025)
        & data["game_completed"].eq(1)
        & data[TARGET].notna()
    ].copy()
    if history.empty:
        raise RuntimeError("No completed K v1 candidates.")

    # Strict feature whitelist: no stat totals or current-game
    # labels may enter the predictors by accident.
    forbidden = [
        field for field in FEATURES
        if field.startswith("actual_")
        or "confirmed_component_points" in field
        or field in (
            "has_unconfirmed_pat_event",
            "had_stat_row",
            "game_completed",
            "chart_pregame_verified",
            "kickoff_utc",
            "game_temp", "game_wind",
        )
    ]
    if forbidden:
        raise RuntimeError(
            f"Forbidden feature(s): {forbidden}"
        )
    print("GRIDIRONIQ K V1 WALK-FORWARD RESEARCH")
    print("=" * 88)
    print(
        "Target: screenshot-confirmed K scoring components "
        "(NOT finalized Yahoo fantasy points)."
    )
    print(
        "Missed/blocked PAT penalty unconfirmed; report "
        "affected test-row count in every fold."
    )
    print("2026 excluded from model selection.")
    print(f"Historical pregame candidates: {len(history):,}")
    print(f"Pregame feature columns: {len(FEATURES)}")
    print(
        "Untimestamped weekly depth charts 2021-24: "
        "weekly historical candidate coverage is not "
        "independent proof of publication before kickoff."
    )

    rows = []
    for train_start, train_end, test_season in FOLDS:
        train = history[
            history["season"].between(
                train_start, train_end
            )
        ].copy()
        test = history[
            history["season"].eq(test_season)
        ].copy()
        if train.empty or test.empty:
            raise RuntimeError(
                f"No train/test candidates for {test_season}."
            )
        baselines = baseline_predictions(
            train, test
        )
        fixed_top12 = fixed_top12_mask(
            test, baselines["Player last-3"]
        )
        print(
            f"\nTrain {train_start}-{train_end} → "
            f"test {test_season} "
            f"({len(train):,} / {len(test):,} rows)"
        )

        for name, pred in baselines.items():
            row = metrics(
                test, pred, fixed_top12,
                name, test_season,
            )
            rows.append(row)
            print(
                f"  {name:<22} MAE {row['mae']:.3f} | "
                f"K1 {row['listed_k1_mae']:.3f} | "
                f"Active {row['active_kicker_mae']:.3f} | "
                f"fixed top12 {row['fixed_top12_mae']:.3f}"
            )

        for name, model in models().items():
            model.fit(
                train[FEATURES], train[TARGET]
            )
            predicted = model.predict(test[FEATURES])
            row = metrics(
                test, predicted, fixed_top12,
                name, test_season,
            )
            rows.append(row)
            print(
                f"  {name:<22} MAE {row['mae']:.3f} | "
                f"K1 {row['listed_k1_mae']:.3f} | "
                f"Active {row['active_kicker_mae']:.3f} | "
                f"fixed top12 {row['fixed_top12_mae']:.3f}"
            )

    results = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    results.to_csv(
        OUTPUT_FILE, index=False
    )
    summary = (
        results.groupby("model", as_index=False)
        .agg(
            mae=("mae", "mean"),
            k1_mae=("listed_k1_mae", "mean"),
            active_mae=("active_kicker_mae", "mean"),
            top12_mae=("fixed_top12_mae", "mean"),
            rmse=("rmse", "mean"),
        )
        .sort_values("mae")
    )
    print("\n" + "=" * 88)
    print("2023-2025 WALK-FORWARD AVERAGE — RESEARCH ONLY")
    print("=" * 88)
    print(
        summary.to_string(
            index=False,
            float_format=lambda n: f"{n:.3f}",
        )
    )
    print(
        f"\nRows with missed or blocked PAT outcomes: "
        f"{history['has_unconfirmed_pat_event'].eq(1).sum():,}"
    )
    print(f"Saved research results: {OUTPUT_FILE}")
    print(
        "Do NOT deploy from these results until exact PAT "
        "scoring and label coverage are verified."
    )


if __name__ == "__main__":
    main()
