"""Generate current-week GridironIQ TE v2 rankings from saved models.

Every pregame TE candidate is scored. Official fantasy projections:
P(50%+ offensive snaps) * conditional fantasy points.
The 50% snap role includes blocking and is not a route participation metric.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uncertainty import apply_residual_intervals

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = (
    ROOT
    / "data"
    / "processed"
    / "te_v2_candidate_dataset.csv"
)
BUNDLE_FILE = (
    ROOT / "models" / "te_v2_bundle.joblib"
)

ALL_FUTURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "te_v2_all_future_candidates.csv"
)
RANKINGS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "te_v2_weekly_rankings.csv"
)


def safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_context(
    row: pd.Series,
) -> tuple[str, str]:
    positives: list[str] = []
    negatives: list[str] = []

    recent_fp = safe_float(
        row.get("avg_fp_last_3")
    )
    recent_targets = safe_float(
        row.get("avg_targets_last_3")
    )
    recent_target_share = safe_float(
        row.get("avg_target_share_last_3")
    )
    recent_snap = safe_float(
        row.get("avg_offense_pct_last_3")
    )
    rec_yards_allowed = safe_float(
        row.get(
            "opp_avg_te_rec_yards_allowed_last_3"
        )
    )
    game_total = safe_float(
        row.get("game_total_line")
    )
    spread = safe_float(
        row.get("team_spread_line")
    )
    competition = safe_float(
        row.get("te_room_max_other_prior_targets")
    )
    injury_score = safe_float(
        row.get("injury_status_score")
    )

    if (
        recent_targets is not None
        and recent_targets >= 5
    ):
        positives.append(
            "Heavy recent target volume "
            f"({recent_targets:.1f}/game)"
        )
    elif (
        recent_targets is not None
        and recent_targets < 2
    ):
        negatives.append(
            "Light recent target volume "
            f"({recent_targets:.1f}/game)"
        )

    if (
        recent_target_share is not None
        and recent_target_share >= 0.15
    ):
        positives.append(
            "Strong target share "
            f"({recent_target_share:.0%} avg last 3)"
        )
    elif (
        recent_target_share is not None
        and recent_target_share < 0.06
    ):
        negatives.append(
            "Limited target share "
            f"({recent_target_share:.0%} avg last 3)"
        )

    if (
        recent_snap is not None
        and recent_snap >= 0.65
    ):
        positives.append(
            "Strong snap role "
            f"({recent_snap:.0%} avg last 3)"
        )
    elif (
        recent_snap is not None
        and recent_snap < 0.30
    ):
        negatives.append(
            "Limited snap role "
            f"({recent_snap:.0%} avg last 3)"
        )

    if (
        recent_fp is not None
        and recent_fp >= 10
    ):
        positives.append(
            "Strong recent scoring "
            f"({recent_fp:.1f} FP last 3)"
        )

    if (
        rec_yards_allowed is not None
        and rec_yards_allowed >= 70
    ):
        positives.append(
            "Opponent allowing "
            f"{rec_yards_allowed:.0f} receiving "
            "TE receiving yds/game recently"
        )
    elif (
        rec_yards_allowed is not None
        and rec_yards_allowed <= 30
    ):
        negatives.append(
            "Opponent allowing only "
            f"{rec_yards_allowed:.0f} receiving "
            "TE receiving yds/game recently"
        )

    if (
        game_total is not None
        and game_total >= 46
    ):
        positives.append(
            f"High game total ({game_total:.1f})"
        )

    # Negative point spreads conventionally indicate the favorite.
    if (
        spread is not None
        and spread >= 4
    ):
        positives.append(
            "Underdog game script may increase "
            "passing volume"
        )
    elif (
        spread is not None
        and spread <= -7
    ):
        negatives.append(
            "Heavy favorite could reduce "
            "late passing volume"
        )

    if (
        competition is not None
        and competition >= 4
    ):
        negatives.append(
            "Strong TE-room target competition "
            f"({competition:.1f} other prior "
            "targets/game)"
        )

    if (
        injury_score is not None
        and injury_score >= 2
    ):
        negatives.append(
            "Meaningful injury/practice concern"
        )

    return (
        "; ".join(positives[:3])
        or "No major positive signal",
        "; ".join(negatives[:3])
        or "No major negative signal",
    )


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing te_v2_candidate_dataset.csv. "
            "Run build_te_v2_candidate_dataset.py first."
        )

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Missing models/te_v2_bundle.joblib. "
            "Run train_te_v2.py once before weekly prediction."
        )

    df = pd.read_csv(
        DATA_FILE,
        low_memory=False,
    )
    bundle = joblib.load(BUNDLE_FILE)

    features = bundle["features"]
    role_model = bundle["role_classifier"]
    points_model = bundle[
        "conditional_regressor"
    ]
    uncertainty = bundle.get("uncertainty")

    if uncertainty is None:
        raise RuntimeError(
            "TE model bundle has no uncertainty calibration. "
            "Run 'py run_weekly.py --retrain' once after "
            "pulling the TE production update."
        )

    missing = [
        c for c in features
        if c not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "Current TE candidate dataset is missing "
            "model features: "
            + ", ".join(missing)
        )

    future = df[
        df["game_completed"].eq(0)
    ].copy()

    if future.empty:
        print(
            "No future/unplayed TE candidate rows found."
        )
        return

    X = future[features]
    probability = role_model.predict_proba(
        X
    )[:, 1]
    conditional_points = np.maximum(
        points_model.predict(X),
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

    keep = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "depth_chart_te_rank",
        "listed_te1",
        "team_spread_line",
        "game_total_line",
        "avg_fp_last_3",
        "avg_targets_last_3",
        "avg_receptions_last_3",
        "avg_rec_yards_last_3",
        "avg_target_share_last_3",
        "avg_offense_pct_last_3",
        "avg_air_yards_last_3",
        "te_room_max_other_prior_targets",
        "te_room_max_other_prior_target_share",
        "opp_avg_te_rec_yards_allowed_last_3",
        "opp_avg_te_rec_tds_allowed_last_3",
        "opp_avg_te_fp_allowed_last_3",
        "opp_avg_te_targets_allowed_last_3",
        "on_injury_report",
        "injury_status_score",
    ]
    keep = [
        c for c in keep
        if c in future.columns
    ]

    output = future[keep].copy()
    output[
        "te_role_probability"
    ] = probability
    output[
        "conditional_fantasy_points"
    ] = conditional_points
    output[
        "gridironiq_projection"
    ] = projection
    output[
        "prediction_low_80"
    ] = low
    output[
        "prediction_high_80"
    ] = high

    contexts = [
        build_context(
            future.iloc[index]
        )
        for index in range(len(future))
    ]
    output["key_positives"] = [
        x[0] for x in contexts
    ]
    output["key_negatives"] = [
        x[1] for x in contexts
    ]

    output = output.sort_values(
        [
            "season",
            "week",
            "gridironiq_projection",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    ALL_FUTURE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output.to_csv(
        ALL_FUTURE_FILE,
        index=False,
    )

    upcoming_week = int(
        pd.to_numeric(
            output["week"],
            errors="coerce",
        )
        .dropna()
        .min()
    )
    upcoming_season = int(
        pd.to_numeric(
            output.loc[
                output["week"].eq(
                    upcoming_week
                ),
                "season",
            ],
            errors="coerce",
        )
        .dropna()
        .min()
    )

    rankings = output[
        output["season"].eq(
            upcoming_season
        )
        & output["week"].eq(
            upcoming_week
        )
    ].copy()

    rankings = rankings.sort_values(
        "gridironiq_projection",
        ascending=False,
    ).reset_index(drop=True)
    rankings.insert(
        0,
        "rank",
        rankings.index + 1,
    )
    rankings.to_csv(
        RANKINGS_FILE,
        index=False,
    )

    print("GRIDIRONIQ WEEKLY TE RANKINGS")
    print("=" * 96)
    print(
        f"Season {upcoming_season} | "
        f"Week {upcoming_week} | "
        f"{len(rankings)} pregame TE candidates"
    )
    print(
        "Projection = P(50%+ offensive snaps) x "
        "fantasy points conditional on that role.\n"
    )

    display = rankings[
        [
            "rank",
            "player_name",
            "team",
            "opponent",
            "depth_chart_te_rank",
            "te_role_probability",
            "conditional_fantasy_points",
            "gridironiq_projection",
            "prediction_low_80",
            "prediction_high_80",
        ]
    ].head(60)

    print(
        display.to_string(
            index=False,
            formatters={
                "te_role_probability": "{:.1%}".format,
                "conditional_fantasy_points": "{:.2f}".format,
                "gridironiq_projection": "{:.2f}".format,
                "prediction_low_80": "{:.2f}".format,
                "prediction_high_80": "{:.2f}".format,
            },
        )
    )

    print(
        "\nWHY GRIDIRONIQ LIKES / "
        "DISLIKES THE TOP 10"
    )
    print("=" * 96)

    for _, row in (
        rankings.head(10).iterrows()
    ):
        print(
            f"\n#{int(row['rank'])} "
            f"{row['player_name']} "
            f"({row['team']} vs "
            f"{row['opponent']}) — "
            f"{row['gridironiq_projection']:.2f} FP "
            f"| role "
            f"{row['te_role_probability']:.1%}"
        )
        print(
            "  80% historical range: "
            f"{row['prediction_low_80']:.2f} to "
            f"{row['prediction_high_80']:.2f} FP"
        )
        print(
            f"  + {row['key_positives']}"
        )
        print(
            f"  - {row['key_negatives']}"
        )

    print(
        f"\nSaved weekly rankings to: "
        f"{RANKINGS_FILE}"
    )
    print(
        f"Saved all future candidates to: "
        f"{ALL_FUTURE_FILE}"
    )


if __name__ == "__main__":
    main()
