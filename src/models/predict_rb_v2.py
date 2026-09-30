"""Generate current-week GridironIQ RB v2 rankings from saved models.

RB v2 does not force one back per team. Every pregame depth-chart candidate is
scored, and the official projection is:
    P(35%+ offensive snaps) * conditional fantasy points
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "rb_v2_candidate_dataset.csv"
BUNDLE_FILE = ROOT / "models" / "rb_v2_bundle.joblib"

ALL_FUTURE_FILE = (
    ROOT / "data" / "processed" / "rb_v2_all_future_candidates.csv"
)
RANKINGS_FILE = (
    ROOT / "data" / "processed" / "rb_v2_weekly_rankings.csv"
)


def safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_context(row: pd.Series) -> tuple[str, str]:
    positives: list[str] = []
    negatives: list[str] = []

    recent_fp = safe_float(row.get("avg_fp_last_3"))
    recent_opp = safe_float(row.get("avg_opportunities_last_3"))
    recent_snap = safe_float(row.get("avg_offense_pct_last_3"))
    rush_allowed = safe_float(
        row.get("opp_avg_rush_yards_allowed_last_3")
    )
    game_total = safe_float(row.get("game_total_line"))
    spread = safe_float(row.get("team_spread_line"))
    competition = safe_float(
        row.get("rb_room_max_other_prior_opportunities")
    )
    injury_score = safe_float(row.get("injury_status_score"))

    if recent_opp is not None and recent_opp >= 15:
        positives.append(
            f"Heavy recent workload ({recent_opp:.1f} opp/game)"
        )
    elif recent_opp is not None and recent_opp < 7:
        negatives.append(
            f"Light recent workload ({recent_opp:.1f} opp/game)"
        )

    if recent_snap is not None and recent_snap >= 0.60:
        positives.append(
            f"Strong snap role ({recent_snap:.0%} avg last 3)"
        )
    elif recent_snap is not None and recent_snap < 0.30:
        negatives.append(
            f"Limited snap role ({recent_snap:.0%} avg last 3)"
        )

    if recent_fp is not None and recent_fp >= 15:
        positives.append(
            f"Strong recent scoring ({recent_fp:.1f} FP last 3)"
        )

    if rush_allowed is not None and rush_allowed >= 125:
        positives.append(
            f"Opponent allowing {rush_allowed:.0f} rush yds/game recently"
        )
    elif rush_allowed is not None and rush_allowed <= 90:
        negatives.append(
            f"Opponent allowing only {rush_allowed:.0f} rush yds/game recently"
        )

    if game_total is not None and game_total >= 47:
        positives.append(
            f"High game total ({game_total:.1f})"
        )

    if spread is not None and spread >= 4:
        positives.append(
            f"Team favored by {spread:.1f}"
        )
    elif spread is not None and spread <= -4:
        negatives.append(
            f"Team underdog by {abs(spread):.1f}"
        )

    if competition is not None and competition >= 12:
        negatives.append(
            f"Strong RB-room competition ({competition:.1f} "
            "other prior opp/game)"
        )

    if injury_score is not None and injury_score >= 2:
        negatives.append("Meaningful injury/practice concern")

    return (
        "; ".join(positives[:3]) or "No major positive signal",
        "; ".join(negatives[:3]) or "No major negative signal",
    )


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing rb_v2_candidate_dataset.csv. "
            "Run build_rb_v2_candidate_dataset.py first."
        )

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Missing models/rb_v2_bundle.joblib. "
            "Run train_rb_v2.py once before weekly prediction."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    bundle = joblib.load(BUNDLE_FILE)

    features = bundle["features"]
    role_model = bundle["role_classifier"]
    points_model = bundle["conditional_regressor"]

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(
            "Current RB candidate dataset is missing model features: "
            + ", ".join(missing)
        )

    future = df[df["game_completed"].eq(0)].copy()

    if future.empty:
        print("No future/unplayed RB candidate rows found.")
        return

    X = future[features]
    probability = role_model.predict_proba(X)[:, 1]
    conditional_points = np.maximum(
        points_model.predict(X),
        0.0,
    )
    projection = probability * conditional_points

    keep = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "depth_chart_rb_rank",
        "listed_rb1",
        "team_spread_line",
        "game_total_line",
        "avg_fp_last_3",
        "avg_opportunities_last_3",
        "avg_offense_pct_last_3",
        "avg_carry_share_last_3",
        "avg_target_share_last_3",
        "rb_room_max_other_prior_opportunities",
        "opp_avg_rush_yards_allowed_last_3",
        "opp_avg_rush_tds_allowed_last_3",
        "on_injury_report",
        "injury_status_score",
    ]
    keep = [c for c in keep if c in future.columns]

    output = future[keep].copy()
    output["rb_role_probability"] = probability
    output["conditional_fantasy_points"] = conditional_points
    output["gridironiq_projection"] = projection

    contexts = [
        build_context(future.iloc[index])
        for index in range(len(future))
    ]
    output["key_positives"] = [x[0] for x in contexts]
    output["key_negatives"] = [x[1] for x in contexts]

    output = output.sort_values(
        [
            "season",
            "week",
            "gridironiq_projection",
        ],
        ascending=[True, True, False],
    )

    ALL_FUTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(ALL_FUTURE_FILE, index=False)

    upcoming_week = int(
        pd.to_numeric(
            output["week"],
            errors="coerce",
        ).dropna().min()
    )
    upcoming_season = int(
        pd.to_numeric(
            output.loc[
                output["week"].eq(upcoming_week),
                "season",
            ],
            errors="coerce",
        ).dropna().min()
    )

    rankings = output[
        output["season"].eq(upcoming_season)
        & output["week"].eq(upcoming_week)
    ].copy()

    rankings = rankings.sort_values(
        "gridironiq_projection",
        ascending=False,
    ).reset_index(drop=True)
    rankings.insert(0, "rank", rankings.index + 1)
    rankings.to_csv(RANKINGS_FILE, index=False)

    print("GRIDIRONIQ WEEKLY RB RANKINGS")
    print("=" * 92)
    print(
        f"Season {upcoming_season} | Week {upcoming_week} | "
        f"{len(rankings)} pregame RB candidates"
    )
    print(
        "Projection = P(35%+ offensive snaps) x "
        "fantasy points conditional on that role.\n"
    )

    display = rankings[
        [
            "rank",
            "player_name",
            "team",
            "opponent",
            "depth_chart_rb_rank",
            "rb_role_probability",
            "conditional_fantasy_points",
            "gridironiq_projection",
        ]
    ].head(40)

    print(
        display.to_string(
            index=False,
            formatters={
                "rb_role_probability": "{:.1%}".format,
                "conditional_fantasy_points": "{:.2f}".format,
                "gridironiq_projection": "{:.2f}".format,
            },
        )
    )

    print("\nWHY GRIDIRONIQ LIKES / DISLIKES THE TOP 10")
    print("=" * 92)
    for _, row in rankings.head(10).iterrows():
        print(
            f"\n#{int(row['rank'])} {row['player_name']} "
            f"({row['team']} vs {row['opponent']}) — "
            f"{row['gridironiq_projection']:.2f} FP | "
            f"role {row['rb_role_probability']:.1%}"
        )
        print(f"  + {row['key_positives']}")
        print(f"  - {row['key_negatives']}")

    print(f"\nSaved weekly rankings to: {RANKINGS_FILE}")
    print(f"Saved all future candidates to: {ALL_FUTURE_FILE}")


if __name__ == "__main__":
    main()
