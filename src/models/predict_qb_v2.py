"""Generate current-week GridironIQ QB v2 rankings from saved models.

This script does NOT retrain the model. It loads models/qb_v2_bundle.joblib,
scores all future QB candidates, applies the production top-role-per-team gate,
and writes a clean ranking for the earliest upcoming NFL week.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_v2_candidate_dataset.csv"
BUNDLE_FILE = ROOT / "models" / "qb_v2_bundle.joblib"

ALL_FUTURE_FILE = (
    ROOT / "data" / "processed" / "qb_v2_all_future_candidates.csv"
)
RANKINGS_FILE = (
    ROOT / "data" / "processed" / "qb_v2_weekly_rankings.csv"
)


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


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing qb_v2_candidate_dataset.csv. "
            "Run build_qb_v2_candidate_dataset.py first."
        )

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Missing models/qb_v2_bundle.joblib. "
            "Run train_qb_v2.py once before weekly prediction."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    bundle = joblib.load(BUNDLE_FILE)

    features = bundle["features"]
    role_model = bundle["role_classifier"]
    points_model = bundle["conditional_regressor"]

    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise RuntimeError(
            "Current candidate dataset is missing model features: "
            + ", ".join(missing_features)
        )

    future = df[df["game_completed"].eq(0)].copy()

    if future.empty:
        print("No future/unplayed QB candidate rows found.")
        return

    X = future[features]

    role_probability = role_model.predict_proba(X)[:, 1]
    conditional_points = np.maximum(
        points_model.predict(X),
        0.0,
    )
    selected = select_top_role_per_team(
        future,
        role_probability,
    )

    output = future[
        [
            "player_id",
            "player_name",
            "season",
            "week",
            "team",
            "opponent",
            "home_away",
            "depth_chart_qb_rank",
            "listed_qb1",
            "team_spread_line",
            "game_total_line",
        ]
    ].copy()

    output["meaningful_role_probability"] = role_probability
    output["selected_team_qb"] = selected
    output["conditional_fantasy_points"] = conditional_points
    output["soft_expected_fantasy_points"] = (
        role_probability * conditional_points
    )
    output["gridironiq_projection"] = np.where(
        selected == 1,
        conditional_points,
        0.0,
    )

    output = output.sort_values(
        ["season", "week", "team", "meaningful_role_probability"],
        ascending=[True, True, True, False],
    )

    ALL_FUTURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(ALL_FUTURE_FILE, index=False)

    upcoming_week = int(
        pd.to_numeric(output["week"], errors="coerce").dropna().min()
    )
    upcoming_season = int(
        pd.to_numeric(
            output.loc[output["week"].eq(upcoming_week), "season"],
            errors="coerce",
        ).dropna().min()
    )

    rankings = output[
        output["season"].eq(upcoming_season)
        & output["week"].eq(upcoming_week)
        & output["selected_team_qb"].eq(1)
    ].copy()

    rankings = rankings.sort_values(
        "gridironiq_projection",
        ascending=False,
    ).reset_index(drop=True)
    rankings.insert(0, "rank", rankings.index + 1)

    rankings.to_csv(RANKINGS_FILE, index=False)

    print("GRIDIRONIQ WEEKLY QB RANKINGS")
    print("=" * 82)
    print(
        f"Season {upcoming_season} | Week {upcoming_week} | "
        f"{len(rankings)} projected team QBs"
    )
    print(
        "Production rule: highest role-probability QB per team receives "
        "the fantasy projection.\n"
    )

    display = rankings[
        [
            "rank",
            "player_name",
            "team",
            "opponent",
            "home_away",
            "meaningful_role_probability",
            "gridironiq_projection",
            "team_spread_line",
            "game_total_line",
        ]
    ].copy()

    print(
        display.to_string(
            index=False,
            formatters={
                "meaningful_role_probability": "{:.1%}".format,
                "gridironiq_projection": "{:.2f}".format,
                "team_spread_line": "{:.1f}".format,
                "game_total_line": "{:.1f}".format,
            },
        )
    )

    print(f"\nSaved weekly rankings to: {RANKINGS_FILE}")
    print(f"Saved all future candidates to: {ALL_FUTURE_FILE}")


if __name__ == "__main__":
    main()
