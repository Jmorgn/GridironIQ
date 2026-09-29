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


def _safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_reference_medians(df: pd.DataFrame) -> dict[str, float]:
    """Reference values from completed starter-level historical QB games."""
    reference = df[
        df["game_completed"].eq(1)
        & df["start_like_role"].eq(1)
    ].copy()

    columns = [
        "avg_fp_last_3",
        "game_total_line",
        "opp_avg_pass_yards_allowed_last_3",
        "opp_avg_def_sacks_last_3",
    ]

    medians: dict[str, float] = {}
    for column in columns:
        if column in reference.columns:
            value = pd.to_numeric(
                reference[column], errors="coerce"
            ).median()
            if pd.notna(value):
                medians[column] = float(value)

    return medians


def explain_candidate(
    row: pd.Series,
    role_probability: float,
    medians: dict[str, float],
) -> tuple[str, str]:
    """Return descriptive pregame signals, not exact model attributions."""
    positives: list[tuple[float, str]] = []
    negatives: list[tuple[float, str]] = []

    recent = _safe_float(row.get("avg_fp_last_3"))
    recent_ref = medians.get("avg_fp_last_3")
    if recent is not None and recent_ref is not None:
        delta = recent - recent_ref
        if delta >= 3:
            positives.append(
                (min(abs(delta) / 8, 2.0),
                 f"Recent form strong ({recent:.1f} FP avg last 3)")
            )
        elif delta <= -3:
            negatives.append(
                (min(abs(delta) / 8, 2.0),
                 f"Recent form below starter norm ({recent:.1f} FP last 3)")
            )

    total = _safe_float(row.get("game_total_line"))
    total_ref = medians.get("game_total_line")
    if total is not None and total_ref is not None:
        delta = total - total_ref
        if delta >= 3:
            positives.append(
                (min(abs(delta) / 7, 1.5),
                 f"High-scoring environment ({total:.1f} game total)")
            )
        elif delta <= -3:
            negatives.append(
                (min(abs(delta) / 7, 1.5),
                 f"Lower-scoring environment ({total:.1f} game total)")
            )

    spread = _safe_float(row.get("team_spread_line"))
    if spread is not None:
        if spread >= 3:
            positives.append(
                (min(abs(spread) / 7, 1.5),
                 f"Team favored by {spread:.1f}")
            )
        elif spread <= -3:
            negatives.append(
                (min(abs(spread) / 7, 1.5),
                 f"Team underdog by {abs(spread):.1f}")
            )

    pass_allowed = _safe_float(
        row.get("opp_avg_pass_yards_allowed_last_3")
    )
    pass_ref = medians.get("opp_avg_pass_yards_allowed_last_3")
    if pass_allowed is not None and pass_ref is not None:
        delta = pass_allowed - pass_ref
        if delta >= 20:
            positives.append(
                (min(abs(delta) / 60, 1.5),
                 f"Opponent allowing {pass_allowed:.0f} pass yds/game recently")
            )
        elif delta <= -20:
            negatives.append(
                (min(abs(delta) / 60, 1.5),
                 f"Opponent allowing only {pass_allowed:.0f} pass yds/game recently")
            )

    sacks = _safe_float(row.get("opp_avg_def_sacks_last_3"))
    sack_ref = medians.get("opp_avg_def_sacks_last_3")
    if sacks is not None and sack_ref is not None:
        delta = sacks - sack_ref
        if delta >= 0.75:
            negatives.append(
                (min(abs(delta) / 2, 1.25),
                 f"Opponent pass rush hot ({sacks:.1f} sacks/game last 3)")
            )
        elif delta <= -0.75:
            positives.append(
                (min(abs(delta) / 2, 1.25),
                 f"Opponent generating few sacks ({sacks:.1f}/game last 3)")
            )

    rest_advantage = _safe_float(row.get("rest_advantage"))
    if rest_advantage is not None:
        if rest_advantage >= 2:
            positives.append(
                (0.6, f"Rest advantage (+{rest_advantage:.0f} days)")
            )
        elif rest_advantage <= -2:
            negatives.append(
                (0.6, f"Rest disadvantage ({rest_advantage:.0f} days)")
            )

    if str(row.get("home_away", "")).lower() == "home":
        positives.append((0.25, "Home game"))

    wind = _safe_float(row.get("game_wind"))
    if wind is not None and wind >= 15:
        negatives.append((0.8, f"Wind risk ({wind:.0f} mph)"))

    injury_score = _safe_float(row.get("injury_status_score"))
    if injury_score is not None and injury_score >= 2:
        negatives.append((1.5, "Meaningful injury/practice concern"))
    elif _safe_float(row.get("on_injury_report")) == 1:
        negatives.append((0.5, "Listed on injury report"))

    if role_probability >= 0.90:
        positives.append(
            (1.5, f"Very high role confidence ({role_probability:.0%})")
        )
    elif role_probability < 0.75:
        negatives.append(
            (1.5, f"Lower role confidence ({role_probability:.0%})")
        )

    positives = sorted(positives, key=lambda x: x[0], reverse=True)
    negatives = sorted(negatives, key=lambda x: x[0], reverse=True)

    positive_text = "; ".join(text for _, text in positives[:3])
    negative_text = "; ".join(text for _, text in negatives[:3])

    return positive_text or "No major positive signal", negative_text or "No major negative signal"


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

    reference_medians = build_reference_medians(df)

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
            "rest_advantage",
            "game_wind",
            "on_injury_report",
            "injury_status_score",
            "avg_fp_last_3",
            "opp_avg_pass_yards_allowed_last_3",
            "opp_avg_def_sacks_last_3",
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

    explanations = [
        explain_candidate(
            future.iloc[index],
            float(role_probability[index]),
            reference_medians,
        )
        for index in range(len(future))
    ]
    output["key_positives"] = [item[0] for item in explanations]
    output["key_negatives"] = [item[1] for item in explanations]

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

    print("\nWHY GRIDIRONIQ LIKES / DISLIKES THE TOP 10")
    print("=" * 82)
    print(
        "These are descriptive feature signals, not exact Random Forest "
        "feature-attribution values."
    )

    for _, row in rankings.head(10).iterrows():
        print(
            f"\n#{int(row['rank'])} {row['player_name']} "
            f"({row['team']} vs {row['opponent']}) — "
            f"{row['gridironiq_projection']:.2f} FP"
        )
        print(f"  + {row['key_positives']}")
        print(f"  - {row['key_negatives']}")

    print(f"\nSaved weekly rankings to: {RANKINGS_FILE}")
    print(f"Saved all future candidates to: {ALL_FUTURE_FILE}")


if __name__ == "__main__":
    main()
