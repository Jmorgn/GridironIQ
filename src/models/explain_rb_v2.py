"""SHAP explanations for the GridironIQ RB v2 conditional points model.

The official RB v2 projection is:
    P(35%+ offensive snaps) * conditional fantasy points

This module explains the conditional Gradient Boosting fantasy-points model
only. SHAP values are additive contributions to that conditional prediction;
they do not explain the separate workload-probability classifier.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "rb_v2_candidate_dataset.csv"
BUNDLE_FILE = ROOT / "models" / "rb_v2_bundle.joblib"


LABELS = {
    "previous_fp": "Previous fantasy points",
    "avg_fp_last_3": "Fantasy avg, last 3",
    "avg_fp_last_5": "Fantasy avg, last 5",
    "previous_carries": "Previous carries",
    "avg_carries_last_3": "Carries avg, last 3",
    "avg_carries_last_5": "Carries avg, last 5",
    "previous_targets": "Previous targets",
    "avg_targets_last_3": "Targets avg, last 3",
    "avg_targets_last_5": "Targets avg, last 5",
    "previous_receptions": "Previous receptions",
    "avg_receptions_last_3": "Receptions avg, last 3",
    "avg_receptions_last_5": "Receptions avg, last 5",
    "previous_opportunities": "Previous opportunities",
    "avg_opportunities_last_3": "Opportunities avg, last 3",
    "avg_opportunities_last_5": "Opportunities avg, last 5",
    "previous_touches": "Previous touches",
    "avg_touches_last_3": "Touches avg, last 3",
    "avg_touches_last_5": "Touches avg, last 5",
    "avg_rush_yards_last_3": "Rush yards avg, last 3",
    "avg_rush_yards_last_5": "Rush yards avg, last 5",
    "avg_rec_yards_last_3": "Receiving yards avg, last 3",
    "avg_rec_yards_last_5": "Receiving yards avg, last 5",
    "avg_carry_share_last_3": "Carry share avg, last 3",
    "avg_carry_share_last_5": "Carry share avg, last 5",
    "avg_target_share_last_3": "Target share avg, last 3",
    "avg_target_share_last_5": "Target share avg, last 5",
    "avg_opportunity_share_last_3": "Opportunity share avg, last 3",
    "avg_opportunity_share_last_5": "Opportunity share avg, last 5",
    "avg_offense_pct_last_3": "Snap share avg, last 3",
    "avg_offense_pct_last_5": "Snap share avg, last 5",
    "depth_chart_rb_rank": "Depth-chart RB rank",
    "listed_rb1": "Listed RB1",
    "team_rb_candidates": "RB-room candidate count",
    "rb_room_player_share_opportunities": "RB-room opportunity share",
    "rb_room_max_other_prior_opportunities": "Top teammate opportunities, last 3",
    "rb_room_player_share_snap_share": "RB-room snap-share portion",
    "rb_room_max_other_prior_snap_share": "Top teammate snap share, last 3",
    "rb_room_player_share_fantasy_points": "RB-room fantasy-point share",
    "rb_room_max_other_prior_fantasy_points": "Top teammate fantasy avg, last 3",
    "rb_room_out_count": "RB-room players listed out",
    "opp_avg_rush_yards_allowed_last_3": "Opponent rush yds allowed, last 3",
    "opp_avg_rush_yards_allowed_last_5": "Opponent rush yds allowed, last 5",
    "opp_avg_rush_tds_allowed_last_3": "Opponent rush TD allowed, last 3",
    "opp_avg_rush_tds_allowed_last_5": "Opponent rush TD allowed, last 5",
    "opp_avg_pass_yards_allowed_last_3": "Opponent pass yds allowed, last 3",
    "opp_avg_pass_tds_allowed_last_3": "Opponent pass TD allowed, last 3",
    "team_spread_line": "Team spread",
    "game_total_line": "Game total",
    "team_rest": "Team rest",
    "opponent_rest": "Opponent rest",
    "rest_advantage": "Rest advantage",
    "game_temp": "Temperature",
    "game_wind": "Wind",
    "on_injury_report": "On injury report",
    "injury_questionable": "Questionable",
    "injury_doubtful": "Doubtful",
    "injury_out": "Out",
    "practice_dnp": "Practice DNP",
    "practice_limited": "Practice limited",
    "practice_full": "Practice full",
    "injury_status_score": "Injury status score",
    "home_away": "Home / away",
    "roof": "Roof",
    "surface": "Surface",
    "team": "Team",
    "opponent": "Opponent",
}


def _transformed_source_features(preprocessor) -> list[str]:
    """Map transformed columns back to their original source features."""
    source_names: list[str] = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue

        columns = list(columns)

        if name == "numeric":
            source_names.extend(columns)
            continue

        if name == "categorical":
            onehot = transformer.named_steps["onehot"]
            for column, categories in zip(columns, onehot.categories_):
                source_names.extend([column] * len(categories))
            continue

        try:
            count = transformer.get_feature_names_out(columns).shape[0]
        except Exception:
            count = len(columns)

        if count == len(columns):
            source_names.extend(columns)
        else:
            source_names.extend([name] * count)

    return source_names


def _raw_value(row: pd.Series, feature: str) -> str:
    value = row.get(feature)

    if pd.isna(value):
        return "N/A"

    if isinstance(value, (float, np.floating)):
        if "share" in feature or "pct" in feature:
            return f"{float(value):.1%}"
        return f"{float(value):.2f}"

    return str(value)


def _pretty(feature: str) -> str:
    return LABELS.get(
        feature,
        feature.replace("_", " ").title(),
    )


def explain_conditional_prediction(
    *,
    player_id: str,
    season: int,
    week: int,
    team: str,
    top_n: int = 5,
) -> dict:
    """Explain one current RB's conditional fantasy-points prediction."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Run: py -m pip install shap"
        ) from exc

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing rb_v2_candidate_dataset.csv. "
            "Run py run_weekly.py first."
        )

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Missing rb_v2_bundle.joblib. Train RB v2 first."
        )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    bundle = joblib.load(BUNDLE_FILE)

    match = df[
        df["player_id"].astype(str).eq(str(player_id))
        & df["season"].eq(int(season))
        & df["week"].eq(float(week))
        & df["team"].astype(str).eq(str(team))
    ].copy()

    if match.empty:
        raise ValueError(
            f"Could not find model input row for {player_id}, "
            f"{season} Week {week}, {team}."
        )

    row = match.iloc[[0]].copy()
    features = bundle["features"]
    pipeline = bundle["conditional_regressor"]

    preprocessor = pipeline.named_steps["preprocessor"]
    regressor = pipeline.named_steps["model"]

    transformed = preprocessor.transform(row[features])
    source_features = _transformed_source_features(preprocessor)

    if transformed.shape[1] != len(source_features):
        raise RuntimeError(
            "Could not map transformed model columns back to source features."
        )

    explainer = shap.TreeExplainer(regressor)
    explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    if values.ndim == 2:
        shap_values = values[0]
    else:
        shap_values = values.reshape(-1)

    base_values = np.asarray(explanation.base_values).reshape(-1)
    baseline = float(base_values[0])

    raw_prediction = float(regressor.predict(transformed)[0])
    pipeline_prediction = float(pipeline.predict(row[features])[0])

    contributions: dict[str, float] = {}
    for feature, contribution in zip(source_features, shap_values):
        contributions[feature] = (
            contributions.get(feature, 0.0) + float(contribution)
        )

    source_row = row.iloc[0]
    details = [
        {
            "feature": feature,
            "label": _pretty(feature),
            "value": _raw_value(source_row, feature),
            "contribution": contribution,
        }
        for feature, contribution in contributions.items()
    ]

    positives = sorted(
        [item for item in details if item["contribution"] > 0],
        key=lambda item: item["contribution"],
        reverse=True,
    )[:top_n]

    negatives = sorted(
        [item for item in details if item["contribution"] < 0],
        key=lambda item: item["contribution"],
    )[:top_n]

    return {
        "baseline": baseline,
        "raw_prediction": raw_prediction,
        "pipeline_prediction": pipeline_prediction,
        "positive": positives,
        "negative": negatives,
        "sum_contributions": float(sum(contributions.values())),
    }
