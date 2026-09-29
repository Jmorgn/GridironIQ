"""SHAP explanations for the GridironIQ QB v2 conditional points model.

The official v2 projection uses two stages:
1. role probability / team-level QB selection;
2. conditional fantasy-points regression.

This module explains stage 2 only. SHAP values are therefore additive
contributions to the Random Forest's conditional fantasy-points prediction,
not to the role classifier or the final team-selection gate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "qb_v2_candidate_dataset.csv"
BUNDLE_FILE = ROOT / "models" / "qb_v2_bundle.joblib"


LABELS = {
    "previous_fp": "Previous fantasy points",
    "avg_fp_last_3": "Fantasy avg, last 3",
    "avg_fp_last_5": "Fantasy avg, last 5",
    "avg_completions_last_3": "Completions avg, last 3",
    "avg_completions_last_5": "Completions avg, last 5",
    "avg_attempts_last_3": "Attempts avg, last 3",
    "avg_attempts_last_5": "Attempts avg, last 5",
    "avg_pass_yards_last_3": "Pass yards avg, last 3",
    "avg_pass_yards_last_5": "Pass yards avg, last 5",
    "avg_pass_tds_last_3": "Pass TD avg, last 3",
    "avg_pass_tds_last_5": "Pass TD avg, last 5",
    "avg_interceptions_last_3": "INT avg, last 3",
    "avg_interceptions_last_5": "INT avg, last 5",
    "avg_rush_yards_last_3": "Rush yards avg, last 3",
    "avg_rush_yards_last_5": "Rush yards avg, last 5",
    "avg_rush_tds_last_3": "Rush TD avg, last 3",
    "avg_rush_tds_last_5": "Rush TD avg, last 5",
    "opp_avg_pass_yards_allowed_last_3": "Opponent pass yds allowed, last 3",
    "opp_avg_pass_yards_allowed_last_5": "Opponent pass yds allowed, last 5",
    "opp_avg_pass_tds_allowed_last_3": "Opponent pass TD allowed, last 3",
    "opp_avg_pass_tds_allowed_last_5": "Opponent pass TD allowed, last 5",
    "opp_avg_def_interceptions_last_3": "Opponent INT avg, last 3",
    "opp_avg_def_interceptions_last_5": "Opponent INT avg, last 5",
    "opp_avg_def_sacks_last_3": "Opponent sacks avg, last 3",
    "opp_avg_def_sacks_last_5": "Opponent sacks avg, last 5",
    "team_spread_line": "Team spread",
    "game_total_line": "Game total",
    "team_rest": "Team rest",
    "opponent_rest": "Opponent rest",
    "rest_advantage": "Rest advantage",
    "game_temp": "Temperature",
    "game_wind": "Wind",
    "depth_chart_qb_rank": "Depth-chart QB rank",
    "listed_qb1": "Listed QB1",
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
    """Map each transformed model column back to its original source feature."""
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

        # Generic fallback for any future transformer family.
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
    """Explain one current QB's conditional fantasy-points prediction."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Run: py -m pip install shap"
        ) from exc

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Missing qb_v2_candidate_dataset.csv. Run py run_weekly.py first."
        )

    if not BUNDLE_FILE.exists():
        raise FileNotFoundError(
            "Missing qb_v2_bundle.joblib. Train QB v2 first."
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
    forest = pipeline.named_steps["model"]

    transformed = preprocessor.transform(row[features])
    source_features = _transformed_source_features(preprocessor)

    if transformed.shape[1] != len(source_features):
        raise RuntimeError(
            "Could not map transformed model columns back to source features."
        )

    explainer = shap.TreeExplainer(forest)
    explanation = explainer(transformed)

    values = np.asarray(explanation.values)
    if values.ndim == 2:
        shap_values = values[0]
    else:
        shap_values = values.reshape(-1)

    base_values = np.asarray(explanation.base_values).reshape(-1)
    baseline = float(base_values[0])

    raw_prediction = float(forest.predict(transformed)[0])
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
