"""Shared schedule and configuration for prospective GridironIQ evaluation.

The schedule's gameday/gametime are interpreted as US Eastern local time,
as published by the nflverse/nfldata games.csv source. An unknown kickoff
is NOT guessed; such games cannot be stamped as verified pregame forecasts.
"""

from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
REPORT_DIR = PROCESSED_DIR / "evaluation"

POSITIONS = {
    "QB": {
        "probability_column": "meaningful_role_probability",
        "role_threshold": 0.50,
        "starter_count": 12,
        "role_name": ">=50% offensive snaps",
        "has_role_model": True,
    },
    "RB": {
        "probability_column": "rb_role_probability",
        "role_threshold": 0.35,
        "starter_count": 24,
        "role_name": ">=35% offensive snaps",
        "has_role_model": True,
    },
    "WR": {
        "probability_column": "wr_role_probability",
        "role_threshold": 0.65,
        "starter_count": 24,
        "role_name": ">=65% offensive snaps",
        "has_role_model": True,
    },
    "TE": {
        "probability_column": "te_role_probability",
        "role_threshold": 0.50,
        "starter_count": 12,
        "role_name": ">=50% offensive snaps",
        "has_role_model": True,
    },
    "K": {
        "probability_column": None,
        "role_threshold": None,
        "starter_count": 12,
        "role_name": "direct regression; no role model",
        "has_role_model": False,
    },
}

def normalize_team(series: pd.Series) -> pd.Series:
    return series.astype(str).replace(
        {"LAR": "LA", "WSH": "WAS"}
    )

def kickoff_utc(gameday: object, gametime: object):
    """Convert explicit Eastern schedule date/time to UTC; never invent TBD."""
    if pd.isna(gameday) or pd.isna(gametime):
        return pd.NaT

    parsed_date = pd.to_datetime(gameday, errors="coerce")
    if pd.isna(parsed_date):
        return pd.NaT

    clock = str(gametime).strip()
    try:
        hour, minute, *seconds = [int(value) for value in clock.split(":")]
        if len(seconds) > 1:
            return pd.NaT
        kickoff = datetime.combine(
            parsed_date.date(),
            time(hour, minute, seconds[0] if seconds else 0),
        )
        eastern = ZoneInfo("America/New_York")
        return pd.Timestamp(
            kickoff.replace(tzinfo=eastern).astimezone(timezone.utc)
        )
    except (ValueError, ZoneInfoNotFoundError):
        return pd.NaT

def load_schedule() -> pd.DataFrame:
    """One canonical regular-season game row per team and opponent."""
    path = RAW_DIR / "games.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the weekly data refresh first."
        )
    games = pd.read_csv(path, low_memory=False)
    required = [
        "season", "week", "game_type", "home_team", "away_team",
        "gameday", "gametime", "home_score", "away_score",
    ]
    missing = [column for column in required if column not in games.columns]
    if missing:
        raise RuntimeError(
            "Schedule is missing required fields: " + ", ".join(missing)
        )
    games = games[
        games["game_type"].astype(str).eq("REG")
    ].copy()
    rows = []
    for _, game in games.iterrows():
        season = pd.to_numeric(game["season"], errors="coerce")
        week = pd.to_numeric(game["week"], errors="coerce")
        if pd.isna(season) or pd.isna(week):
            continue

        home = normalize_team(pd.Series([game["home_team"]])).iloc[0]
        away = normalize_team(pd.Series([game["away_team"]])).iloc[0]
        kickoff = kickoff_utc(game["gameday"], game["gametime"])
        completed = bool(
            pd.notna(game["home_score"]) and pd.notna(game["away_score"])
        )
        game_id = str(game.get("game_id", ""))
        for team, opponent in ((home, away), (away, home)):
            rows.append({
                "season": int(season),
                "week": int(week),
                "team": team,
                "opponent": opponent,
                "game_id": game_id,
                "kickoff_utc": kickoff,
                "game_completed": completed,
            })
    schedule = pd.DataFrame(rows)
    if schedule.empty:
        raise RuntimeError("No regular-season games found in games.csv.")
    keys = ["season", "week", "team", "opponent"]
    duplicate = schedule.duplicated(keys, keep=False)
    if duplicate.any():
        sample = schedule.loc[duplicate, keys].head(4).to_dict("records")
        raise RuntimeError(
            f"Duplicate schedule team-weeks; cannot match games: {sample}"
        )
    schedule["kickoff_utc"] = pd.to_datetime(
        schedule["kickoff_utc"], utc=True
    )
    return schedule

def candidate_file(position: str) -> Path:
    return PROCESSED_DIR / f"{position.lower()}_v2_all_future_candidates.csv"

def model_file(position: str) -> Path:
    return ROOT / "models" / f"{position.lower()}_v2_bundle.joblib"

def utc_now() -> datetime:
    return datetime.now(timezone.utc)
