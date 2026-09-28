"""Build the GridironIQ v2 pregame QB candidate dataset.

Unlike v1, which only contains QBs who recorded game activity, v2 starts from
pregame depth charts. This allows the model to represent realistic fantasy
candidates who may later be inactive, replaced, or record zero points.

Targets:
- actual_fantasy_points
- played_any_snap
- start_like_role (>= 50% offensive snaps)

All model features are based on information available before the game.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_qb_candidates import load_depth_candidates, normalize_team
from build_qb_model_dataset import (
    add_injury_features,
    build_defense_features,
    custom_fantasy_points,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED_DIR / "team_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_META_FILE = RAW_DIR / "players.csv"
SCHEDULE_FILE = RAW_DIR / "games.csv"
OUTPUT_FILE = PROCESSED_DIR / "qb_v2_candidate_dataset.csv"


def num(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def week_key(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["season"], errors="coerce") * 100
        + pd.to_numeric(df["week"], errors="coerce")
    )


def build_schedule_team_weeks() -> pd.DataFrame:
    games = pd.read_csv(SCHEDULE_FILE, low_memory=False)

    if "game_type" in games.columns:
        games = games[games["game_type"].astype(str).eq("REG")].copy()

    rows = []

    for _, game in games.iterrows():
        season = pd.to_numeric(game.get("season"), errors="coerce")
        week = pd.to_numeric(game.get("week"), errors="coerce")

        if pd.isna(season) or pd.isna(week):
            continue

        home = normalize_team(pd.Series([str(game.get("home_team"))])).iloc[0]
        away = normalize_team(pd.Series([str(game.get("away_team"))])).iloc[0]

        home_rest = pd.to_numeric(game.get("home_rest"), errors="coerce")
        away_rest = pd.to_numeric(game.get("away_rest"), errors="coerce")
        spread = pd.to_numeric(game.get("spread_line"), errors="coerce")
        total = pd.to_numeric(game.get("total_line"), errors="coerce")
        temp = pd.to_numeric(game.get("temp"), errors="coerce")
        wind = pd.to_numeric(game.get("wind"), errors="coerce")

        common = {
            "season": int(season),
            "week": int(week),
            "game_id": game.get("game_id"),
            "roof": game.get("roof"),
            "surface": game.get("surface"),
            "game_temp": temp,
            "game_wind": wind,
            "game_total_line": total,
            "neutral_site": int(
                str(game.get("location", "")).lower() == "neutral"
            ),
        }

        rows.append(
            {
                **common,
                "team": home,
                "opponent": away,
                "home_away": "home",
                "team_rest": home_rest,
                "opponent_rest": away_rest,
                "rest_advantage": home_rest - away_rest
                if pd.notna(home_rest) and pd.notna(away_rest)
                else np.nan,
                "team_spread_line": spread,
            }
        )

        rows.append(
            {
                **common,
                "team": away,
                "opponent": home,
                "home_away": "away",
                "team_rest": away_rest,
                "opponent_rest": home_rest,
                "rest_advantage": away_rest - home_rest
                if pd.notna(home_rest) and pd.notna(away_rest)
                else np.nan,
                "team_spread_line": -spread if pd.notna(spread) else np.nan,
            }
        )

    return pd.DataFrame(rows).drop_duplicates(["season", "week", "team"])


def build_history_snapshots() -> pd.DataFrame:
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    qbs = players[players["position"].astype(str).eq("QB")].copy()
    qbs["team"] = normalize_team(qbs["team"].astype(str))

    activity = num(qbs, "attempts") + num(qbs, "carries") + num(qbs, "receptions")
    qbs = qbs[activity.gt(0)].copy()

    qbs["custom_fantasy_points"] = custom_fantasy_points(qbs)
    qbs["_week_key"] = week_key(qbs)
    qbs = qbs.sort_values(["player_id", "_week_key"]).copy()

    stat_map = {
        "custom_fantasy_points": "fp",
        "completions": "completions",
        "attempts": "attempts",
        "passing_yards": "pass_yards",
        "passing_tds": "pass_tds",
        "passing_interceptions": "interceptions",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
    }

    for source, short in stat_map.items():
        if source not in qbs.columns:
            qbs[source] = 0.0
        qbs[source] = pd.to_numeric(qbs[source], errors="coerce").fillna(0.0)

        qbs[f"snapshot_previous_{short}"] = qbs[source]

        for window in (3, 5):
            qbs[f"snapshot_avg_{short}_last_{window}"] = (
                qbs.groupby("player_id", sort=False)[source]
                .transform(lambda s: s.rolling(window, min_periods=1).mean())
            )

    snap = qbs[
        [
            "player_id",
            "_week_key",
            *[
                c
                for c in qbs.columns
                if c.startswith("snapshot_")
            ],
        ]
    ].copy()

    return snap


def add_prior_player_history(candidates: pd.DataFrame) -> pd.DataFrame:
    history = build_history_snapshots().sort_values(
        ["_week_key", "player_id"]
    )

    candidates = candidates.copy()
    candidates["_week_key"] = week_key(candidates)
    candidates["_candidate_order"] = np.arange(len(candidates))

    left = candidates.sort_values(["_week_key", "player_id"]).copy()

    merged = pd.merge_asof(
        left,
        history,
        on="_week_key",
        by="player_id",
        direction="backward",
        allow_exact_matches=False,
    )

    rename = {}
    for column in merged.columns:
        if column.startswith("snapshot_previous_"):
            rename[column] = column.replace("snapshot_", "", 1)
        elif column.startswith("snapshot_avg_"):
            rename[column] = column.replace("snapshot_", "", 1)

    merged = merged.rename(columns=rename)
    return merged.sort_values("_candidate_order").drop(
        columns=["_candidate_order"]
    )


def add_actual_targets(candidates: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    qbs = players[players["position"].astype(str).eq("QB")].copy()
    qbs["team"] = normalize_team(qbs["team"].astype(str))
    qbs["actual_fantasy_points"] = custom_fantasy_points(qbs)

    actual = (
        qbs.groupby(["player_id", "season", "week", "team"], as_index=False)
        .agg(actual_fantasy_points=("actual_fantasy_points", "sum"))
    )
    actual["had_stat_row"] = 1

    out = candidates.merge(
        actual,
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    out["actual_fantasy_points"] = out["actual_fantasy_points"].fillna(0.0)
    out["had_stat_row"] = out["had_stat_row"].fillna(0).astype(int)
    return out


def add_snap_targets(candidates: pd.DataFrame) -> pd.DataFrame:
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    meta = pd.read_csv(PLAYERS_META_FILE, low_memory=False)

    id_map = meta[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")

    snaps = snaps[snaps["position"].astype(str).eq("QB")].copy()
    snaps["team"] = normalize_team(snaps["team"].astype(str))
    snaps["offense_snaps"] = pd.to_numeric(
        snaps["offense_snaps"], errors="coerce"
    ).fillna(0.0)
    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"], errors="coerce"
    ).fillna(0.0)

    snaps = snaps.merge(
        id_map,
        how="left",
        left_on="pfr_player_id",
        right_on="pfr_id",
    )

    snap_week = (
        snaps[
            [
                "gsis_id",
                "season",
                "week",
                "team",
                "offense_snaps",
                "offense_pct",
            ]
        ]
        .rename(columns={"gsis_id": "player_id"})
        .drop_duplicates(["player_id", "season", "week", "team"])
    )

    out = candidates.merge(
        snap_week,
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    out["offense_snaps"] = out["offense_snaps"].fillna(0.0)
    out["offense_pct"] = out["offense_pct"].fillna(0.0)

    out["played_any_snap"] = (
        out["offense_snaps"].gt(0) | out["had_stat_row"].eq(1)
    ).astype(int)
    out["start_like_role"] = out["offense_pct"].ge(50).astype(int)
    return out


def add_names(candidates: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(PLAYERS_META_FILE, low_memory=False)

    name_candidates = [
        "display_name",
        "full_name",
        "football_name",
        "short_name",
    ]
    name_col = next((c for c in name_candidates if c in meta.columns), None)

    if name_col is None:
        candidates["player_name"] = candidates["player_id"]
        return candidates

    names = (
        meta[["gsis_id", name_col]]
        .rename(columns={"gsis_id": "player_id", name_col: "player_name"})
        .drop_duplicates("player_id")
    )
    return candidates.merge(names, how="left", on="player_id")


def main() -> None:
    print("Building GridironIQ v2 QB candidate table...")

    candidates = load_depth_candidates()
    candidates = candidates[candidates["season"].between(2021, 2026)].copy()

    schedule = build_schedule_team_weeks()
    candidates = candidates.merge(
        schedule,
        how="inner",
        on=["season", "week", "team"],
    )

    candidates = add_prior_player_history(candidates)
    candidates = add_injury_features(candidates)

    team = pd.read_csv(TEAM_FILE, low_memory=False)
    team["team"] = normalize_team(team["team"].astype(str))
    team["opponent_team"] = normalize_team(team["opponent_team"].astype(str))

    defense = build_defense_features(team)
    defense["defense_team"] = normalize_team(
        defense["defense_team"].astype(str)
    )

    candidates = candidates.merge(
        defense,
        how="left",
        left_on=["season", "week", "opponent"],
        right_on=["season", "week", "defense_team"],
    )

    candidates = add_actual_targets(candidates)
    candidates = add_snap_targets(candidates)
    candidates = add_names(candidates)

    output_columns = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "depth_chart_qb_rank",
        "listed_qb1",
        "team_rest",
        "opponent_rest",
        "rest_advantage",
        "neutral_site",
        "roof",
        "surface",
        "game_temp",
        "game_wind",
        "team_spread_line",
        "game_total_line",
        "on_injury_report",
        "injury_questionable",
        "injury_doubtful",
        "injury_out",
        "practice_dnp",
        "practice_limited",
        "practice_full",
        "injury_status_score",
        "previous_fp",
        "avg_fp_last_3",
        "avg_fp_last_5",
        "avg_completions_last_3",
        "avg_completions_last_5",
        "avg_attempts_last_3",
        "avg_attempts_last_5",
        "avg_pass_yards_last_3",
        "avg_pass_yards_last_5",
        "avg_pass_tds_last_3",
        "avg_pass_tds_last_5",
        "avg_interceptions_last_3",
        "avg_interceptions_last_5",
        "avg_rush_yards_last_3",
        "avg_rush_yards_last_5",
        "avg_rush_tds_last_3",
        "avg_rush_tds_last_5",
        "opp_avg_pass_yards_allowed_last_3",
        "opp_avg_pass_yards_allowed_last_5",
        "opp_avg_pass_tds_allowed_last_3",
        "opp_avg_pass_tds_allowed_last_5",
        "opp_avg_def_interceptions_last_3",
        "opp_avg_def_interceptions_last_5",
        "opp_avg_def_sacks_last_3",
        "opp_avg_def_sacks_last_5",
        "actual_fantasy_points",
        "played_any_snap",
        "start_like_role",
        "offense_snaps",
        "offense_pct",
    ]

    output_columns = [c for c in output_columns if c in candidates.columns]
    model = candidates[output_columns].copy()
    model = model.sort_values(
        ["season", "week", "team", "depth_chart_qb_rank", "player_name"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(OUTPUT_FILE, index=False)

    historical = model[model["season"].between(2021, 2025)]

    print(f"\nHistorical candidate rows: {len(historical):,}")
    print(
        "Played any offensive snap: "
        f"{historical['played_any_snap'].mean() * 100:.1f}%"
    )
    print(
        "Start-like role (>=50% snaps): "
        f"{historical['start_like_role'].mean() * 100:.1f}%"
    )
    print(
        "Zero fantasy-point rows: "
        f"{historical['actual_fantasy_points'].eq(0).mean() * 100:.1f}%"
    )

    qb1 = historical[historical["listed_qb1"].eq(1)]
    print(f"\nHistorical QB1 rows: {len(qb1):,}")
    print(
        "QB1 played any snap: "
        f"{qb1['played_any_snap'].mean() * 100:.1f}%"
    )
    print(
        "QB1 start-like role: "
        f"{qb1['start_like_role'].mean() * 100:.1f}%"
    )

    print(f"\nColumns: {len(model.columns)}")
    print(f"WRITE {OUTPUT_FILE}")
    print("\nSample:")
    print(model.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
