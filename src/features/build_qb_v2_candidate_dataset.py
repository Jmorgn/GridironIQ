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

        home_score = pd.to_numeric(game.get("home_score"), errors="coerce")
        away_score = pd.to_numeric(game.get("away_score"), errors="coerce")
        game_completed = int(pd.notna(home_score) and pd.notna(away_score))

        common = {
            "season": int(season),
            "week": int(week),
            "game_id": game.get("game_id"),
            "game_completed": game_completed,
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
    history = build_history_snapshots().copy()

    candidates = candidates.copy()
    candidates["_week_key"] = week_key(candidates)

    # merge_asof requires the join keys to have exactly the same dtype.
    # Some depth-chart seasons store week as a float (for example 2.0),
    # while the player-history table stores it as an integer. Normalize both
    # sides before the time-aware join.
    candidates["_week_key"] = pd.to_numeric(
        candidates["_week_key"], errors="raise"
    ).astype("int64")
    history["_week_key"] = pd.to_numeric(
        history["_week_key"], errors="raise"
    ).astype("int64")

    candidates["player_id"] = candidates["player_id"].astype(str)
    history["player_id"] = history["player_id"].astype(str)

    candidates["_candidate_order"] = np.arange(len(candidates))

    left = candidates.sort_values(["_week_key", "player_id"]).copy()
    history = history.sort_values(["_week_key", "player_id"]).copy()

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


def build_defense_history_snapshots() -> pd.DataFrame:
    """Build postgame defense snapshots for leakage-safe future-week joins."""
    team = pd.read_csv(TEAM_FILE, low_memory=False)
    team["team"] = normalize_team(team["team"].astype(str))
    team["opponent_team"] = normalize_team(
        team["opponent_team"].astype(str)
    )

    # First build one observed defensive performance per team/game.
    defense = build_defense_features(team)

    # build_defense_features intentionally shifts its rolling metrics so they
    # are safe on same-week historical rows. For v2, future candidate rows do
    # not yet exist in team_weekly, so reconstruct observed game metrics and
    # create snapshots that include each completed game. The candidate then
    # receives the latest snapshot strictly BEFORE its game week.
    from build_qb_model_dataset import find_sacks_allowed_column, col

    sacks_col = find_sacks_allowed_column(team)

    observed = pd.DataFrame(
        {
            "season": team["season"],
            "week": team["week"],
            "defense_team": team["opponent_team"],
            "pass_yards_allowed": col(team, "passing_yards"),
            "pass_tds_allowed": col(team, "passing_tds"),
            "def_interceptions": col(team, "passing_interceptions"),
            "def_sacks": col(team, sacks_col) if sacks_col else 0.0,
        }
    )

    observed = (
        observed.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            pass_yards_allowed=("pass_yards_allowed", "sum"),
            pass_tds_allowed=("pass_tds_allowed", "sum"),
            def_interceptions=("def_interceptions", "sum"),
            def_sacks=("def_sacks", "sum"),
        )
    )

    observed["_week_key"] = week_key(observed).astype("int64")
    observed = observed.sort_values(
        ["defense_team", "_week_key"]
    ).copy()

    metrics = [
        "pass_yards_allowed",
        "pass_tds_allowed",
        "def_interceptions",
        "def_sacks",
    ]

    for metric in metrics:
        observed[metric] = pd.to_numeric(
            observed[metric], errors="coerce"
        ).fillna(0.0)

        for window in (3, 5):
            observed[f"snapshot_opp_avg_{metric}_last_{window}"] = (
                observed.groupby("defense_team", sort=False)[metric]
                .transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).mean()
                )
            )

    keep = [
        "defense_team",
        "_week_key",
        *[
            c
            for c in observed.columns
            if c.startswith("snapshot_opp_avg_")
        ],
    ]
    return observed[keep].copy()


def add_prior_defense_history(candidates: pd.DataFrame) -> pd.DataFrame:
    """Attach opponent defense form from the latest completed prior game."""
    history = build_defense_history_snapshots().copy()
    out = candidates.copy()

    out["_week_key"] = week_key(out).astype("int64")
    out["opponent"] = out["opponent"].astype(str)
    history["defense_team"] = history["defense_team"].astype(str)

    out["_candidate_order_def"] = np.arange(len(out))

    left = out.sort_values(["_week_key", "opponent"]).copy()
    right = history.sort_values(
        ["_week_key", "defense_team"]
    ).copy()

    merged = pd.merge_asof(
        left,
        right,
        left_on="_week_key",
        right_on="_week_key",
        left_by="opponent",
        right_by="defense_team",
        direction="backward",
        allow_exact_matches=False,
    )

    rename = {
        column: column.replace("snapshot_", "", 1)
        for column in merged.columns
        if column.startswith("snapshot_opp_avg_")
    }

    merged = merged.rename(columns=rename)
    return (
        merged.sort_values("_candidate_order_def")
        .drop(
            columns=[
                "_candidate_order_def",
                "defense_team",
            ],
            errors="ignore",
        )
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
    completed = out["game_completed"].eq(1)

    # A completed game with no player stat row is a legitimate zero-point
    # candidate. Future/unplayed games must remain unlabeled rather than being
    # silently converted to zeroes.
    out.loc[completed, "actual_fantasy_points"] = (
        out.loc[completed, "actual_fantasy_points"].fillna(0.0)
    )
    out.loc[completed, "had_stat_row"] = (
        out.loc[completed, "had_stat_row"].fillna(0)
    )
    out.loc[~completed, "actual_fantasy_points"] = np.nan
    out.loc[~completed, "had_stat_row"] = np.nan
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
    completed = out["game_completed"].eq(1)

    out.loc[completed, "offense_snaps"] = (
        out.loc[completed, "offense_snaps"].fillna(0.0)
    )
    out.loc[completed, "offense_pct"] = (
        out.loc[completed, "offense_pct"].fillna(0.0)
    )

    # nflverse snap share is normally stored as a fraction (0.94 = 94%).
    # Normalize defensively in case a source ever supplies 94 instead.
    normalized_pct = out["offense_pct"].where(
        out["offense_pct"].le(1.0),
        out["offense_pct"] / 100.0,
    )

    out["played_any_snap"] = np.nan
    out["start_like_role"] = np.nan

    out.loc[completed, "played_any_snap"] = (
        out.loc[completed, "offense_snaps"].gt(0)
        | out.loc[completed, "had_stat_row"].eq(1)
    ).astype(int)

    out.loc[completed, "start_like_role"] = (
        normalized_pct.loc[completed].ge(0.50).astype(int)
    )

    out.loc[~completed, ["offense_snaps", "offense_pct"]] = np.nan
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

    candidates = add_prior_defense_history(candidates)

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
        "game_completed",
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

    historical = model[
        model["season"].between(2021, 2025)
        & model["game_completed"].eq(1)
    ].copy()

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

    live_unplayed = model[
        model["season"].eq(2026) & model["game_completed"].eq(0)
    ]
    print(
        f"\n2026 future/unplayed candidate rows kept unlabeled: "
        f"{len(live_unplayed):,}"
    )

    defense_feature = "opp_avg_pass_yards_allowed_last_3"
    if len(live_unplayed) and defense_feature in live_unplayed.columns:
        future_defense_coverage = (
            live_unplayed[defense_feature].notna().mean() * 100
        )
        print(
            "Future opponent-defense feature coverage: "
            f"{future_defense_coverage:.1f}%"
        )

    print(f"\nColumns: {len(model.columns)}")
    print(f"WRITE {OUTPUT_FILE}")
    print("\nSample:")
    print(model.tail(15).to_string(index=False))


if __name__ == "__main__":
    main()
