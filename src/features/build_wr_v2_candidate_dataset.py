"""Build the GridironIQ v2 pregame WR candidate dataset.

WR v2 starts from pregame depth charts instead of postgame participation.
That lets the model represent inactive receivers, rotations, injury replacements,
and multiple fantasy-relevant WRs on the same team.

Candidate role outcomes retained for model experiments:
- snap_50_role: >=50% offensive snaps
- snap_65_role: >=65% offensive snaps
- target_5_role: >=5 targets
- target_share_20_role: >=20% of team targets
- snap50_or_target5_role: either 50% snaps or 5+ targets
- snap65_or_target5_role: either 65% snaps or 5+ targets

All rolling features are shifted so the current game's outcome is never used
to predict itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_wr_candidates import load_wr_depth_candidates
from build_qb_model_dataset import (
    col,
    custom_fantasy_points,
    normalize_team_code,
)
from build_qb_v2_candidate_dataset import build_schedule_team_weeks
from build_wr_model_dataset import add_wr_injury_features

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED_DIR / "team_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_FILE = RAW_DIR / "players.csv"
OUTPUT_FILE = PROCESSED_DIR / "wr_v2_candidate_dataset.csv"


def week_key(df: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(df["season"], errors="coerce") * 100
        + pd.to_numeric(df["week"], errors="coerce")
    ).astype("int64")


def add_names(candidates: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(PLAYERS_FILE, low_memory=False)
    name_col = next(
        (
            column
            for column in [
                "display_name",
                "full_name",
                "football_name",
                "short_name",
            ]
            if column in meta.columns
        ),
        None,
    )

    out = candidates.copy()
    if name_col is None:
        out["player_name"] = out["player_id"]
        return out

    names = (
        meta[["gsis_id", name_col]]
        .rename(
            columns={
                "gsis_id": "player_id",
                name_col: "player_name",
            }
        )
        .drop_duplicates("player_id")
    )
    return out.merge(names, how="left", on="player_id")


def add_actual_results(candidates: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    players["team"] = normalize_team_code(
        players["team"].astype(str)
    )

    players = pd.concat(
        [
            players,
            pd.DataFrame(
                {
                    "_team_targets": col(players, "targets"),
                    "_team_receptions": col(players, "receptions"),
                    "_team_rec_yards": col(players, "receiving_yards"),
                },
                index=players.index,
            ),
        ],
        axis=1,
    )

    team_totals = (
        players.groupby(
            ["season", "week", "team"],
            as_index=False,
        )
        .agg(
            team_targets=("_team_targets", "sum"),
            team_receptions=("_team_receptions", "sum"),
            team_rec_yards=("_team_rec_yards", "sum"),
        )
    )

    wrs = players[
        players["position"].astype(str).eq("WR")
    ].copy()

    numeric_map = {
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "carries": "carries",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_air_yards": "air_yards",
        "receiving_yards_after_catch": "yac",
        "receiving_first_downs": "rec_first_downs",
    }

    for source, destination in numeric_map.items():
        wrs[destination] = col(wrs, source)

    wrs["opportunities"] = (
        wrs["targets"] + wrs["carries"]
    )
    wrs["actual_fantasy_points"] = custom_fantasy_points(wrs)

    actual = (
        wrs.groupby(
            ["player_id", "season", "week", "team"],
            as_index=False,
        )
        .agg(
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            rec_yards=("rec_yards", "sum"),
            rec_tds=("rec_tds", "sum"),
            carries=("carries", "sum"),
            rush_yards=("rush_yards", "sum"),
            rush_tds=("rush_tds", "sum"),
            opportunities=("opportunities", "sum"),
            air_yards=("air_yards", "sum"),
            yac=("yac", "sum"),
            rec_first_downs=("rec_first_downs", "sum"),
            actual_fantasy_points=(
                "actual_fantasy_points",
                "sum",
            ),
        )
    )
    actual["had_stat_row"] = 1.0

    out = candidates.merge(
        actual,
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    out = out.merge(
        team_totals,
        how="left",
        on=["season", "week", "team"],
    )

    completed = out["game_completed"].eq(1)
    outcome_columns = [
        "targets",
        "receptions",
        "rec_yards",
        "rec_tds",
        "carries",
        "rush_yards",
        "rush_tds",
        "opportunities",
        "air_yards",
        "yac",
        "rec_first_downs",
        "actual_fantasy_points",
        "had_stat_row",
    ]

    for column in outcome_columns:
        out.loc[completed, column] = (
            out.loc[completed, column].fillna(0.0)
        )
        out.loc[~completed, column] = np.nan

    out.loc[
        ~completed,
        ["team_targets", "team_receptions", "team_rec_yards"],
    ] = np.nan

    out["target_share"] = np.where(
        out["team_targets"].gt(0),
        out["targets"] / out["team_targets"],
        np.nan,
    )
    out["reception_share"] = np.where(
        out["team_receptions"].gt(0),
        out["receptions"] / out["team_receptions"],
        np.nan,
    )
    out["receiving_yard_share"] = np.where(
        out["team_rec_yards"].gt(0),
        out["rec_yards"] / out["team_rec_yards"],
        np.nan,
    )
    out["yards_per_target"] = np.where(
        out["targets"].gt(0),
        out["rec_yards"] / out["targets"],
        np.nan,
    )
    out["catch_rate"] = np.where(
        out["targets"].gt(0),
        out["receptions"] / out["targets"],
        np.nan,
    )

    out["recorded_activity"] = np.nan
    out.loc[completed, "recorded_activity"] = (
        out.loc[completed, "targets"].gt(0)
        | out.loc[completed, "receptions"].gt(0)
        | out.loc[completed, "carries"].gt(0)
    ).astype(int)

    return out


def add_snap_results(candidates: pd.DataFrame) -> pd.DataFrame:
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    meta = pd.read_csv(PLAYERS_FILE, low_memory=False)

    id_map = (
        meta[["gsis_id", "pfr_id"]]
        .dropna()
        .drop_duplicates("gsis_id")
    )

    snaps["team"] = normalize_team_code(
        snaps["team"].astype(str)
    )
    snaps["offense_snaps"] = pd.to_numeric(
        snaps["offense_snaps"],
        errors="coerce",
    ).fillna(0.0)
    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"],
        errors="coerce",
    ).fillna(0.0)
    snaps["offense_pct"] = snaps["offense_pct"].where(
        snaps["offense_pct"].le(1.0),
        snaps["offense_pct"] / 100.0,
    )

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
        .drop_duplicates(
            ["player_id", "season", "week", "team"]
        )
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
    out.loc[
        ~completed,
        ["offense_snaps", "offense_pct"],
    ] = np.nan

    out["played_any_snap"] = np.nan
    out.loc[completed, "played_any_snap"] = (
        out.loc[completed, "offense_snaps"].gt(0)
        | out.loc[completed, "had_stat_row"].eq(1)
    ).astype(int)

    role_targets = [
        "snap_50_role",
        "snap_65_role",
        "target_5_role",
        "target_share_20_role",
        "snap50_or_target5_role",
        "snap65_or_target5_role",
    ]
    for column in role_targets:
        out[column] = np.nan

    out.loc[completed, "snap_50_role"] = (
        out.loc[completed, "offense_pct"].ge(0.50).astype(int)
    )
    out.loc[completed, "snap_65_role"] = (
        out.loc[completed, "offense_pct"].ge(0.65).astype(int)
    )
    out.loc[completed, "target_5_role"] = (
        out.loc[completed, "targets"].ge(5).astype(int)
    )
    out.loc[completed, "target_share_20_role"] = (
        out.loc[completed, "target_share"].ge(0.20).astype(int)
    )
    out.loc[completed, "snap50_or_target5_role"] = (
        out.loc[completed, "snap_50_role"].eq(1)
        | out.loc[completed, "target_5_role"].eq(1)
    ).astype(int)
    out.loc[completed, "snap65_or_target5_role"] = (
        out.loc[completed, "snap_65_role"].eq(1)
        | out.loc[completed, "target_5_role"].eq(1)
    ).astype(int)

    return out


def add_prior_player_history(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.sort_values(
        ["player_id", "season", "week"]
    ).copy()

    metrics = {
        "actual_fantasy_points": "fp",
        "targets": "targets",
        "receptions": "receptions",
        "rec_yards": "rec_yards",
        "rec_tds": "rec_tds",
        "opportunities": "opportunities",
        "yards_per_target": "yards_per_target",
        "catch_rate": "catch_rate",
        "target_share": "target_share",
        "reception_share": "reception_share",
        "receiving_yard_share": "receiving_yard_share",
        "carries": "carries",
        "rush_yards": "rush_yards",
        "rush_tds": "rush_tds",
        "air_yards": "air_yards",
        "yac": "yac",
        "rec_first_downs": "rec_first_downs",
        "offense_snaps": "offense_snaps",
        "offense_pct": "offense_pct",
        "recorded_activity": "recorded_activity",
    }

    for source, short in metrics.items():
        out[source] = pd.to_numeric(
            out[source],
            errors="coerce",
        )

        out[f"previous_{short}"] = (
            out.groupby("player_id", sort=False)[source]
            .shift(1)
        )

        for window in (3, 5):
            out[f"avg_{short}_last_{window}"] = (
                out.groupby("player_id", sort=False)[source]
                .transform(
                    lambda s: s.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

    return out


def add_wr_room_features(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Describe each candidate's pregame competition inside the WR room."""
    out = candidates.copy()
    keys = ["season", "week", "team"]

    out["team_wr_candidates"] = (
        out.groupby(keys)["player_id"].transform("size")
    )

    for source, suffix in [
        ("avg_targets_last_3", "targets"),
        ("avg_target_share_last_3", "target_share"),
        ("avg_offense_pct_last_3", "snap_share"),
        ("avg_fp_last_3", "fantasy_points"),
    ]:
        values = pd.to_numeric(
            out[source],
            errors="coerce",
        ).fillna(0.0)

        total = values.groupby(
            [out[key] for key in keys]
        ).transform("sum")

        out[f"wr_room_total_prior_{suffix}"] = total
        out[f"wr_room_player_share_{suffix}"] = np.where(
            total.gt(0),
            values / total,
            np.nan,
        )

        temp = pd.DataFrame(
            {
                **{key: out[key] for key in keys},
                "value": values,
            },
            index=out.index,
        )

        def max_other(group: pd.DataFrame) -> pd.Series:
            array = group["value"].to_numpy()
            result = []
            for index in range(len(array)):
                if len(array) <= 1:
                    result.append(0.0)
                    continue
                result.append(
                    float(
                        np.nanmax(
                            np.delete(array, index)
                        )
                    )
                )
            return pd.Series(
                result,
                index=group.index,
            )

        out[f"wr_room_max_other_prior_{suffix}"] = (
            temp.groupby(
                keys,
                group_keys=False,
            )
            .apply(
                max_other,
                include_groups=False,
            )
            .sort_index()
        )

    if "injury_out" in out.columns:
        out["wr_room_out_count"] = (
            out.groupby(keys)["injury_out"]
            .transform("sum")
        )

    return out


def build_defense_history_snapshots() -> pd.DataFrame:
    team = pd.read_csv(TEAM_FILE, low_memory=False)
    team["team"] = normalize_team_code(
        team["team"].astype(str)
    )
    team["opponent_team"] = normalize_team_code(
        team["opponent_team"].astype(str)
    )

    observed = pd.DataFrame(
        {
            "season": team["season"],
            "week": team["week"],
            "defense_team": team["opponent_team"],
            "targets_allowed": col(team, "targets"),
            "receptions_allowed": col(team, "receptions"),
            "rec_yards_allowed": col(team, "receiving_yards"),
            "rec_tds_allowed": col(team, "receiving_tds"),
        }
    )

    observed = (
        observed.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            targets_allowed=("targets_allowed", "sum"),
            receptions_allowed=("receptions_allowed", "sum"),
            rec_yards_allowed=("rec_yards_allowed", "sum"),
            rec_tds_allowed=("rec_tds_allowed", "sum"),
        )
    )

    observed["_week_key"] = week_key(observed)
    observed = observed.sort_values(
        ["defense_team", "_week_key"]
    ).copy()

    metrics = [
        "targets_allowed",
        "receptions_allowed",
        "rec_yards_allowed",
        "rec_tds_allowed",
    ]

    for metric in metrics:
        observed[metric] = pd.to_numeric(
            observed[metric],
            errors="coerce",
        ).fillna(0.0)

        for window in (3, 5):
            observed[
                f"snapshot_opp_avg_{metric}_last_{window}"
            ] = (
                observed.groupby(
                    "defense_team",
                    sort=False,
                )[metric]
                .transform(
                    lambda s: s.rolling(
                        window,
                        min_periods=1,
                    ).mean()
                )
            )

    keep = [
        "defense_team",
        "_week_key",
        *[
            column
            for column in observed.columns
            if column.startswith(
                "snapshot_opp_avg_"
            )
        ],
    ]
    return observed[keep].copy()


def add_prior_defense_history(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    history = build_defense_history_snapshots()
    out = candidates.copy()

    out["_week_key"] = week_key(out)
    history["_week_key"] = pd.to_numeric(
        history["_week_key"],
        errors="raise",
    ).astype("int64")

    out["opponent"] = out["opponent"].astype(str)
    history["defense_team"] = (
        history["defense_team"].astype(str)
    )

    out["_candidate_order_def"] = np.arange(
        len(out)
    )

    left = out.sort_values(
        ["_week_key", "opponent"]
    ).copy()
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
        column: column.replace(
            "snapshot_",
            "",
            1,
        )
        for column in merged.columns
        if column.startswith(
            "snapshot_opp_avg_"
        )
    }
    merged = merged.rename(columns=rename)

    return (
        merged.sort_values(
            "_candidate_order_def"
        )
        .drop(
            columns=[
                "_candidate_order_def",
                "defense_team",
                "_week_key",
            ],
            errors="ignore",
        )
    )


def main() -> None:
    print("Building GridironIQ v2 WR candidate table...")

    candidates = load_wr_depth_candidates()
    candidates = candidates[
        candidates["season"].between(2021, 2026)
    ].copy()

    schedule = build_schedule_team_weeks()
    candidates = candidates.merge(
        schedule,
        how="inner",
        on=["season", "week", "team"],
    )

    candidates = add_names(candidates)
    candidates = add_actual_results(candidates)
    candidates = add_snap_results(candidates)
    candidates = add_prior_player_history(candidates)
    candidates = add_wr_injury_features(candidates)
    candidates = add_wr_room_features(candidates)
    candidates = add_prior_defense_history(candidates)

    historical = candidates[
        candidates["season"].between(2021, 2025)
        & candidates["game_completed"].eq(1)
    ].copy()
    future = candidates[
        candidates["season"].eq(2026)
        & candidates["game_completed"].eq(0)
    ].copy()

    core_columns = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "game_completed",
        "depth_chart_wr_rank",
        "listed_wr1",
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
        "team_wr_candidates",
        "wr_room_total_prior_targets",
        "wr_room_player_share_targets",
        "wr_room_max_other_prior_targets",
        "wr_room_total_prior_target_share",
        "wr_room_player_share_target_share",
        "wr_room_max_other_prior_target_share",
        "wr_room_total_prior_snap_share",
        "wr_room_player_share_snap_share",
        "wr_room_max_other_prior_snap_share",
        "wr_room_total_prior_fantasy_points",
        "wr_room_player_share_fantasy_points",
        "wr_room_max_other_prior_fantasy_points",
        "wr_room_out_count",
    ]

    rolling_shorts = [
        "fp",
        "targets",
        "receptions",
        "rec_yards",
        "rec_tds",
        "opportunities",
        "yards_per_target",
        "catch_rate",
        "target_share",
        "reception_share",
        "receiving_yard_share",
        "carries",
        "rush_yards",
        "rush_tds",
        "air_yards",
        "yac",
        "rec_first_downs",
        "offense_snaps",
        "offense_pct",
        "recorded_activity",
    ]

    rolling_columns = []
    for short in rolling_shorts:
        rolling_columns.extend(
            [
                f"previous_{short}",
                f"avg_{short}_last_3",
                f"avg_{short}_last_5",
            ]
        )

    defense_columns = [
        "opp_avg_targets_allowed_last_3",
        "opp_avg_targets_allowed_last_5",
        "opp_avg_receptions_allowed_last_3",
        "opp_avg_receptions_allowed_last_5",
        "opp_avg_rec_yards_allowed_last_3",
        "opp_avg_rec_yards_allowed_last_5",
        "opp_avg_rec_tds_allowed_last_3",
        "opp_avg_rec_tds_allowed_last_5",
    ]

    outcome_columns = [
        "actual_fantasy_points",
        "recorded_activity",
        "played_any_snap",
        "snap_50_role",
        "snap_65_role",
        "target_5_role",
        "target_share_20_role",
        "snap50_or_target5_role",
        "snap65_or_target5_role",
        "targets",
        "receptions",
        "rec_yards",
        "rec_tds",
        "carries",
        "rush_yards",
        "rush_tds",
        "opportunities",
        "air_yards",
        "yac",
        "rec_first_downs",
        "target_share",
        "reception_share",
        "receiving_yard_share",
        "yards_per_target",
        "catch_rate",
        "offense_snaps",
        "offense_pct",
    ]

    output_columns = [
        *core_columns,
        *rolling_columns,
        *defense_columns,
        *outcome_columns,
    ]

    output_columns = [
        column
        for column in output_columns
        if column in candidates.columns
    ]

    model = candidates[output_columns].copy()
    model = model.sort_values(
        [
            "season",
            "week",
            "team",
            "depth_chart_wr_rank",
            "player_name",
        ]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model.to_csv(OUTPUT_FILE, index=False)

    print(
        f"Historical candidate rows: {len(historical):,}"
    )
    print(
        "50%+ snap role:                         "
        f"{historical['snap_50_role'].mean():.1%}"
    )
    print(
        "65%+ snap role:                         "
        f"{historical['snap_65_role'].mean():.1%}"
    )
    print(
        "5+ target role:                         "
        f"{historical['target_5_role'].mean():.1%}"
    )
    print(
        "20%+ target-share role:                 "
        f"{historical['target_share_20_role'].mean():.1%}"
    )
    print(
        "50% snaps OR 5+ targets:                "
        f"{historical['snap50_or_target5_role'].mean():.1%}"
    )
    print(
        "65% snaps OR 5+ targets:                "
        f"{historical['snap65_or_target5_role'].mean():.1%}"
    )
    print(
        "Played any offensive snap:              "
        f"{historical['played_any_snap'].mean():.1%}"
    )
    print(
        "Zero fantasy-point rows:                 "
        f"{historical['actual_fantasy_points'].eq(0).mean():.1%}"
    )
    print(
        f"2026 future/unplayed candidates:          {len(future):,}"
    )

    defense_cols = [
        "opp_avg_targets_allowed_last_3",
        "opp_avg_rec_yards_allowed_last_3",
    ]
    if len(future):
        coverage = (
            future[defense_cols]
            .notna()
            .all(axis=1)
            .mean()
        )
        print(
            "Future opponent-defense coverage:        "
            f"{coverage:.1%}"
        )

    print(f"Columns: {len(model.columns)}")
    print(f"WRITE {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
