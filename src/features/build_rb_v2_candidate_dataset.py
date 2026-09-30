"""Build the GridironIQ v2 pregame RB candidate dataset.

RB v2 starts from pregame depth charts rather than postgame participation.
That lets the model represent committees, inactive backs, injury replacements,
and multiple fantasy-relevant RBs on the same team.

Candidate outcomes retained for model experiments:
- actual_fantasy_points
- played_any_snap
- snap_35_role: >=35% offensive snaps
- opp_10_role: >=10 carries + targets
- meaningful_workload: snap_35_role OR opp_10_role

All rolling features are shifted so the current game's outcome is never used
to predict itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from audit_rb_candidates import load_rb_depth_candidates
from build_qb_model_dataset import (
    col,
    custom_fantasy_points,
    normalize_team_code,
)
from build_qb_v2_candidate_dataset import build_schedule_team_weeks
from build_rb_model_dataset import add_rb_injury_features

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED_DIR / "team_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_FILE = RAW_DIR / "players.csv"
OUTPUT_FILE = PROCESSED_DIR / "rb_v2_candidate_dataset.csv"


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

    # Team opportunity denominators come from all offensive players.
    players["team_carries_component"] = col(players, "carries")
    players["team_targets_component"] = col(players, "targets")
    team_totals = (
        players.groupby(
            ["season", "week", "team"],
            as_index=False,
        )
        .agg(
            team_carries=("team_carries_component", "sum"),
            team_targets=("team_targets_component", "sum"),
        )
    )

    rbs = players[
        players["position"].astype(str).eq("RB")
    ].copy()

    numeric_map = {
        "carries": "carries",
        "targets": "targets",
        "receptions": "receptions",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
    }

    for source, destination in numeric_map.items():
        rbs[destination] = col(rbs, source)

    rbs["opportunities"] = rbs["carries"] + rbs["targets"]
    rbs["touches"] = rbs["carries"] + rbs["receptions"]
    rbs["actual_fantasy_points"] = custom_fantasy_points(rbs)

    actual = (
        rbs.groupby(
            ["player_id", "season", "week", "team"],
            as_index=False,
        )
        .agg(
            carries=("carries", "sum"),
            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            opportunities=("opportunities", "sum"),
            touches=("touches", "sum"),
            rush_yards=("rush_yards", "sum"),
            rush_tds=("rush_tds", "sum"),
            rec_yards=("rec_yards", "sum"),
            rec_tds=("rec_tds", "sum"),
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
        "carries",
        "targets",
        "receptions",
        "opportunities",
        "touches",
        "rush_yards",
        "rush_tds",
        "rec_yards",
        "rec_tds",
        "actual_fantasy_points",
        "had_stat_row",
    ]

    for column in outcome_columns:
        out.loc[completed, column] = (
            out.loc[completed, column].fillna(0.0)
        )
        out.loc[~completed, column] = np.nan

    # Team totals are known only for completed games in the weekly stats feed.
    out.loc[~completed, ["team_carries", "team_targets"]] = np.nan

    out["carry_share"] = np.where(
        out["team_carries"].gt(0),
        out["carries"] / out["team_carries"],
        np.nan,
    )
    out["target_share"] = np.where(
        out["team_targets"].gt(0),
        out["targets"] / out["team_targets"],
        np.nan,
    )

    team_opportunities = (
        out["team_carries"] + out["team_targets"]
    )
    out["opportunity_share"] = np.where(
        team_opportunities.gt(0),
        out["opportunities"] / team_opportunities,
        np.nan,
    )

    out["recorded_activity"] = np.nan
    out.loc[completed, "recorded_activity"] = (
        out.loc[completed, "opportunities"].gt(0)
        | out.loc[completed, "receptions"].gt(0)
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

    out["snap_35_role"] = np.nan
    out["opp_10_role"] = np.nan
    out["meaningful_workload"] = np.nan

    out.loc[completed, "snap_35_role"] = (
        out.loc[completed, "offense_pct"].ge(0.35).astype(int)
    )
    out.loc[completed, "opp_10_role"] = (
        out.loc[completed, "opportunities"].ge(10).astype(int)
    )
    out.loc[completed, "meaningful_workload"] = (
        out.loc[completed, "snap_35_role"].eq(1)
        | out.loc[completed, "opp_10_role"].eq(1)
    ).astype(int)

    return out


def add_prior_player_history(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.sort_values(
        ["player_id", "season", "week"]
    ).copy()

    metrics = {
        "actual_fantasy_points": "fp",
        "carries": "carries",
        "targets": "targets",
        "receptions": "receptions",
        "opportunities": "opportunities",
        "touches": "touches",
        "rush_yards": "rush_yards",
        "rush_tds": "rush_tds",
        "rec_yards": "rec_yards",
        "rec_tds": "rec_tds",
        "carry_share": "carry_share",
        "target_share": "target_share",
        "opportunity_share": "opportunity_share",
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


def add_rb_room_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Describe the player's pregame competition inside the RB room."""
    out = candidates.copy()
    keys = ["season", "week", "team"]

    out["team_rb_candidates"] = (
        out.groupby(keys)["player_id"].transform("size")
    )

    for source, suffix in [
        ("avg_opportunities_last_3", "opportunities"),
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

        out[f"rb_room_total_prior_{suffix}"] = total
        out[f"rb_room_player_share_{suffix}"] = np.where(
            total.gt(0),
            values / total,
            np.nan,
        )

        # Highest OTHER teammate value. This directly expresses committee
        # pressure without forcing exactly one or two relevant backs.
        temp = pd.DataFrame(
            {
                **{key: out[key] for key in keys},
                "player_id": out["player_id"],
                "value": values,
            },
            index=out.index,
        )

        def max_other(group: pd.DataFrame) -> pd.Series:
            array = group["value"].to_numpy()
            result = []
            for idx, value in enumerate(array):
                if len(array) <= 1:
                    result.append(0.0)
                    continue
                others = np.delete(array, idx)
                result.append(float(np.nanmax(others)))
            return pd.Series(result, index=group.index)

        out[f"rb_room_max_other_prior_{suffix}"] = (
            temp.groupby(keys, group_keys=False)
            .apply(max_other, include_groups=False)
            .sort_index()
        )

    if "injury_out" in out.columns:
        out["rb_room_out_count"] = (
            out.groupby(keys)["injury_out"].transform("sum")
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
            "rush_yards_allowed": col(team, "rushing_yards"),
            "rush_tds_allowed": col(team, "rushing_tds"),
            "pass_yards_allowed": col(team, "passing_yards"),
            "pass_tds_allowed": col(team, "passing_tds"),
        }
    )

    observed = (
        observed.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            rush_yards_allowed=("rush_yards_allowed", "sum"),
            rush_tds_allowed=("rush_tds_allowed", "sum"),
            pass_yards_allowed=("pass_yards_allowed", "sum"),
            pass_tds_allowed=("pass_tds_allowed", "sum"),
        )
    )

    observed["_week_key"] = week_key(observed)
    observed = observed.sort_values(
        ["defense_team", "_week_key"]
    ).copy()

    metrics = [
        "rush_yards_allowed",
        "rush_tds_allowed",
        "pass_yards_allowed",
        "pass_tds_allowed",
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
            if column.startswith("snapshot_opp_avg_")
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

    out["_candidate_order_def"] = np.arange(len(out))

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
                "_week_key",
            ],
            errors="ignore",
        )
    )


def main() -> None:
    print("Building GridironIQ v2 RB candidate table...")

    candidates = load_rb_depth_candidates()
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
    candidates = add_rb_injury_features(candidates)
    candidates = add_rb_room_features(candidates)
    candidates = add_prior_defense_history(candidates)

    historical = candidates[
        candidates["season"].between(2021, 2025)
        & candidates["game_completed"].eq(1)
    ].copy()
    future = candidates[
        candidates["season"].eq(2026)
        & candidates["game_completed"].eq(0)
    ].copy()

    output_columns = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
        "game_completed",
        "depth_chart_rb_rank",
        "listed_rb1",
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
        "team_rb_candidates",
        "rb_room_total_prior_opportunities",
        "rb_room_player_share_opportunities",
        "rb_room_max_other_prior_opportunities",
        "rb_room_total_prior_snap_share",
        "rb_room_player_share_snap_share",
        "rb_room_max_other_prior_snap_share",
        "rb_room_total_prior_fantasy_points",
        "rb_room_player_share_fantasy_points",
        "rb_room_max_other_prior_fantasy_points",
        "rb_room_out_count",
        "previous_fp",
        "avg_fp_last_3",
        "avg_fp_last_5",
        "previous_carries",
        "avg_carries_last_3",
        "avg_carries_last_5",
        "previous_targets",
        "avg_targets_last_3",
        "avg_targets_last_5",
        "previous_receptions",
        "avg_receptions_last_3",
        "avg_receptions_last_5",
        "previous_opportunities",
        "avg_opportunities_last_3",
        "avg_opportunities_last_5",
        "previous_touches",
        "avg_touches_last_3",
        "avg_touches_last_5",
        "previous_rush_yards",
        "avg_rush_yards_last_3",
        "avg_rush_yards_last_5",
        "previous_rush_tds",
        "avg_rush_tds_last_3",
        "avg_rush_tds_last_5",
        "previous_rec_yards",
        "avg_rec_yards_last_3",
        "avg_rec_yards_last_5",
        "previous_rec_tds",
        "avg_rec_tds_last_3",
        "avg_rec_tds_last_5",
        "previous_carry_share",
        "avg_carry_share_last_3",
        "avg_carry_share_last_5",
        "previous_target_share",
        "avg_target_share_last_3",
        "avg_target_share_last_5",
        "previous_opportunity_share",
        "avg_opportunity_share_last_3",
        "avg_opportunity_share_last_5",
        "previous_offense_snaps",
        "avg_offense_snaps_last_3",
        "avg_offense_snaps_last_5",
        "previous_offense_pct",
        "avg_offense_pct_last_3",
        "avg_offense_pct_last_5",
        "previous_recorded_activity",
        "avg_recorded_activity_last_3",
        "avg_recorded_activity_last_5",
        "opp_avg_rush_yards_allowed_last_3",
        "opp_avg_rush_yards_allowed_last_5",
        "opp_avg_rush_tds_allowed_last_3",
        "opp_avg_rush_tds_allowed_last_5",
        "opp_avg_pass_yards_allowed_last_3",
        "opp_avg_pass_yards_allowed_last_5",
        "opp_avg_pass_tds_allowed_last_3",
        "opp_avg_pass_tds_allowed_last_5",
        "actual_fantasy_points",
        "recorded_activity",
        "played_any_snap",
        "snap_35_role",
        "opp_10_role",
        "meaningful_workload",
        "carries",
        "targets",
        "receptions",
        "opportunities",
        "touches",
        "offense_snaps",
        "offense_pct",
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
            "depth_chart_rb_rank",
            "player_name",
        ]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(OUTPUT_FILE, index=False)

    print(f"Historical candidate rows: {len(historical):,}")
    print(
        "Meaningful workload (35% snaps OR 10+ opp): "
        f"{historical['meaningful_workload'].mean():.1%}"
    )
    print(
        "35%+ snap role:                         "
        f"{historical['snap_35_role'].mean():.1%}"
    )
    print(
        "10+ opportunity role:                  "
        f"{historical['opp_10_role'].mean():.1%}"
    )
    print(
        "Played any offensive snap:             "
        f"{historical['played_any_snap'].mean():.1%}"
    )
    print(
        "Zero fantasy-point rows:                "
        f"{historical['actual_fantasy_points'].eq(0).mean():.1%}"
    )
    print(
        f"2026 future/unplayed candidates:         {len(future):,}"
    )

    defense_cols = [
        "opp_avg_rush_yards_allowed_last_3",
        "opp_avg_rush_tds_allowed_last_3",
    ]
    if len(future):
        coverage = future[defense_cols].notna().all(axis=1).mean()
        print(
            "Future opponent-defense coverage:       "
            f"{coverage:.1%}"
        )

    print(f"Columns: {len(model.columns)}")
    print(f"WRITE {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
