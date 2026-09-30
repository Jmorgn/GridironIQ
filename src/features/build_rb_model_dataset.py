"""Build the first GridironIQ running-back model dataset.

RB v1 deliberately starts with RBs who recorded fantasy-relevant game activity.
That gives us an apples-to-apples regression benchmark before we add the harder
pregame candidate/workload model for committees and inactive backs.

Features are leakage-safe: player workload/form, snap share, opportunity share,
and opponent-defense rolling statistics use only PRIOR games.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_qb_model_dataset import (
    add_game_context_features,
    add_home_away,
    col,
    custom_fantasy_points,
    normalize_team_code,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED_DIR / "team_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_FILE = RAW_DIR / "players.csv"
OUTPUT_FILE = PROCESSED_DIR / "rb_model_dataset.csv"


def add_rb_injury_features(rbs: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for season in range(2021, 2027):
        path = RAW_DIR / f"injuries_{season}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))

    out = rbs.copy()
    injury_columns = [
        "on_injury_report",
        "injury_questionable",
        "injury_doubtful",
        "injury_out",
        "practice_dnp",
        "practice_limited",
        "practice_full",
        "injury_status_score",
    ]

    if not frames:
        for column in injury_columns:
            out[column] = np.nan
        return out

    injuries = pd.concat(frames, ignore_index=True)

    if "season_type" in injuries.columns:
        injuries = injuries[
            injuries["season_type"].astype(str).eq("REG")
        ].copy()

    if "position" in injuries.columns:
        injuries = injuries[
            injuries["position"].astype(str).eq("RB")
        ].copy()

    injuries["team_injury"] = normalize_team_code(
        injuries["team"].astype(str)
    )
    injuries["date_modified_parsed"] = pd.to_datetime(
        injuries.get("date_modified"),
        errors="coerce",
        utc=True,
    )

    injuries = (
        injuries.sort_values("date_modified_parsed")
        .drop_duplicates(
            ["gsis_id", "season", "week", "team_injury"],
            keep="last",
        )
        .copy()
    )

    report_status = (
        injuries.get("report_status", pd.Series("", index=injuries.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    practice_status = (
        injuries.get("practice_status", pd.Series("", index=injuries.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )

    injuries["on_injury_report"] = 1.0
    injuries["injury_questionable"] = report_status.str.contains(
        "questionable", regex=False
    ).astype(float)
    injuries["injury_doubtful"] = report_status.str.contains(
        "doubtful", regex=False
    ).astype(float)
    injuries["injury_out"] = report_status.str.contains(
        r"\bout\b", regex=True
    ).astype(float)

    injuries["practice_dnp"] = (
        practice_status.str.contains("did not", regex=False)
        | practice_status.str.contains("dnp", regex=False)
    ).astype(float)
    injuries["practice_limited"] = practice_status.str.contains(
        "limited", regex=False
    ).astype(float)
    injuries["practice_full"] = practice_status.str.contains(
        "full", regex=False
    ).astype(float)

    injuries["injury_status_score"] = (
        injuries["injury_questionable"]
        + injuries["injury_doubtful"] * 2
        + injuries["injury_out"] * 3
    )

    keep = [
        "gsis_id",
        "season",
        "week",
        "team_injury",
        *injury_columns,
    ]

    out = out.merge(
        injuries[keep],
        how="left",
        left_on=["player_id", "season", "week", "team"],
        right_on=["gsis_id", "season", "week", "team_injury"],
    )

    for column in injury_columns:
        out[column] = out[column].fillna(0.0)

    return out.drop(
        columns=["gsis_id", "team_injury"],
        errors="ignore",
    )


def add_rb_depth_features(rbs: pd.DataFrame) -> pd.DataFrame:
    """Add pregame RB depth-chart rank across the nflverse schema change."""
    out = rbs.copy()
    out["_row_id"] = np.arange(len(out))
    out["depth_chart_rb_rank"] = np.nan

    old_frames = []
    for season in range(2021, 2025):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if path.exists():
            old_frames.append(pd.read_csv(path, low_memory=False))

    if old_frames:
        old = pd.concat(old_frames, ignore_index=True)

        if "game_type" in old.columns:
            old = old[old["game_type"].astype(str).eq("REG")].copy()

        if "position" in old.columns:
            old = old[old["position"].astype(str).eq("RB")].copy()

        old["team_dc"] = normalize_team_code(old["club_code"].astype(str))
        old["depth_chart_rb_rank_old"] = pd.to_numeric(
            old["depth_team"],
            errors="coerce",
        )

        old_keep = old[
            [
                "gsis_id",
                "season",
                "week",
                "team_dc",
                "depth_chart_rb_rank_old",
            ]
        ].drop_duplicates(
            ["gsis_id", "season", "week", "team_dc"],
            keep="first",
        )

        out = out.merge(
            old_keep,
            how="left",
            left_on=["player_id", "season", "week", "team"],
            right_on=["gsis_id", "season", "week", "team_dc"],
        )

        old_mask = out["season"].le(2024)
        out.loc[old_mask, "depth_chart_rb_rank"] = out.loc[
            old_mask,
            "depth_chart_rb_rank_old",
        ]

        out = out.drop(
            columns=[
                "gsis_id",
                "team_dc",
                "depth_chart_rb_rank_old",
            ],
            errors="ignore",
        )

    new_frames = []
    for season in range(2025, 2027):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if path.exists():
            new_frames.append(pd.read_csv(path, low_memory=False))

    if new_frames and "gameday" in out.columns:
        new = pd.concat(new_frames, ignore_index=True)

        if "pos_abb" in new.columns:
            new = new[new["pos_abb"].astype(str).eq("RB")].copy()

        new["team_dc"] = normalize_team_code(new["team"].astype(str))
        new["depth_dt"] = pd.to_datetime(
            new["dt"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
        new["depth_date"] = new["depth_dt"].dt.normalize()
        new["depth_chart_rb_rank_new"] = pd.to_numeric(
            new["pos_rank"],
            errors="coerce",
        )

        gameside = out[out["season"].ge(2025)][
            ["_row_id", "player_id", "team", "gameday"]
        ].copy()
        gameside["game_date"] = pd.to_datetime(
            gameside["gameday"],
            errors="coerce",
        ).dt.normalize()

        matches = gameside.merge(
            new[
                [
                    "gsis_id",
                    "team_dc",
                    "depth_date",
                    "depth_chart_rb_rank_new",
                ]
            ],
            how="left",
            left_on=["player_id", "team"],
            right_on=["gsis_id", "team_dc"],
        )

        valid = matches[
            matches["depth_date"].notna()
            & matches["game_date"].notna()
            & (matches["depth_date"] <= matches["game_date"])
            & (
                matches["depth_date"]
                >= matches["game_date"] - pd.Timedelta(days=5)
            )
        ].copy()

        if not valid.empty:
            latest = (
                valid.sort_values(
                    ["_row_id", "depth_date", "depth_chart_rb_rank_new"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("_row_id", keep="first")
                [["_row_id", "depth_chart_rb_rank_new"]]
            )

            rank_map = latest.set_index("_row_id")[
                "depth_chart_rb_rank_new"
            ]
            mask = out["season"].ge(2025)
            out.loc[mask, "depth_chart_rb_rank"] = (
                out.loc[mask, "_row_id"].map(rank_map)
            )

    out["listed_rb1"] = np.where(
        out["depth_chart_rb_rank"].notna(),
        out["depth_chart_rb_rank"].eq(1).astype(float),
        np.nan,
    )

    return out.drop(columns=["_row_id"])


def add_team_opportunity_shares(
    players: pd.DataFrame,
    rbs: pd.DataFrame,
) -> pd.DataFrame:
    players = players.copy()
    rbs = rbs.copy()

    players["team_carries"] = col(players, "carries")
    players["team_targets"] = col(players, "targets")

    team = (
        players.groupby(["season", "week", "team"], as_index=False)
        .agg(
            team_carries=("team_carries", "sum"),
            team_targets=("team_targets", "sum"),
        )
    )

    rbs = rbs.merge(
        team,
        how="left",
        on=["season", "week", "team"],
    )

    rbs["carry_share"] = np.where(
        rbs["team_carries"].gt(0),
        col(rbs, "carries") / rbs["team_carries"],
        np.nan,
    )
    rbs["target_share"] = np.where(
        rbs["team_targets"].gt(0),
        col(rbs, "targets") / rbs["team_targets"],
        np.nan,
    )

    team_opportunities = rbs["team_carries"] + rbs["team_targets"]
    rbs["opportunity_share"] = np.where(
        team_opportunities.gt(0),
        (col(rbs, "carries") + col(rbs, "targets"))
        / team_opportunities,
        np.nan,
    )

    return rbs


def add_rb_rolling_features(rbs: pd.DataFrame) -> pd.DataFrame:
    out = rbs.sort_values(["player_id", "season", "week"]).copy()

    out["opportunities"] = col(out, "carries") + col(out, "targets")
    out["touches"] = col(out, "carries") + col(out, "receptions")

    stat_map = {
        "custom_fantasy_points": "fp",
        "carries": "carries",
        "targets": "targets",
        "receptions": "receptions",
        "opportunities": "opportunities",
        "touches": "touches",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "carry_share": "carry_share",
        "target_share": "target_share",
        "opportunity_share": "opportunity_share",
    }

    for source, short in stat_map.items():
        if source not in out.columns:
            out[source] = 0.0

        out[source] = pd.to_numeric(out[source], errors="coerce")

        out[f"previous_{short}"] = (
            out.groupby("player_id", sort=False)[source].shift(1)
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


def add_prior_snap_features(rbs: pd.DataFrame) -> pd.DataFrame:
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    players = pd.read_csv(PLAYERS_FILE, low_memory=False)

    id_map = (
        players[["gsis_id", "pfr_id"]]
        .dropna()
        .drop_duplicates("gsis_id")
    )

    snaps["team"] = normalize_team_code(snaps["team"].astype(str))
    snaps["offense_snaps"] = pd.to_numeric(
        snaps["offense_snaps"],
        errors="coerce",
    )
    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"],
        errors="coerce",
    )
    snaps["offense_pct"] = snaps["offense_pct"].where(
        snaps["offense_pct"].le(1.0),
        snaps["offense_pct"] / 100.0,
    )

    out = rbs.merge(
        id_map,
        how="left",
        left_on="player_id",
        right_on="gsis_id",
    )

    snap_week = snaps[
        [
            "pfr_player_id",
            "season",
            "week",
            "team",
            "offense_snaps",
            "offense_pct",
        ]
    ].drop_duplicates(
        ["pfr_player_id", "season", "week", "team"]
    )

    out = out.merge(
        snap_week,
        how="left",
        left_on=["pfr_id", "season", "week", "team"],
        right_on=["pfr_player_id", "season", "week", "team"],
    )

    out = out.sort_values(["player_id", "season", "week"]).copy()

    for source, short in [
        ("offense_snaps", "offense_snaps"),
        ("offense_pct", "offense_pct"),
    ]:
        out[f"previous_{short}"] = (
            out.groupby("player_id", sort=False)[source].shift(1)
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

    return out.drop(
        columns=[
            "gsis_id",
            "pfr_id",
            "pfr_player_id",
            "offense_snaps",
            "offense_pct",
        ],
        errors="ignore",
    )


def build_rb_defense_features(team: pd.DataFrame) -> pd.DataFrame:
    defense = pd.DataFrame(
        {
            "season": team["season"],
            "week": team["week"],
            "defense_team": normalize_team_code(
                team["opponent_team"].astype(str)
            ),
            "rush_yards_allowed": col(team, "rushing_yards"),
            "rush_tds_allowed": col(team, "rushing_tds"),
            "pass_yards_allowed": col(team, "passing_yards"),
            "pass_tds_allowed": col(team, "passing_tds"),
        }
    )

    defense = (
        defense.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            rush_yards_allowed=("rush_yards_allowed", "sum"),
            rush_tds_allowed=("rush_tds_allowed", "sum"),
            pass_yards_allowed=("pass_yards_allowed", "sum"),
            pass_tds_allowed=("pass_tds_allowed", "sum"),
        )
        .sort_values(["defense_team", "season", "week"])
    )

    metrics = [
        "rush_yards_allowed",
        "rush_tds_allowed",
        "pass_yards_allowed",
        "pass_tds_allowed",
    ]

    for metric in metrics:
        for window in (3, 5):
            defense[f"opp_avg_{metric}_last_{window}"] = (
                defense.groupby("defense_team", sort=False)[metric]
                .transform(
                    lambda s: s.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

    keep = [
        "season",
        "week",
        "defense_team",
        *[
            f"opp_avg_{metric}_last_{window}"
            for metric in metrics
            for window in (3, 5)
        ],
    ]
    return defense[keep]


def main() -> None:
    print("Building GridironIQ RB v1 model dataset...")

    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    team = pd.read_csv(TEAM_FILE, low_memory=False)

    players["team"] = normalize_team_code(players["team"].astype(str))

    rbs = players[
        players["position"].astype(str).eq("RB")
    ].copy()
    rbs["opponent_team"] = normalize_team_code(
        rbs["opponent_team"].astype(str)
    )

    return_activity = pd.Series(0.0, index=rbs.index)
    for name in [
        "special_teams_return_yards",
        "kickoff_return_yards",
        "punt_return_yards",
    ]:
        if name in rbs.columns:
            return_activity += col(rbs, name)

    activity = (
        col(rbs, "carries")
        + col(rbs, "targets")
        + col(rbs, "receptions")
        + return_activity / 20.0
    )
    rbs = rbs[activity.gt(0)].copy()

    rbs["custom_fantasy_points"] = custom_fantasy_points(rbs)

    rbs = add_home_away(rbs)
    rbs = add_game_context_features(rbs)
    rbs = add_rb_injury_features(rbs)
    rbs = add_rb_depth_features(rbs)
    rbs = add_team_opportunity_shares(players, rbs)
    rbs = add_rb_rolling_features(rbs)
    rbs = add_prior_snap_features(rbs)

    defense = build_rb_defense_features(team)
    rbs = rbs.merge(
        defense,
        how="left",
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "defense_team"],
    )

    if "player_display_name" in rbs.columns:
        rbs["model_player_name"] = rbs["player_display_name"]
    else:
        rbs["model_player_name"] = rbs["player_name"]

    rbs = rbs.rename(
        columns={
            "opponent_team": "opponent",
            "custom_fantasy_points": "actual_fantasy_points",
        }
    )

    output_columns = [
        "player_id",
        "model_player_name",
        "season",
        "week",
        "team",
        "opponent",
        "home_away",
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
        "depth_chart_rb_rank",
        "listed_rb1",
        "on_injury_report",
        "injury_questionable",
        "injury_doubtful",
        "injury_out",
        "practice_dnp",
        "practice_limited",
        "practice_full",
        "injury_status_score",
        "previous_offense_snaps",
        "avg_offense_snaps_last_3",
        "avg_offense_snaps_last_5",
        "previous_offense_pct",
        "avg_offense_pct_last_3",
        "avg_offense_pct_last_5",
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
        "opp_avg_rush_yards_allowed_last_3",
        "opp_avg_rush_yards_allowed_last_5",
        "opp_avg_rush_tds_allowed_last_3",
        "opp_avg_rush_tds_allowed_last_5",
        "opp_avg_pass_yards_allowed_last_3",
        "opp_avg_pass_yards_allowed_last_5",
        "opp_avg_pass_tds_allowed_last_3",
        "opp_avg_pass_tds_allowed_last_5",
        "actual_fantasy_points",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in rbs.columns
    ]

    model = rbs[output_columns].rename(
        columns={"model_player_name": "player_name"}
    )
    model = model.sort_values(
        ["season", "week", "team", "player_name"]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(OUTPUT_FILE, index=False)

    print(f"RB active-game rows: {len(model):,}")
    print(f"Columns:             {len(model.columns)}")
    print(f"2021-2025 rows:      {len(model[model['season'].between(2021, 2025)]):,}")
    print(f"2026 live rows:      {len(model[model['season'].eq(2026)]):,}")
    print(f"WRITE {OUTPUT_FILE}")

    print("\nSample:")
    print(model.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
