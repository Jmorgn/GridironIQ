"""Build the first GridironIQ tight-end model dataset.

TE v1 deliberately starts with TEs who recorded fantasy-relevant game activity.
That gives us a clean active-game regression benchmark before we model the
harder pregame questions: route participation, target earning, depth-chart
movement, injuries, and multiple fantasy-relevant tight ends on one team.

All rolling player, usage-share, snap, and opponent-defense features use only
PRIOR games.
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
OUTPUT_FILE = PROCESSED_DIR / "te_model_dataset.csv"


def add_te_injury_features(tes: pd.DataFrame) -> pd.DataFrame:
    frames = []

    for season in range(2021, 2027):
        path = RAW_DIR / f"injuries_{season}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))

    out = tes.copy()
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
            injuries["position"].astype(str).eq("TE")
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
        injuries.get(
            "report_status",
            pd.Series("", index=injuries.index),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )
    practice_status = (
        injuries.get(
            "practice_status",
            pd.Series("", index=injuries.index),
        )
        .fillna("")
        .astype(str)
        .str.lower()
    )

    injuries["on_injury_report"] = 1.0
    injuries["injury_questionable"] = report_status.str.contains(
        "questionable",
        regex=False,
    ).astype(float)
    injuries["injury_doubtful"] = report_status.str.contains(
        "doubtful",
        regex=False,
    ).astype(float)
    injuries["injury_out"] = report_status.str.contains(
        r"\bout\b",
        regex=True,
    ).astype(float)

    injuries["practice_dnp"] = (
        practice_status.str.contains("did not", regex=False)
        | practice_status.str.contains("dnp", regex=False)
    ).astype(float)
    injuries["practice_limited"] = practice_status.str.contains(
        "limited",
        regex=False,
    ).astype(float)
    injuries["practice_full"] = practice_status.str.contains(
        "full",
        regex=False,
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


def add_te_depth_features(tes: pd.DataFrame) -> pd.DataFrame:
    """Add pregame TE depth-chart rank across the nflverse schema change."""
    out = tes.copy()
    out["_row_id"] = np.arange(len(out))
    out["depth_chart_te_rank"] = np.nan

    old_frames = []
    for season in range(2021, 2025):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if path.exists():
            old_frames.append(pd.read_csv(path, low_memory=False))

    if old_frames:
        old = pd.concat(old_frames, ignore_index=True)

        if "game_type" in old.columns:
            old = old[
                old["game_type"].astype(str).eq("REG")
            ].copy()

        if "position" in old.columns:
            old = old[
                old["position"].astype(str).eq("TE")
            ].copy()

        old["team_dc"] = normalize_team_code(
            old["club_code"].astype(str)
        )
        old["depth_chart_te_rank_old"] = pd.to_numeric(
            old["depth_team"],
            errors="coerce",
        )

        old_keep = old[
            [
                "gsis_id",
                "season",
                "week",
                "team_dc",
                "depth_chart_te_rank_old",
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
        out.loc[old_mask, "depth_chart_te_rank"] = out.loc[
            old_mask,
            "depth_chart_te_rank_old",
        ]

        out = out.drop(
            columns=[
                "gsis_id",
                "team_dc",
                "depth_chart_te_rank_old",
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
            new = new[
                new["pos_abb"].astype(str).eq("TE")
            ].copy()

        new["team_dc"] = normalize_team_code(
            new["team"].astype(str)
        )
        new["depth_dt"] = pd.to_datetime(
            new["dt"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None)
        new["depth_date"] = new["depth_dt"].dt.normalize()
        new["depth_chart_te_rank_new"] = pd.to_numeric(
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
                    "depth_chart_te_rank_new",
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
                    [
                        "_row_id",
                        "depth_date",
                        "depth_chart_te_rank_new",
                    ],
                    ascending=[True, False, True],
                )
                .drop_duplicates("_row_id", keep="first")
                [["_row_id", "depth_chart_te_rank_new"]]
            )

            rank_map = latest.set_index("_row_id")[
                "depth_chart_te_rank_new"
            ]
            mask = out["season"].ge(2025)
            out.loc[mask, "depth_chart_te_rank"] = (
                out.loc[mask, "_row_id"].map(rank_map)
            )

    out["listed_te1"] = np.where(
        out["depth_chart_te_rank"].notna(),
        out["depth_chart_te_rank"].eq(1).astype(float),
        np.nan,
    )

    return out.drop(columns=["_row_id"])


def add_team_receiving_shares(
    players: pd.DataFrame,
    tes: pd.DataFrame,
) -> pd.DataFrame:
    players = players.copy()
    tes = tes.copy()

    players = pd.concat(
        [
            players,
            pd.DataFrame(
                {
                    "_team_targets": col(players, "targets"),
                    "_team_receptions": col(players, "receptions"),
                    "_team_receiving_yards": col(
                        players,
                        "receiving_yards",
                    ),
                },
                index=players.index,
            ),
        ],
        axis=1,
    )

    team = (
        players.groupby(
            ["season", "week", "team"],
            as_index=False,
        )
        .agg(
            team_targets=("_team_targets", "sum"),
            team_receptions=("_team_receptions", "sum"),
            team_receiving_yards=(
                "_team_receiving_yards",
                "sum",
            ),
        )
    )

    tes = tes.merge(
        team,
        how="left",
        on=["season", "week", "team"],
    )

    tes["target_share"] = np.where(
        tes["team_targets"].gt(0),
        col(tes, "targets") / tes["team_targets"],
        np.nan,
    )
    tes["reception_share"] = np.where(
        tes["team_receptions"].gt(0),
        col(tes, "receptions") / tes["team_receptions"],
        np.nan,
    )
    tes["receiving_yard_share"] = np.where(
        tes["team_receiving_yards"].gt(0),
        col(tes, "receiving_yards")
        / tes["team_receiving_yards"],
        np.nan,
    )

    return tes


def add_te_rolling_features(tes: pd.DataFrame) -> pd.DataFrame:
    out = tes.sort_values(
        ["player_id", "season", "week"]
    ).copy()

    out["opportunities"] = (
        col(out, "targets") + col(out, "carries")
    )
    out["yards_per_target"] = np.where(
        col(out, "targets").gt(0),
        col(out, "receiving_yards") / col(out, "targets"),
        np.nan,
    )
    out["catch_rate"] = np.where(
        col(out, "targets").gt(0),
        col(out, "receptions") / col(out, "targets"),
        np.nan,
    )

    stat_map = {
        "custom_fantasy_points": "fp",
        "targets": "targets",
        "receptions": "receptions",
        "receiving_yards": "rec_yards",
        "receiving_tds": "rec_tds",
        "opportunities": "opportunities",
        "yards_per_target": "yards_per_target",
        "catch_rate": "catch_rate",
        "target_share": "target_share",
        "reception_share": "reception_share",
        "receiving_yard_share": "receiving_yard_share",
        "carries": "carries",
        "rushing_yards": "rush_yards",
        "rushing_tds": "rush_tds",
    }

    optional_map = {
        "receiving_air_yards": "air_yards",
        "receiving_yards_after_catch": "yac",
        "receiving_first_downs": "rec_first_downs",
    }
    for source, short in optional_map.items():
        if source in out.columns:
            stat_map[source] = short

    for source, short in stat_map.items():
        if source not in out.columns:
            out[source] = 0.0

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


def add_prior_snap_features(tes: pd.DataFrame) -> pd.DataFrame:
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    players = pd.read_csv(PLAYERS_FILE, low_memory=False)

    id_map = (
        players[["gsis_id", "pfr_id"]]
        .dropna()
        .drop_duplicates("gsis_id")
    )

    snaps["team"] = normalize_team_code(
        snaps["team"].astype(str)
    )
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

    out = tes.merge(
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
        left_on=[
            "pfr_id",
            "season",
            "week",
            "team",
        ],
        right_on=[
            "pfr_player_id",
            "season",
            "week",
            "team",
        ],
    )

    out = out.sort_values(
        ["player_id", "season", "week"]
    ).copy()

    for source, short in [
        ("offense_snaps", "offense_snaps"),
        ("offense_pct", "offense_pct"),
    ]:
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


def build_te_defense_features(
    team: pd.DataFrame,
) -> pd.DataFrame:
    defense = pd.DataFrame(
        {
            "season": team["season"],
            "week": team["week"],
            "defense_team": normalize_team_code(
                team["opponent_team"].astype(str)
            ),
            "pass_completions_allowed": col(
                team,
                "completions",
            ),
            "pass_yards_allowed": col(
                team,
                "passing_yards",
            ),
            "pass_tds_allowed": col(
                team,
                "passing_tds",
            ),
        }
    )

    defense = (
        defense.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            pass_completions_allowed=(
                "pass_completions_allowed",
                "sum",
            ),
            pass_yards_allowed=(
                "pass_yards_allowed",
                "sum",
            ),
            pass_tds_allowed=(
                "pass_tds_allowed",
                "sum",
            ),
        )
        .sort_values(
            ["defense_team", "season", "week"]
        )
    )

    metrics = [
        "pass_completions_allowed",
        "pass_yards_allowed",
        "pass_tds_allowed",
    ]

    for metric in metrics:
        for window in (3, 5):
            defense[
                f"opp_avg_{metric}_last_{window}"
            ] = (
                defense.groupby(
                    "defense_team",
                    sort=False,
                )[metric]
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



def build_te_specific_defense_features(
    team: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Prior TE production conceded per defense, including zero-target games.

    A defensive team's completed games come from the weekly team table.
    Missing TE stat lines in one of those games mean zero TE production,
    not a missing defensive observation. All opponent metrics are shifted
    before calculating rolling averages.
    """
    te_games = players[
        players["position"].astype(str).eq("TE")
    ].copy()
    te_games["defense_team"] = normalize_team_code(
        te_games["opponent_team"].astype(str)
    )
    te_games["_te_fp"] = custom_fantasy_points(te_games)
    te_games["_te_targets"] = col(te_games, "targets")
    te_games["_te_receptions"] = col(te_games, "receptions")
    te_games["_te_rec_yards"] = col(
        te_games, "receiving_yards"
    )

    totals = (
        te_games.groupby(
            ["season", "week", "defense_team"],
            as_index=False,
        )
        .agg(
            te_targets_allowed=("_te_targets", "sum"),
            te_receptions_allowed=("_te_receptions", "sum"),
            te_rec_yards_allowed=("_te_rec_yards", "sum"),
            te_fp_allowed=("_te_fp", "sum"),
        )
    )

    games = team[
        ["season", "week", "opponent_team"]
    ].copy()
    games["defense_team"] = normalize_team_code(
        games.pop("opponent_team").astype(str)
    )
    games = games.drop_duplicates(
        ["season", "week", "defense_team"]
    )
    defense = games.merge(
        totals,
        how="left",
        on=["season", "week", "defense_team"],
    )

    metrics = [
        "te_targets_allowed",
        "te_receptions_allowed",
        "te_rec_yards_allowed",
        "te_fp_allowed",
    ]
    for metric in metrics:
        defense[metric] = defense[metric].fillna(0.0)

    defense = defense.sort_values(
        ["defense_team", "season", "week"]
    ).copy()

    for metric in metrics:
        for window in (3, 5):
            defense[
                f"opp_avg_{metric}_last_{window}"
            ] = (
                defense.groupby(
                    "defense_team", sort=False
                )[metric]
                .transform(
                    lambda series: series.shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
            )

    return defense[
        [
            "season",
            "week",
            "defense_team",
            *[
                f"opp_avg_{metric}_last_{window}"
                for metric in metrics
                for window in (3, 5)
            ],
        ]
    ]


def main() -> None:
    print("Building GridironIQ TE v1 model dataset...")

    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    team = pd.read_csv(TEAM_FILE, low_memory=False)

    players["team"] = normalize_team_code(
        players["team"].astype(str)
    )

    tes = players[
        players["position"].astype(str).eq("TE")
    ].copy()
    tes["opponent_team"] = normalize_team_code(
        tes["opponent_team"].astype(str)
    )

    return_activity = pd.Series(
        0.0,
        index=tes.index,
    )
    for name in [
        "special_teams_return_yards",
        "kickoff_return_yards",
        "punt_return_yards",
    ]:
        if name in tes.columns:
            return_activity += col(tes, name)

    activity = (
        col(tes, "targets")
        + col(tes, "receptions")
        + col(tes, "carries")
        + return_activity / 20.0
    )
    tes = tes[activity.gt(0)].copy()

    tes["custom_fantasy_points"] = custom_fantasy_points(tes)

    tes = add_home_away(tes)
    tes = add_game_context_features(tes)
    tes = add_te_injury_features(tes)
    tes = add_te_depth_features(tes)
    tes = add_team_receiving_shares(players, tes)
    tes = add_te_rolling_features(tes)
    tes = add_prior_snap_features(tes)

    defense = build_te_defense_features(team)
    tes = tes.merge(
        defense,
        how="left",
        left_on=[
            "season",
            "week",
            "opponent_team",
        ],
        right_on=[
            "season",
            "week",
            "defense_team",
        ],
    )

    # Position-specific defensive trends supplement general pass defense.
    # Both are based on completed games PRIOR to this matchup.
    tes = tes.drop(columns=["defense_team"], errors="ignore")
    te_defense = build_te_specific_defense_features(team, players)
    tes = tes.merge(
        te_defense,
        how="left",
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "defense_team"],
    )

    if "player_display_name" in tes.columns:
        tes["model_player_name"] = (
            tes["player_display_name"]
        )
    else:
        tes["model_player_name"] = tes["player_name"]

    tes = tes.rename(
        columns={
            "opponent_team": "opponent",
            "custom_fantasy_points": "actual_fantasy_points",
        }
    )

    core_columns = [
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
        "depth_chart_te_rank",
        "listed_te1",
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
        "opp_avg_pass_completions_allowed_last_3",
        "opp_avg_pass_completions_allowed_last_5",
        "opp_avg_pass_yards_allowed_last_3",
        "opp_avg_pass_yards_allowed_last_5",
        "opp_avg_pass_tds_allowed_last_3",
        "opp_avg_pass_tds_allowed_last_5",
        "opp_avg_te_targets_allowed_last_3",
        "opp_avg_te_targets_allowed_last_5",
        "opp_avg_te_receptions_allowed_last_3",
        "opp_avg_te_receptions_allowed_last_5",
        "opp_avg_te_rec_yards_allowed_last_3",
        "opp_avg_te_rec_yards_allowed_last_5",
        "opp_avg_te_fp_allowed_last_3",
        "opp_avg_te_fp_allowed_last_5",
    ]

    output_columns = [
        *core_columns,
        *rolling_columns,
        *defense_columns,
        "actual_fantasy_points",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in tes.columns
    ]

    model = tes[output_columns].rename(
        columns={"model_player_name": "player_name"}
    )
    model = model.sort_values(
        ["season", "week", "team", "player_name"]
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model.to_csv(OUTPUT_FILE, index=False)

    historical = model[
        model["season"].between(2021, 2025)
    ]
    live = model[model["season"].eq(2026)]

    print(f"TE active-game rows: {len(model):,}")
    print(f"Columns:             {len(model.columns)}")
    print(f"2021-2025 rows:      {len(historical):,}")
    print(f"2026 live rows:      {len(live):,}")
    print(f"WRITE {OUTPUT_FILE}")

    depth_coverage = (
        historical["depth_chart_te_rank"].notna().mean()
        if "depth_chart_te_rank" in historical.columns
        else 0.0
    )
    snap_coverage = (
        historical["previous_offense_pct"].notna().mean()
        if "previous_offense_pct" in historical.columns
        else 0.0
    )

    print(
        f"Historical depth-chart coverage: {depth_coverage:.1%}"
    )
    print(
        f"Historical prior-snap coverage:  {snap_coverage:.1%}"
    )

    print("\nSample:")
    print(model.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
