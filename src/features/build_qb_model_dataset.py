"""Build the first machine-learning table for quarterback fantasy prediction.

This script:
1. Reads the combined player and team weekly tables.
2. Calculates GridironIQ custom fantasy points for QBs.
3. Builds rolling player features using only PRIOR games.
4. Builds rolling opponent-defense features using only PRIOR games.
5. Writes data/processed/qb_model_dataset.csv

Important:
The source weekly CSV tells us how many 40+ yard plays occurred, but it does not
tell us whether each 40+ yard play was also a touchdown. Because of that, the
league's separate +1 "40+ yard TD" bonuses are NOT included yet. Everything else
used here is reconstructible from the weekly aggregate data.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED_DIR / "team_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_FILE = ROOT / "data" / "raw" / "players.csv"
SCHEDULE_FILE = ROOT / "data" / "raw" / "games.csv"
OUTPUT_FILE = PROCESSED_DIR / "qb_model_dataset.csv"


def col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column, or zeros if the column is unavailable."""
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def threshold_bonus(values: pd.Series, thresholds: list[tuple[float, float]]) -> pd.Series:
    """Apply cumulative threshold bonuses such as +3 at 300 and +1 at 400."""
    bonus = pd.Series(0.0, index=values.index)
    for threshold, points in thresholds:
        bonus += np.where(values >= threshold, points, 0.0)
    return bonus


def custom_fantasy_points(df: pd.DataFrame) -> pd.Series:
    """Calculate the reconstructible portion of the user's custom Yahoo scoring."""

    pass_yards = col(df, "passing_yards")
    rush_yards = col(df, "rushing_yards")
    rec_yards = col(df, "receiving_yards")

    points = (
        col(df, "completions") * 0.5
        + pass_yards * 0.04
        + threshold_bonus(pass_yards, [(300, 3), (400, 1), (500, 1)])
        + col(df, "passing_tds") * 4
        - col(df, "passing_interceptions") * 2
        + col(df, "passing_40") * 1
        + rush_yards * 0.10
        + threshold_bonus(rush_yards, [(100, 3), (150, 1), (200, 1)])
        + col(df, "rushing_tds") * 6
        + col(df, "rushing_40") * 2
        + col(df, "receptions") * 1
        + rec_yards * 0.10
        + threshold_bonus(rec_yards, [(100, 3), (150, 1), (200, 1)])
        + col(df, "receiving_tds") * 6
        + col(df, "receiving_40") * 2
        + col(df, "passing_2pt_conversions") * 2
        + col(df, "rushing_2pt_conversions") * 2
        + col(df, "receiving_2pt_conversions") * 2
        - col(df, "fumbles_lost_total") * 2
    )

    # Return-yard scoring exists in the league. nflverse column names can vary
    # slightly by dataset version, so add whichever standard fields are present.
    return_yard_cols = [
        "special_teams_return_yards",
        "kickoff_return_yards",
        "punt_return_yards",
    ]
    seen_return_cols = [name for name in return_yard_cols if name in df.columns]
    if seen_return_cols:
        total_return_yards = sum((col(df, name) for name in seen_return_cols), start=pd.Series(0.0, index=df.index))
        points += total_return_yards * 0.05

    return_td_cols = [
        "special_teams_tds",
        "kickoff_return_tds",
        "punt_return_tds",
    ]
    seen_td_cols = [name for name in return_td_cols if name in df.columns]
    if seen_td_cols:
        total_return_tds = sum((col(df, name) for name in seen_td_cols), start=pd.Series(0.0, index=df.index))
        points += total_return_tds * 6

    return points.astype(float)


def add_player_rolling_features(qbs: pd.DataFrame) -> pd.DataFrame:
    qbs = qbs.sort_values(["player_id", "season", "week"]).copy()

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

    # Group by player only so Week 1 can use the player's final games from the
    # previous season. shift(1) still guarantees the current game is excluded.
    group_keys = ["player_id"]

    for source, short_name in stat_map.items():
        if source not in qbs.columns:
            qbs[source] = 0.0

        qbs[f"previous_{short_name}"] = (
            qbs.groupby(group_keys, sort=False)[source].shift(1)
        )

        for window in (3, 5):
            qbs[f"avg_{short_name}_last_{window}"] = (
                qbs.groupby(group_keys, sort=False)[source]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    return qbs


def find_sacks_allowed_column(df: pd.DataFrame) -> str | None:
    """Find the team-offense column representing sacks taken."""
    candidates = [
        "sacks_suffered",
        "sacks",
        "passing_sacks",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def build_defense_features(team: pd.DataFrame) -> pd.DataFrame:
    """Convert team offense rows into opponent defense-allowed rows."""

    team = team.copy()
    sacks_col = find_sacks_allowed_column(team)

    defense = pd.DataFrame(
        {
            "season": team["season"],
            "week": team["week"],
            "defense_team": team["opponent_team"],
            "pass_yards_allowed": col(team, "passing_yards"),
            "pass_tds_allowed": col(team, "passing_tds"),
            # An offensive interception is an interception made by the defense.
            "def_interceptions": col(team, "passing_interceptions"),
            "def_sacks": col(team, sacks_col) if sacks_col else 0.0,
        }
    )

    # One offensive row should map to one opponent defense row per game.
    defense = (
        defense.groupby(["season", "week", "defense_team"], as_index=False)
        .agg(
            pass_yards_allowed=("pass_yards_allowed", "sum"),
            pass_tds_allowed=("pass_tds_allowed", "sum"),
            def_interceptions=("def_interceptions", "sum"),
            def_sacks=("def_sacks", "sum"),
        )
        .sort_values(["defense_team", "season", "week"])
    )

    metrics = [
        "pass_yards_allowed",
        "pass_tds_allowed",
        "def_interceptions",
        "def_sacks",
    ]

    for metric in metrics:
        for window in (3, 5):
            defense[f"opp_avg_{metric}_last_{window}"] = (
                defense.groupby(["defense_team"], sort=False)[metric]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    keep = ["season", "week", "defense_team"] + [
        f"opp_avg_{metric}_last_{window}"
        for metric in metrics
        for window in (3, 5)
    ]
    return defense[keep]


def add_home_away(qbs: pd.DataFrame) -> pd.DataFrame:
    """Derive home/away from nflverse game_id (season_week_away_home)."""
    qbs = qbs.copy()
    home_team = qbs["game_id"].astype(str).str.split("_").str[-1]
    qbs["home_away"] = np.where(qbs["team"].eq(home_team), "home", "away")
    return qbs


def add_game_context_features(qbs: pd.DataFrame) -> pd.DataFrame:
    """Add pregame schedule, rest, environment, and market context."""
    games = pd.read_csv(SCHEDULE_FILE, low_memory=False)

    if "game_type" in games.columns:
        games = games[games["game_type"].eq("REG")].copy()

    wanted = [
        "game_id",
        "gameday",
        "home_team",
        "away_team",
        "location",
        "away_rest",
        "home_rest",
        "spread_line",
        "total_line",
        "roof",
        "surface",
        "temp",
        "wind",
    ]
    wanted = [name for name in wanted if name in games.columns]
    games = games[wanted].drop_duplicates("game_id")

    qbs = qbs.merge(games, how="left", on="game_id")
    is_home = qbs["home_away"].eq("home")

    if {"home_rest", "away_rest"}.issubset(qbs.columns):
        qbs["team_rest"] = np.where(
            is_home,
            pd.to_numeric(qbs["home_rest"], errors="coerce"),
            pd.to_numeric(qbs["away_rest"], errors="coerce"),
        )
        qbs["opponent_rest"] = np.where(
            is_home,
            pd.to_numeric(qbs["away_rest"], errors="coerce"),
            pd.to_numeric(qbs["home_rest"], errors="coerce"),
        )
        qbs["rest_advantage"] = qbs["team_rest"] - qbs["opponent_rest"]

    if "spread_line" in qbs.columns:
        spread = pd.to_numeric(qbs["spread_line"], errors="coerce")
        qbs["team_spread_line"] = np.where(is_home, spread, -spread)

    if "total_line" in qbs.columns:
        qbs["game_total_line"] = pd.to_numeric(
            qbs["total_line"], errors="coerce"
        )

    if "temp" in qbs.columns:
        qbs["game_temp"] = pd.to_numeric(qbs["temp"], errors="coerce")

    if "wind" in qbs.columns:
        qbs["game_wind"] = pd.to_numeric(qbs["wind"], errors="coerce")

    if "location" in qbs.columns:
        qbs["neutral_site"] = (
            qbs["location"].astype(str).str.lower().eq("neutral").astype(int)
        )

    return qbs


def normalize_team_code(series: pd.Series) -> pd.Series:
    """Normalize a few provider-specific team abbreviations."""
    return series.replace(
        {
            "LAR": "LA",
            "WSH": "WAS",
        }
    )


def add_injury_features(qbs: pd.DataFrame) -> pd.DataFrame:
    """Add pregame QB injury-report and practice-participation features."""
    frames = []

    for season in range(2021, 2027):
        path = ROOT / "data" / "raw" / f"injuries_{season}.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frames.append(frame)

    qbs = qbs.copy()

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
            qbs[column] = np.nan
        return qbs

    injuries = pd.concat(frames, ignore_index=True)

    if "season_type" in injuries.columns:
        injuries = injuries[
            injuries["season_type"].astype(str).eq("REG")
        ].copy()

    if "position" in injuries.columns:
        injuries = injuries[
            injuries["position"].astype(str).eq("QB")
        ].copy()

    injuries["team_injury"] = normalize_team_code(
        injuries["team"].astype(str)
    )

    injuries["date_modified_parsed"] = pd.to_datetime(
        injuries.get("date_modified"),
        errors="coerce",
        utc=True,
    )

    # Keep the latest weekly report record for each QB/team/week.
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
    injuries["injury_out"] = report_status.str.fullmatch(
        r".*\bout\b.*"
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
        injuries["injury_questionable"] * 1
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

    qbs = qbs.merge(
        injuries[keep],
        how="left",
        left_on=["player_id", "season", "week", "team"],
        right_on=["gsis_id", "season", "week", "team_injury"],
    )

    # No injury-report match means the player was not listed that week.
    for column in injury_columns:
        qbs[column] = qbs[column].fillna(0.0)

    return qbs.drop(
        columns=[
            name
            for name in ["gsis_id", "team_injury"]
            if name in qbs.columns
        ]
    )


def add_depth_chart_features(qbs: pd.DataFrame) -> pd.DataFrame:
    """Add pregame QB depth-chart rank across the 2021-2026 schema change.

    2021-2024 depth charts are weekly and join directly by season/week/team.
    2025+ depth charts are timestamped snapshots, so we use the most recent
    snapshot no later than the game date and no more than five days old.
    """
    qbs = qbs.copy()
    qbs["_row_id"] = np.arange(len(qbs))

    qbs["depth_chart_qb_rank"] = np.nan

    # ----- 2021-2024 weekly schema -----
    old_frames = []
    for season in range(2021, 2025):
        path = ROOT / "data" / "raw" / f"depth_charts_{season}.csv"
        if path.exists():
            old_frames.append(pd.read_csv(path, low_memory=False))

    if old_frames:
        old = pd.concat(old_frames, ignore_index=True)

        if "game_type" in old.columns:
            old = old[old["game_type"].eq("REG")].copy()

        if "position" in old.columns:
            old = old[old["position"].astype(str).eq("QB")].copy()

        old["team_dc"] = normalize_team_code(old["club_code"].astype(str))
        old["depth_chart_qb_rank_old"] = pd.to_numeric(
            old["depth_team"], errors="coerce"
        )

        old_keep = old[
            [
                "gsis_id",
                "season",
                "week",
                "team_dc",
                "depth_chart_qb_rank_old",
            ]
        ].drop_duplicates(
            ["gsis_id", "season", "week", "team_dc"],
            keep="first",
        )

        qbs = qbs.merge(
            old_keep,
            how="left",
            left_on=["player_id", "season", "week", "team"],
            right_on=["gsis_id", "season", "week", "team_dc"],
        )

        old_mask = qbs["season"].le(2024)
        qbs.loc[old_mask, "depth_chart_qb_rank"] = qbs.loc[
            old_mask, "depth_chart_qb_rank_old"
        ]

        qbs = qbs.drop(
            columns=[
                col_name
                for col_name in [
                    "gsis_id",
                    "team_dc",
                    "depth_chart_qb_rank_old",
                ]
                if col_name in qbs.columns
            ]
        )

    # ----- 2025+ timestamped schema -----
    new_frames = []
    for season in range(2025, 2027):
        path = ROOT / "data" / "raw" / f"depth_charts_{season}.csv"
        if path.exists():
            frame = pd.read_csv(path, low_memory=False)
            frame["_depth_season"] = season
            new_frames.append(frame)

    if new_frames and "gameday" in qbs.columns:
        new = pd.concat(new_frames, ignore_index=True)

        if "pos_abb" in new.columns:
            new = new[new["pos_abb"].astype(str).eq("QB")].copy()

        new["team_dc"] = normalize_team_code(new["team"].astype(str))
        new["depth_dt"] = pd.to_datetime(
            new["dt"], errors="coerce", utc=True
        ).dt.tz_convert(None)
        new["depth_date"] = new["depth_dt"].dt.normalize()
        new["depth_chart_qb_rank_new"] = pd.to_numeric(
            new["pos_rank"], errors="coerce"
        )

        gameside = qbs[qbs["season"].ge(2025)][
            ["_row_id", "player_id", "team", "gameday"]
        ].copy()
        gameside["game_date"] = pd.to_datetime(
            gameside["gameday"], errors="coerce"
        ).dt.normalize()

        candidates = gameside.merge(
            new[
                [
                    "gsis_id",
                    "team_dc",
                    "depth_date",
                    "depth_chart_qb_rank_new",
                ]
            ],
            how="left",
            left_on=["player_id", "team"],
            right_on=["gsis_id", "team_dc"],
        )

        valid = candidates[
            candidates["depth_date"].notna()
            & candidates["game_date"].notna()
            & (candidates["depth_date"] <= candidates["game_date"])
            & (
                candidates["depth_date"]
                >= candidates["game_date"] - pd.Timedelta(days=5)
            )
        ].copy()

        if not valid.empty:
            latest = (
                valid.sort_values(
                    ["_row_id", "depth_date", "depth_chart_qb_rank_new"],
                    ascending=[True, False, True],
                )
                .drop_duplicates("_row_id", keep="first")
                [["_row_id", "depth_chart_qb_rank_new"]]
            )

            rank_map = latest.set_index("_row_id")[
                "depth_chart_qb_rank_new"
            ]
            current_mask = qbs["season"].ge(2025)
            qbs.loc[current_mask, "depth_chart_qb_rank"] = (
                qbs.loc[current_mask, "_row_id"].map(rank_map)
            )

    qbs["listed_qb1"] = np.where(
        qbs["depth_chart_qb_rank"].notna(),
        qbs["depth_chart_qb_rank"].eq(1).astype(float),
        np.nan,
    )

    return qbs.drop(columns=["_row_id"])


def add_snap_role_features(qbs: pd.DataFrame) -> pd.DataFrame:
    """Add prior-game offensive snap share as a leakage-safe role proxy."""
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    players = pd.read_csv(PLAYERS_FILE, low_memory=False)

    id_map = players[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")
    snaps = snaps[snaps["position"].eq("QB")].copy()

    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"], errors="coerce"
    )

    qbs = qbs.merge(
        id_map,
        how="left",
        left_on="player_id",
        right_on="gsis_id",
    )

    snap_cols = [
        "pfr_player_id",
        "season",
        "week",
        "team",
        "offense_snaps",
        "offense_pct",
    ]
    snaps = snaps[snap_cols].drop_duplicates(
        ["pfr_player_id", "season", "week", "team"]
    )

    qbs = qbs.merge(
        snaps,
        how="left",
        left_on=["pfr_id", "season", "week", "team"],
        right_on=["pfr_player_id", "season", "week", "team"],
    )

    qbs = qbs.sort_values(["player_id", "season", "week"]).copy()

    # IMPORTANT: only PRIOR games are used. Current-week snap share would leak
    # knowledge of how much the QB actually played in the game being predicted.
    qbs["previous_offense_pct"] = (
        qbs.groupby("player_id", sort=False)["offense_pct"].shift(1)
    )

    for window in (3, 5):
        qbs[f"avg_offense_pct_last_{window}"] = (
            qbs.groupby("player_id", sort=False)["offense_pct"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        qbs[f"start_like_games_last_{window}"] = (
            qbs.groupby("player_id", sort=False)["offense_pct"]
            .transform(
                lambda s: s.shift(1)
                .ge(50)
                .rolling(window, min_periods=1)
                .sum()
            )
        )

    return qbs


def main() -> None:
    print("Reading historical tables...")
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    team = pd.read_csv(TEAM_FILE, low_memory=False)

    print(f"Player rows: {len(players):,}")
    print(f"Team rows:   {len(team):,}")

    qbs = players[players["position"].eq("QB")].copy()

    # Remove QB rows that did not actually participate as passers/rushers.
    activity = (
        col(qbs, "attempts")
        + col(qbs, "carries")
        + col(qbs, "receptions")
    )
    qbs = qbs[activity > 0].copy()

    qbs["custom_fantasy_points"] = custom_fantasy_points(qbs)
    qbs = add_home_away(qbs)
    qbs = add_game_context_features(qbs)
    qbs = add_injury_features(qbs)
    qbs = add_depth_chart_features(qbs)
    qbs = add_player_rolling_features(qbs)
    qbs = add_snap_role_features(qbs)

    defense = build_defense_features(team)

    qbs = qbs.merge(
        defense,
        how="left",
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "defense_team"],
    )

    # Friendly model-table names.
    # The nflverse source already contains both player_name (short name) and
    # player_display_name (full display name). Renaming player_display_name to
    # player_name would create two columns with the same label, so build a new
    # unambiguous model column instead.
    if "player_display_name" in qbs.columns:
        qbs["model_player_name"] = qbs["player_display_name"]
    else:
        qbs["model_player_name"] = qbs["player_name"]

    qbs = qbs.rename(
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
        "depth_chart_qb_rank",
        "listed_qb1",
        "on_injury_report",
        "injury_questionable",
        "injury_doubtful",
        "injury_out",
        "practice_dnp",
        "practice_limited",
        "practice_full",
        "injury_status_score",
        "previous_offense_pct",
        "avg_offense_pct_last_3",
        "avg_offense_pct_last_5",
        "start_like_games_last_3",
        "start_like_games_last_5",
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
    ]

    # Keep only columns that exist, making the script tolerant of source-schema changes.
    output_columns = [c for c in output_columns if c in qbs.columns]
    model = qbs[output_columns].rename(
        columns={"model_player_name": "player_name"}
    )
    model = model.sort_values(["season", "week", "player_name"])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.to_csv(OUTPUT_FILE, index=False)

    print(f"\nQB rows: {len(model):,}")
    print(f"Columns: {len(model.columns)}")
    print(f"WRITE {OUTPUT_FILE}")
    print("\nSample:")
    print(model.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
