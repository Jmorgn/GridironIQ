"""Build GridironIQ Kicker v1's pregame candidate dataset.

2021-24: schedule-filtered weekly K depth charts (publication timestamp
unavailable). 2025-26: use the latest K/PK snapshot timestamped within
five days before each game's scheduled kickoff. Completed game/team
boxscore is required before interpreting missing kicker stats as zero.

Outcome is CONFIRMED COMPONENT POINTS (FG distance rules + made PAT).
Missed/blocked PAT scoring has NOT been confirmed; this is a research
target, NOT an official Yahoo fantasy score or production projection.
Features are prior-week rolling history and pregame game context.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from build_qb_model_dataset import normalize_team_code
from build_qb_v2_candidate_dataset import build_schedule_team_weeks

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
sys.path.insert(0, str(ROOT / "src" / "scoring"))

from tracker_common import load_schedule  # noqa: E402
from league_rules import kicker_known_components  # noqa: E402

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PLAYER_FILE = PROCESSED / "player_weekly_2021_2026.csv"
TEAM_FILE = PROCESSED / "team_weekly_2021_2026.csv"
PLAYERS_META = RAW / "players.csv"
OUTPUT_FILE = PROCESSED / "k_v1_candidate_dataset.csv"

KEY = ["player_id", "season", "week", "team"]
GAME_KEY = ["season", "week", "team"]

PLAYER_METRICS = {
    "actual_confirmed_component_points": "fp",
    "actual_fg_att": "fg_att",
    "actual_fg_made": "fg_made",
    "actual_pat_att": "pat_att",
    "actual_pat_made": "pat_made",
    "actual_fg_40plus_made": "fg_40plus_made",
    "actual_fg_50plus_made": "fg_50plus_made",
    "actual_long_misses": "long_misses",
    "actual_kick_attempts": "kick_attempts",
    "actual_active_kicker": "active_kicker",
}

TEAM_METRICS = {
    "team_kicker_component_points": "kicker_fp",
    "team_fg_att": "fg_att",
    "team_fg_made": "fg_made",
    "team_pat_att": "pat_att",
    "team_pat_made": "pat_made",
    "team_total_yards": "total_yards",
    "team_offensive_tds": "offensive_tds",
}

OPP_METRICS = {
    "allowed_fg_att": "fg_att_allowed",
    "allowed_pat_att": "pat_att_allowed",
    "allowed_total_yards": "yards_allowed",
}


def week_key(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["season"], errors="raise").astype("int64")
        * 100
        + pd.to_numeric(frame["week"], errors="raise").astype("int64")
    )


def schedule_context() -> pd.DataFrame:
    context = build_schedule_team_weeks()
    official = load_schedule()
    context = context.merge(
        official[
            ["season", "week", "team", "opponent", "kickoff_utc"]
        ],
        on=["season", "week", "team", "opponent"],
        how="inner",
        validate="one_to_one",
    )
    return context[
        context["season"].between(2021, 2026)
    ].copy()


def depth_chart_candidates(schedule: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for season in range(2021, 2025):
        path = RAW / f"depth_charts_{season}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        chart = pd.read_csv(path, low_memory=False)
        if "game_type" in chart.columns:
            chart = chart[
                chart["game_type"].astype(str).eq("REG")
            ].copy()
        chart = chart[
            chart["position"].astype(str).str.upper()
            .isin(["K", "PK"])
        ].copy()
        frames.append(pd.DataFrame({
            "player_id": chart["gsis_id"].astype(str),
            "season": pd.to_numeric(
                chart["season"], errors="coerce"
            ),
            "week": pd.to_numeric(chart["week"], errors="coerce"),
            "team": normalize_team_code(
                chart["club_code"].astype(str)
            ),
            "depth_chart_k_rank": pd.to_numeric(
                chart["depth_team"], errors="coerce"
            ),
            "chart_pregame_verified": 0,
        }))

    future_schedule = schedule[
        schedule["season"].between(2025, 2026)
    ][GAME_KEY + ["kickoff_utc"]].copy()

    for season in (2025, 2026):
        path = RAW / f"depth_charts_{season}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        chart = pd.read_csv(path, low_memory=False)
        pos_col = "pos_abb" if "pos_abb" in chart else "position"
        chart = chart[
            chart[pos_col].astype(str).str.upper()
            .isin(["K", "PK"])
        ].copy()
        chart["team"] = normalize_team_code(
            chart["team"].astype(str)
        )
        chart["chart_timestamp_utc"] = pd.to_datetime(
            chart["dt"], errors="coerce", utc=True,
        )
        chart["depth_chart_k_rank"] = pd.to_numeric(
            chart["pos_rank"], errors="coerce"
        )
        games = future_schedule[
            future_schedule["season"].eq(season)
        ]
        candidates = games.merge(
            chart, on="team", how="inner",
        )
        # Timestamp-level safety: a same-day postkickoff update
        # cannot enter the candidate dataset.
        eligible = (
            candidates["chart_timestamp_utc"].notna()
            & candidates["kickoff_utc"].notna()
            & candidates["chart_timestamp_utc"].le(
                candidates["kickoff_utc"]
            )
            & candidates["chart_timestamp_utc"].ge(
                candidates["kickoff_utc"] - pd.Timedelta(days=5)
            )
        )
        candidates = candidates.loc[eligible].copy()
        if candidates.empty:
            continue
        latest = candidates.groupby(
            GAME_KEY
        )["chart_timestamp_utc"].transform("max")
        candidates = candidates[
            candidates["chart_timestamp_utc"].eq(latest)
        ].copy()
        frames.append(pd.DataFrame({
            "player_id": candidates["gsis_id"].astype(str),
            "season": candidates["season"],
            "week": candidates["week"],
            "team": candidates["team"],
            "depth_chart_k_rank": candidates[
                "depth_chart_k_rank"
            ],
            "chart_pregame_verified": 1,
        }))

    if not frames:
        raise RuntimeError("No historical K/PK depth charts found.")
    out = pd.concat(frames, ignore_index=True)
    out = out[
        out["player_id"].notna()
        & out["player_id"].ne("nan")
        & out["season"].notna()
        & out["week"].notna()
    ].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out[
        out["season"].between(2021, 2026)
    ].copy()
    out = out.sort_values(
        ["player_id", "season", "week", "team",
         "depth_chart_k_rank"]
    ).drop_duplicates(KEY, keep="first")
    # The schedule join excludes bye-week chart rows.
    out = out.merge(
        schedule, on=GAME_KEY, how="inner",
        validate="many_to_one",
    )
    if out.duplicated(KEY).any():
        raise RuntimeError("Duplicate kicker/game candidate keys.")
    out["listed_k1"] = out["depth_chart_k_rank"].eq(1).astype(int)
    return out


def add_names(frame: pd.DataFrame) -> pd.DataFrame:
    meta = pd.read_csv(PLAYERS_META, low_memory=False)
    name_col = next(
        (c for c in [
            "display_name", "full_name", "football_name",
            "short_name",
        ] if c in meta.columns),
        None,
    )
    frame = frame.copy()
    if name_col is None:
        frame["player_name"] = frame["player_id"]
        return frame
    names = (
        meta[["gsis_id", name_col]]
        .rename(columns={
            "gsis_id": "player_id",
            name_col: "player_name",
        })
        .drop_duplicates("player_id")
    )
    names["player_id"] = names["player_id"].astype(str)
    frame = frame.merge(
        names, on="player_id", how="left",
        validate="many_to_one",
    )
    frame["player_name"] = frame["player_name"].fillna(
        frame["player_id"]
    )
    return frame


def kicker_game_stats() -> pd.DataFrame:
    players = pd.read_csv(
        PLAYER_FILE, low_memory=False,
        dtype={"player_id": str},
    )
    players = players[
        players["position"].astype(str).eq("K")
        & players["season"].between(2021, 2026)
    ].copy()
    players["team"] = normalize_team_code(
        players["team"].astype(str)
    )
    if players.empty:
        raise RuntimeError("No K rows in player weekly data.")

    scoring = kicker_known_components(players)
    players["actual_confirmed_component_points"] = scoring[
        "confirmed_component_points"
    ]
    numeric_fields = [
        "fg_att", "fg_made", "pat_att", "pat_made",
        "pat_missed", "pat_blocked",
        "fg_made_40_49", "fg_made_50_59", "fg_made_60_",
    ]
    for name in numeric_fields:
        if name not in players.columns:
            raise RuntimeError(
                f"Kicker stat data missing required {name}."
            )
        players[name] = pd.to_numeric(
            players[name], errors="raise"
        )
        if players[name].isna().any():
            raise RuntimeError(
                f"Kicker stat {name} contains null values."
            )
    players["actual_fg_att"] = players["fg_att"]
    players["actual_fg_made"] = players["fg_made"]
    players["actual_pat_att"] = players["pat_att"]
    players["actual_pat_made"] = players["pat_made"]
    players["actual_fg_40plus_made"] = (
        players["fg_made_40_49"]
        + players["fg_made_50_59"]
        + players["fg_made_60_"]
    )
    players["actual_fg_50plus_made"] = (
        players["fg_made_50_59"]
        + players["fg_made_60_"]
    )
    players["actual_long_misses"] = scoring[
        "long_misses_no_penalty"
    ]
    players["actual_pat_missed"] = players["pat_missed"]
    players["actual_pat_blocked"] = players["pat_blocked"]
    players["actual_kick_attempts"] = (
        players["actual_fg_att"]
        + players["actual_pat_att"]
    )
    # Non-kicking player-week record may exist for other stats;
    # the role here is actual FG/PAT opportunity, not being rostered.
    players["actual_active_kicker"] = (
        players["actual_kick_attempts"].gt(0).astype(int)
    )

    fields = [
        "actual_confirmed_component_points",
        "actual_fg_att", "actual_fg_made",
        "actual_pat_att", "actual_pat_made",
        "actual_fg_40plus_made",
        "actual_fg_50plus_made",
        "actual_long_misses",
        "actual_pat_missed", "actual_pat_blocked",
        "actual_kick_attempts",
    ]
    actual = players.groupby(KEY, as_index=False)[
        fields
    ].sum()
    actual["actual_active_kicker"] = (
        actual["actual_kick_attempts"].gt(0).astype(int)
    )
    actual["had_stat_row"] = 1
    return actual


def add_actuals(
    candidates: pd.DataFrame,
    actual: pd.DataFrame,
    box_coverage: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.merge(
        box_coverage, on=GAME_KEY, how="left",
        validate="many_to_one",
    )
    out = out.merge(
        actual, on=KEY, how="left",
        validate="one_to_one",
    )
    scorable = (
        out["game_completed"].eq(1)
        & out["boxscore_available"].eq(1)
    )
    outcomes = [
        "actual_confirmed_component_points",
        "actual_fg_att", "actual_fg_made",
        "actual_pat_att", "actual_pat_made",
        "actual_fg_40plus_made",
        "actual_fg_50plus_made",
        "actual_long_misses",
        "actual_pat_missed", "actual_pat_blocked",
        "actual_kick_attempts",
        "actual_active_kicker", "had_stat_row",
    ]
    for name in outcomes:
        out.loc[scorable, name] = out.loc[
            scorable, name
        ].fillna(0.0)
        out.loc[~scorable, name] = np.nan
    out["has_unconfirmed_pat_event"] = np.nan
    out.loc[scorable, "has_unconfirmed_pat_event"] = (
        out.loc[scorable, "actual_pat_missed"].gt(0)
        | out.loc[scorable, "actual_pat_blocked"].gt(0)
    ).astype(int)
    return out


def add_player_history(frame: pd.DataFrame) -> pd.DataFrame:
    # A midweek transfer could otherwise shift one same-week game's
    # realized points into another same-week candidate row.
    if frame.duplicated(["player_id", "season", "week"]).any():
        raise RuntimeError(
            "Player appears on multiple K depth charts in one week; "
            "resolve the transfer before calculating player history."
        )
    out = frame.sort_values(
        ["player_id", "season", "week"]
    ).copy()
    out["prior_chart_games"] = out.groupby(
        "player_id"
    ).cumcount()
    for source, short in PLAYER_METRICS.items():
        out[source] = pd.to_numeric(
            out[source], errors="coerce"
        )
        grouped = out.groupby(
            "player_id", sort=False
        )[source]
        out[f"previous_{short}"] = grouped.shift(1)
        for window in (3, 5):
            out[f"avg_{short}_last_{window}"] = (
                grouped.transform(
                    lambda s: s.shift(1)
                    .rolling(window, min_periods=1).mean()
                )
            )
    return out


def team_game_stats() -> tuple[pd.DataFrame, pd.DataFrame]:
    teams = pd.read_csv(
        TEAM_FILE, low_memory=False,
    )
    teams = teams[
        teams["season"].between(2021, 2026)
    ].copy()
    teams["team"] = normalize_team_code(
        teams["team"].astype(str)
    )
    teams["opponent_team"] = normalize_team_code(
        teams["opponent_team"].astype(str)
    )
    required = [
        "fg_att", "fg_made", "pat_att", "pat_made",
        "passing_yards", "rushing_yards",
        "passing_tds", "rushing_tds",
    ]
    for name in required:
        if name not in teams.columns:
            raise RuntimeError(
                f"Team stats missing required field {name}."
            )
        teams[name] = pd.to_numeric(
            teams[name], errors="raise"
        )

    # Team-week feeds are one row per offensive team/game.
    duplicates = teams.duplicated(GAME_KEY)
    if duplicates.any():
        raise RuntimeError(
            "Duplicate team-week boxscores; verify nflverse data."
        )
    coverage = teams[GAME_KEY].copy()
    coverage["boxscore_available"] = 1

    teams["team_fg_att"] = teams["fg_att"]
    teams["team_fg_made"] = teams["fg_made"]
    teams["team_pat_att"] = teams["pat_att"]
    teams["team_pat_made"] = teams["pat_made"]
    teams["team_total_yards"] = (
        teams["passing_yards"] + teams["rushing_yards"]
    )
    teams["team_offensive_tds"] = (
        teams["passing_tds"] + teams["rushing_tds"]
    )
    # Actual K-only scoring added after merging kicker observations;
    # do not infer component FP from team-level kicks lacking distance.
    teams = teams[[
        *GAME_KEY, "opponent_team",
        "team_fg_att", "team_fg_made",
        "team_pat_att", "team_pat_made",
        "team_total_yards", "team_offensive_tds",
    ]]
    return teams, coverage


def add_team_history(
    candidates: pd.DataFrame,
    teams: pd.DataFrame,
    actual: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    kicker_totals = (
        actual.groupby(GAME_KEY, as_index=False)[
            "actual_confirmed_component_points"
        ].sum().rename(columns={
            "actual_confirmed_component_points":
            "team_kicker_component_points",
        })
    )
    history = schedule[GAME_KEY + [
        "game_completed"
    ]].drop_duplicates(GAME_KEY).merge(
        teams, on=GAME_KEY, how="left",
        validate="one_to_one",
    ).merge(
        kicker_totals, on=GAME_KEY, how="left",
        validate="one_to_one",
    )
    known = history["game_completed"].eq(1) & (
        history["team_fg_att"].notna()
    )
    history.loc[known, "team_kicker_component_points"] = (
        history.loc[
            known, "team_kicker_component_points"
        ].fillna(0)
    )
    for source, short in TEAM_METRICS.items():
        history[source] = pd.to_numeric(
            history[source], errors="coerce"
        )
    history = history.sort_values([
        "team", "season", "week"
    ]).copy()
    history["_week_key"] = week_key(history)
    for source, short in TEAM_METRICS.items():
        grouped = history.groupby(
            "team", sort=False
        )[source]
        history[f"snapshot_team_prev_{short}"] = grouped.shift(0)
        # The row's own completed-game stats become usable only
        # in *later* games: merge_asof forbids exact week matches.
        for window in (3, 5):
            history[f"snapshot_team_avg_{short}_last_{window}"] = (
                grouped.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).mean()
                )
            )
    columns = [
        c for c in history.columns
        if c.startswith("snapshot_team_")
    ]
    left = candidates.copy()
    left["_week_key"] = week_key(left)
    left["_candidate_order"] = np.arange(len(left))
    right = history[[
        "team", "_week_key", *columns
    ]].copy()
    merged = pd.merge_asof(
        left.sort_values(["_week_key", "team"]),
        right.sort_values(["_week_key", "team"]),
        on="_week_key",
        by="team",
        allow_exact_matches=False,
        direction="backward",
    )
    renames = {
        col: col.replace("snapshot_", "", 1)
        for col in columns
    }
    return (
        merged.rename(columns=renames)
        .sort_values("_candidate_order")
        .drop(columns=["_week_key", "_candidate_order"])
    )


def add_opponent_history(
    candidates: pd.DataFrame,
    teams: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    obs = schedule[GAME_KEY + [
        "game_completed"
    ]].drop_duplicates(GAME_KEY).merge(
        teams, on=GAME_KEY, how="left",
        validate="one_to_one",
    )
    obs = obs[
        obs["game_completed"].eq(1)
        & obs["team_fg_att"].notna()
    ].copy()
    obs["opponent"] = obs["opponent_team"]
    obs = obs.rename(columns={
        "team_fg_att": "allowed_fg_att",
        "team_pat_att": "allowed_pat_att",
        "team_total_yards": "allowed_total_yards",
    })
    obs = obs.sort_values([
        "opponent", "season", "week"
    ]).copy()
    obs["_week_key"] = week_key(obs)
    for source, short in OPP_METRICS.items():
        grouped = obs.groupby(
            "opponent", sort=False
        )[source]
        obs[f"snapshot_opp_prev_{short}"] = grouped.shift(0)
        for window in (3, 5):
            obs[f"snapshot_opp_avg_{short}_last_{window}"] = (
                grouped.transform(
                    lambda s: s.rolling(
                        window, min_periods=1
                    ).mean()
                )
            )
    columns = [
        col for col in obs.columns
        if col.startswith("snapshot_opp_")
    ]
    left = candidates.copy()
    left["_week_key"] = week_key(left)
    left["_candidate_order"] = np.arange(len(left))
    merged = pd.merge_asof(
        left.sort_values(["_week_key", "opponent"]),
        obs[["opponent", "_week_key", *columns]]
        .sort_values(["_week_key", "opponent"]),
        on="_week_key",
        by="opponent",
        allow_exact_matches=False,
        direction="backward",
    )
    renames = {
        name: name.replace("snapshot_", "", 1)
        for name in columns
    }
    return (
        merged.rename(columns=renames)
        .sort_values("_candidate_order")
        .drop(columns=["_week_key", "_candidate_order"])
    )


def main() -> None:
    print("GRIDIRONIQ K V1 PRE-GAME CANDIDATE BUILDER")
    print("=" * 78)
    print(
        "Target: screenshot-confirmed FG/PAT-made scoring components. "
        "Missed/blocked PAT rules remain unconfirmed."
    )
    schedule = schedule_context()
    candidates = depth_chart_candidates(schedule)
    candidates = add_names(candidates)
    actual = kicker_game_stats()
    teams, coverage = team_game_stats()
    candidates = add_actuals(candidates, actual, coverage)
    candidates = add_player_history(candidates)
    candidates = add_team_history(
        candidates, teams, actual, schedule
    )
    candidates = add_opponent_history(
        candidates, teams, schedule
    )
    historical = candidates[
        candidates["season"].between(2021, 2025)
        & candidates["game_completed"].eq(1)
        & candidates["boxscore_available"].eq(1)
    ].copy()
    future = candidates[
        candidates["season"].eq(2026)
        & candidates["game_completed"].eq(0)
    ].copy()
    if historical.empty:
        raise RuntimeError(
            "No completed historical kicker candidates found."
        )

    actual_attempts = actual[
        actual["season"].between(2021, 2025)
        & actual["actual_kick_attempts"].gt(0)
    ][KEY].drop_duplicates()
    covered = actual_attempts.merge(
        historical[KEY], on=KEY, how="left",
        indicator=True,
    )
    coverage_rate = covered["_merge"].eq("both").mean()
    print(
        f"Observed active kicker-game candidate coverage: "
        f"{coverage_rate:.1%} "
        f"({covered['_merge'].eq('both').sum():,}"
        f"/{len(covered):,})"
    )
    # Historical chart publication timing is unverifiable in
    # 2021-24; show coverage by year before interpreting errors.
    yearly_coverage = covered.groupby("season")["_merge"].agg(
        total="size",
        matched=lambda values: values.eq("both").sum(),
    )
    yearly_coverage["rate"] = (
        yearly_coverage["matched"]
        / yearly_coverage["total"]
    )
    print("Historical active kicker coverage by season:")
    print(
        yearly_coverage.to_string(
            float_format=lambda value: f"{value:.1%}"
        )
    )
    if coverage_rate < 0.90:
        raise RuntimeError(
            "Pregame depth charts miss over 10% of observed active "
            "kicker-games. Audit chart dating and player IDs first."
        )

    selected = pd.concat(
        [historical, future], ignore_index=True
    )
    fields = [
        "player_id", "player_name",
        "season", "week", "team", "opponent",
        "home_away", "game_completed",
        "kickoff_utc", "chart_pregame_verified",
        "depth_chart_k_rank", "listed_k1",
        "team_rest", "opponent_rest",
        "rest_advantage", "neutral_site",
        "roof", "surface",
        "game_temp", "game_wind",
        "team_spread_line", "game_total_line",
        "prior_chart_games",
        *[
            f"previous_{short}"
            for short in PLAYER_METRICS.values()
        ],
        *[
            f"avg_{short}_last_{window}"
            for short in PLAYER_METRICS.values()
            for window in (3, 5)
        ],
        *[
            f"team_{prefix}_{short}"
            for short in TEAM_METRICS.values()
            for prefix in ["prev"]
        ],
        *[
            f"team_avg_{short}_last_{window}"
            for short in TEAM_METRICS.values()
            for window in (3, 5)
        ],
        *[
            f"opp_prev_{short}"
            for short in OPP_METRICS.values()
        ],
        *[
            f"opp_avg_{short}_last_{window}"
            for short in OPP_METRICS.values()
            for window in (3, 5)
        ],
        "actual_confirmed_component_points",
        "actual_fg_att", "actual_fg_made",
        "actual_pat_att", "actual_pat_made",
        "actual_fg_40plus_made",
        "actual_fg_50plus_made",
        "actual_long_misses",
        "actual_pat_missed", "actual_pat_blocked",
        "actual_kick_attempts",
        "actual_active_kicker",
        "has_unconfirmed_pat_event", "had_stat_row",
    ]
    absent = [field for field in fields if field not in selected]
    if absent:
        raise RuntimeError(
            f"Kicker builder missing output fields: {absent}"
        )
    selected = selected[fields].sort_values(
        ["season", "week", "team", "depth_chart_k_rank"]
    ).copy()
    if selected.duplicated(KEY).any():
        raise RuntimeError("Duplicate K candidate/game rows.")
    if future["actual_confirmed_component_points"].notna().any():
        raise RuntimeError(
            "Future K candidates contain game outcome leakage."
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    selected.to_csv(
        OUTPUT_FILE, index=False
    )
    print(f"Historical candidate rows: {len(historical):,}")
    print(
        "Historical K1 rows: "
        f"{historical['listed_k1'].sum():,.0f}"
    )
    print(
        "Historical no-kick-attempt candidates: "
        f"{historical['actual_active_kicker'].eq(0).mean():.1%}"
    )
    print(
        "Historical rows with missed/blocked PAT events: "
        f"{historical['has_unconfirmed_pat_event'].eq(1).sum():,}"
    )
    print(f"2026 future/unplayed candidates: {len(future):,}")
    print(
        "Future pregame chart timestamp verified: "
        f"{future['chart_pregame_verified'].mean():.1%}"
        if len(future) else
        "Future pregame chart timestamp verified: n/a"
    )
    print(f"Output columns: {len(fields)}")
    print(f"WRITE {OUTPUT_FILE}")
    print(
        "RESEARCH DATASET ONLY: kicker target is confirmed scoring "
        "components, not yet official Yahoo points."
    )


if __name__ == "__main__":
    main()
