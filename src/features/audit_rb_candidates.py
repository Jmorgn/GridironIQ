"""Audit pregame RB depth-chart candidates before building GridironIQ RB v2.

The RB v1 benchmark only includes backs who recorded fantasy-relevant activity.
RB v2 needs a pregame candidate population so the model can learn committees,
inactive players, injury replacements, and multiple fantasy-relevant backs on
the same team.

This audit does NOT choose the RB role target yet. It measures how depth-chart
rank relates to actual snaps, opportunities, and fantasy output so the role
definition can be selected from evidence rather than guessed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_qb_model_dataset import (
    col,
    custom_fantasy_points,
    normalize_team_code,
)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_FILE = RAW_DIR / "players.csv"
SCHEDULE_FILE = RAW_DIR / "games.csv"
OUTPUT_FILE = PROCESSED_DIR / "rb_candidate_audit.csv"


def build_game_team_weeks() -> pd.DataFrame:
    games = pd.read_csv(SCHEDULE_FILE, low_memory=False)

    if "game_type" in games.columns:
        games = games[
            games["game_type"].astype(str).eq("REG")
        ].copy()

    rows = []
    for _, game in games[
        games["season"].between(2021, 2026)
    ].iterrows():
        season = pd.to_numeric(game.get("season"), errors="coerce")
        week = pd.to_numeric(game.get("week"), errors="coerce")
        game_date = pd.to_datetime(
            game.get("gameday"),
            errors="coerce",
        )

        if pd.isna(season) or pd.isna(week) or pd.isna(game_date):
            continue

        for side in ("home", "away"):
            team = normalize_team_code(
                pd.Series([str(game.get(f"{side}_team"))])
            ).iloc[0]

            rows.append(
                {
                    "season": int(season),
                    "week": int(week),
                    "team": team,
                    "game_date": game_date.normalize(),
                }
            )

    return pd.DataFrame(rows).drop_duplicates(
        ["season", "week", "team"]
    )


def load_rb_depth_candidates() -> pd.DataFrame:
    frames = []
    game_team = build_game_team_weeks()

    # 2021-2024: weekly depth charts.
    for season in range(2021, 2025):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)

        if "game_type" in df.columns:
            df = df[
                df["game_type"].astype(str).eq("REG")
            ].copy()

        if "position" in df.columns:
            df = df[
                df["position"].astype(str).eq("RB")
            ].copy()

        frames.append(
            pd.DataFrame(
                {
                    "player_id": df["gsis_id"],
                    "season": pd.to_numeric(
                        df["season"],
                        errors="coerce",
                    ),
                    "week": pd.to_numeric(
                        df["week"],
                        errors="coerce",
                    ),
                    "team": normalize_team_code(
                        df["club_code"].astype(str)
                    ),
                    "depth_chart_rb_rank": pd.to_numeric(
                        df["depth_team"],
                        errors="coerce",
                    ),
                }
            )
        )

    # 2025+: timestamped snapshots. Use the latest snapshot within five days
    # leading into that team's scheduled game.
    for season in range(2025, 2027):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)

        if "pos_abb" in df.columns:
            df = df[
                df["pos_abb"].astype(str).eq("RB")
            ].copy()

        df["team"] = normalize_team_code(
            df["team"].astype(str)
        )
        df["depth_date"] = pd.to_datetime(
            df["dt"],
            errors="coerce",
            utc=True,
        ).dt.tz_convert(None).dt.normalize()
        df["depth_chart_rb_rank"] = pd.to_numeric(
            df["pos_rank"],
            errors="coerce",
        )

        season_games = game_team[
            game_team["season"].eq(season)
        ].copy()

        candidates = season_games.merge(
            df,
            how="left",
            on="team",
        )

        candidates = candidates[
            candidates["depth_date"].notna()
            & (candidates["depth_date"] <= candidates["game_date"])
            & (
                candidates["depth_date"]
                >= candidates["game_date"] - pd.Timedelta(days=5)
            )
        ].copy()

        if candidates.empty:
            continue

        latest_date = (
            candidates.groupby(
                ["season", "week", "team"],
                as_index=False,
            )["depth_date"]
            .max()
        )

        candidates = candidates.merge(
            latest_date,
            how="inner",
            on=[
                "season",
                "week",
                "team",
                "depth_date",
            ],
        )

        frames.append(
            pd.DataFrame(
                {
                    "player_id": candidates["gsis_id"],
                    "season": candidates["season"],
                    "week": candidates["week"],
                    "team": candidates["team"],
                    "depth_chart_rb_rank": candidates[
                        "depth_chart_rb_rank"
                    ],
                }
            )
        )

    if not frames:
        raise RuntimeError("No RB depth-chart candidate data found.")

    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.dropna(
        subset=["player_id", "season", "week", "team"]
    )
    candidates["season"] = candidates["season"].astype(int)
    candidates["week"] = candidates["week"].astype(int)

    candidates = candidates.drop_duplicates(
        ["player_id", "season", "week", "team"]
    )

    # Remove bye-week contamination from older weekly depth-chart files.
    scheduled = game_team[
        ["season", "week", "team"]
    ].drop_duplicates()

    candidates = candidates.merge(
        scheduled,
        how="inner",
        on=["season", "week", "team"],
    )

    candidates["listed_rb1"] = (
        candidates["depth_chart_rb_rank"].eq(1).astype(int)
    )

    return candidates


def add_actual_rb_results(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    players["team"] = normalize_team_code(
        players["team"].astype(str)
    )

    rbs = players[
        players["position"].astype(str).eq("RB")
    ].copy()

    rbs["carries_num"] = col(rbs, "carries")
    rbs["targets_num"] = col(rbs, "targets")
    rbs["receptions_num"] = col(rbs, "receptions")
    rbs["opportunities"] = (
        rbs["carries_num"] + rbs["targets_num"]
    )
    rbs["touches"] = (
        rbs["carries_num"] + rbs["receptions_num"]
    )
    rbs["actual_fantasy_points"] = custom_fantasy_points(rbs)

    actual = (
        rbs.groupby(
            ["player_id", "season", "week", "team"],
            as_index=False,
        )
        .agg(
            carries=("carries_num", "sum"),
            targets=("targets_num", "sum"),
            receptions=("receptions_num", "sum"),
            opportunities=("opportunities", "sum"),
            touches=("touches", "sum"),
            actual_fantasy_points=(
                "actual_fantasy_points",
                "sum",
            ),
        )
    )
    actual["had_stat_row"] = 1

    out = candidates.merge(
        actual,
        how="left",
        on=["player_id", "season", "week", "team"],
    )

    fill_columns = [
        "carries",
        "targets",
        "receptions",
        "opportunities",
        "touches",
        "actual_fantasy_points",
        "had_stat_row",
    ]

    for column in fill_columns:
        out[column] = out[column].fillna(0.0)

    out["recorded_activity"] = (
        out["opportunities"].gt(0)
        | out["receptions"].gt(0)
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

    out["offense_snaps"] = out["offense_snaps"].fillna(0.0)
    out["offense_pct"] = out["offense_pct"].fillna(0.0)
    out["played_any_snap"] = (
        out["offense_snaps"].gt(0)
        | out["had_stat_row"].eq(1)
    ).astype(int)

    return out


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

    if name_col is None:
        candidates["player_name"] = candidates["player_id"]
        return candidates

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

    return candidates.merge(
        names,
        how="left",
        on="player_id",
    )


def main() -> None:
    print("GRIDIRONIQ RB CANDIDATE AUDIT")
    print("=" * 72)

    candidates = load_rb_depth_candidates()
    candidates = add_actual_rb_results(candidates)
    candidates = add_snap_results(candidates)
    candidates = add_names(candidates)

    historical = candidates[
        candidates["season"].between(2021, 2025)
    ].copy()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    historical.to_csv(OUTPUT_FILE, index=False)

    print(f"Historical candidate rows: {len(historical):,}")
    print(
        "Zero recorded activity:     "
        f"{historical['recorded_activity'].eq(0).mean():.1%}"
    )
    print(
        "Played any offensive snap: "
        f"{historical['played_any_snap'].eq(1).mean():.1%}"
    )

    print("\nROLE / WORKLOAD THRESHOLDS")
    thresholds = [
        ("20%+ offensive snaps", historical["offense_pct"].ge(0.20)),
        ("35%+ offensive snaps", historical["offense_pct"].ge(0.35)),
        ("50%+ offensive snaps", historical["offense_pct"].ge(0.50)),
        ("5+ opportunities", historical["opportunities"].ge(5)),
        ("10+ opportunities", historical["opportunities"].ge(10)),
        ("15+ opportunities", historical["opportunities"].ge(15)),
    ]

    for label, mask in thresholds:
        print(f"{label:<25} {mask.mean():>6.1%} ({int(mask.sum()):,})")

    print("\nBY DEPTH-CHART RANK")
    summary = (
        historical.assign(
            zero_activity=historical["recorded_activity"].eq(0).astype(int),
            snap_20=historical["offense_pct"].ge(0.20).astype(int),
            snap_35=historical["offense_pct"].ge(0.35).astype(int),
            opp_5=historical["opportunities"].ge(5).astype(int),
            opp_10=historical["opportunities"].ge(10).astype(int),
        )
        .groupby("depth_chart_rb_rank", dropna=False)
        .agg(
            rows=("player_id", "size"),
            zero_activity_pct=("zero_activity", "mean"),
            played_snap_pct=("played_any_snap", "mean"),
            snap_20_pct=("snap_20", "mean"),
            snap_35_pct=("snap_35", "mean"),
            opp_5_pct=("opp_5", "mean"),
            opp_10_pct=("opp_10", "mean"),
            median_opportunities=("opportunities", "median"),
            mean_fantasy_points=("actual_fantasy_points", "mean"),
        )
        .reset_index()
        .sort_values("depth_chart_rb_rank")
    )

    pct_columns = [
        "zero_activity_pct",
        "played_snap_pct",
        "snap_20_pct",
        "snap_35_pct",
        "opp_5_pct",
        "opp_10_pct",
    ]

    print(
        summary.to_string(
            index=False,
            formatters={
                **{
                    column: "{:.1%}".format
                    for column in pct_columns
                },
                "median_opportunities": "{:.1f}".format,
                "mean_fantasy_points": "{:.2f}".format,
            },
        )
    )

    rb1 = historical[
        historical["depth_chart_rb_rank"].eq(1)
    ]
    rb2 = historical[
        historical["depth_chart_rb_rank"].eq(2)
    ]

    print("\nRB1 / RB2 QUICK CHECK")
    for label, frame in [("RB1", rb1), ("RB2", rb2)]:
        if frame.empty:
            continue

        print(
            f"{label}: rows={len(frame):,} | "
            f"played={frame['played_any_snap'].mean():.1%} | "
            f"35%+ snaps={frame['offense_pct'].ge(0.35).mean():.1%} | "
            f"10+ opp={frame['opportunities'].ge(10).mean():.1%} | "
            f"mean FP={frame['actual_fantasy_points'].mean():.2f}"
        )

    print(f"\nSaved detailed audit to: {OUTPUT_FILE}")
    print(
        "\nNo role target has been chosen yet. Use this audit to decide "
        "whether RB v2 should classify snap share, opportunities, or a "
        "combined workload outcome."
    )


if __name__ == "__main__":
    main()
