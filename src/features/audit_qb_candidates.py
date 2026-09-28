"""Audit pregame QB candidate coverage before building GridironIQ v2.

The current v1 dataset contains QBs only when they recorded game activity.
For v2 we want one row for every QB who was a realistic pregame candidate,
including players who were later inactive or recorded zero stats.

This audit uses historical depth charts to estimate how many QB-week candidates
we can construct and how often they produced zero activity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
SCHEDULE_FILE = RAW_DIR / "games.csv"


def normalize_team(series: pd.Series) -> pd.Series:
    return series.replace({"LAR": "LA", "WSH": "WAS"})


def load_depth_candidates() -> pd.DataFrame:
    frames = []

    # 2021-2024: weekly depth-chart schema.
    for season in range(2021, 2025):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)

        if "game_type" in df.columns:
            df = df[df["game_type"].astype(str).eq("REG")]

        if "position" in df.columns:
            df = df[df["position"].astype(str).eq("QB")]

        out = pd.DataFrame(
            {
                "player_id": df["gsis_id"],
                "season": pd.to_numeric(df["season"], errors="coerce"),
                "week": pd.to_numeric(df["week"], errors="coerce"),
                "team": normalize_team(df["club_code"].astype(str)),
                "depth_chart_qb_rank": pd.to_numeric(
                    df["depth_team"], errors="coerce"
                ),
            }
        )
        frames.append(out)

    # 2025+: timestamped snapshots. Map the latest snapshot in the five days
    # leading up to each game to that team's game/week.
    games = pd.read_csv(SCHEDULE_FILE, low_memory=False)
    if "game_type" in games.columns:
        games = games[games["game_type"].astype(str).eq("REG")]

    game_rows = []
    for _, game in games[games["season"].between(2025, 2026)].iterrows():
        game_date = pd.to_datetime(game.get("gameday"), errors="coerce")
        if pd.isna(game_date):
            continue
        for side in ("home", "away"):
            game_rows.append(
                {
                    "season": int(game["season"]),
                    "week": int(game["week"]),
                    "team": normalize_team(
                        pd.Series([str(game[f"{side}_team"])])
                    ).iloc[0],
                    "game_date": game_date.normalize(),
                }
            )
    game_team = pd.DataFrame(game_rows)

    for season in range(2025, 2027):
        path = RAW_DIR / f"depth_charts_{season}.csv"
        if not path.exists():
            continue

        df = pd.read_csv(path, low_memory=False)
        if "pos_abb" in df.columns:
            df = df[df["pos_abb"].astype(str).eq("QB")]

        df["team"] = normalize_team(df["team"].astype(str))
        df["depth_date"] = pd.to_datetime(
            df["dt"], errors="coerce", utc=True
        ).dt.tz_convert(None).dt.normalize()
        df["depth_chart_qb_rank"] = pd.to_numeric(
            df["pos_rank"], errors="coerce"
        )

        season_games = game_team[game_team["season"].eq(season)]

        candidates = season_games.merge(df, how="left", on="team")
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

        latest_date = candidates.groupby(
            ["season", "week", "team"], as_index=False
        )["depth_date"].max()

        candidates = candidates.merge(
            latest_date,
            on=["season", "week", "team", "depth_date"],
            how="inner",
        )

        out = pd.DataFrame(
            {
                "player_id": candidates["gsis_id"],
                "season": candidates["season"],
                "week": candidates["week"],
                "team": candidates["team"],
                "depth_chart_qb_rank": candidates[
                    "depth_chart_qb_rank"
                ],
            }
        )
        frames.append(out)

    if not frames:
        raise RuntimeError("No depth-chart candidate data found.")

    candidates = pd.concat(frames, ignore_index=True)
    candidates = candidates.dropna(
        subset=["player_id", "season", "week", "team"]
    )

    candidates = candidates.drop_duplicates(
        ["player_id", "season", "week", "team"]
    )

    candidates["listed_qb1"] = (
        candidates["depth_chart_qb_rank"].eq(1).astype(int)
    )
    return candidates


def main() -> None:
    candidates = load_depth_candidates()

    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    qbs = players[players["position"].eq("QB")].copy()

    activity = (
        pd.to_numeric(qbs.get("attempts", 0), errors="coerce").fillna(0)
        + pd.to_numeric(qbs.get("carries", 0), errors="coerce").fillna(0)
        + pd.to_numeric(qbs.get("receptions", 0), errors="coerce").fillna(0)
    )
    active = qbs.loc[
        activity.gt(0),
        ["player_id", "season", "week", "team"],
    ].drop_duplicates()
    active["recorded_activity"] = 1

    audit = candidates.merge(
        active,
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    audit["recorded_activity"] = audit["recorded_activity"].fillna(0).astype(int)

    historical = audit[audit["season"].between(2021, 2025)].copy()

    print("GRIDIRONIQ V2 QB CANDIDATE AUDIT")
    print("=" * 58)
    print(f"Historical candidate rows: {len(historical):,}")
    print(
        "Candidate rows with zero recorded QB activity: "
        f"{(historical['recorded_activity'].eq(0).mean() * 100):.1f}%"
    )

    qb1 = historical[historical["listed_qb1"].eq(1)]
    print(f"Historical QB1 candidate rows: {len(qb1):,}")
    print(
        "QB1 rows with zero recorded QB activity: "
        f"{(qb1['recorded_activity'].eq(0).mean() * 100):.1f}%"
    )

    print("\nBY SEASON")
    summary = (
        historical.groupby("season")
        .agg(
            candidate_rows=("player_id", "size"),
            qb1_rows=("listed_qb1", "sum"),
            zero_activity_rows=(
                "recorded_activity",
                lambda s: int((s == 0).sum()),
            ),
        )
    )
    summary["zero_activity_pct"] = (
        summary["zero_activity_rows"] / summary["candidate_rows"] * 100
    )
    print(
        summary.to_string(
            formatters={"zero_activity_pct": "{:.1f}".format}
        )
    )

    print(
        "\nThis audit does not train a new model yet. It checks whether the "
        "historical depth charts provide enough zero-activity examples to build "
        "the v2 pregame candidate table safely."
    )


if __name__ == "__main__":
    main()
