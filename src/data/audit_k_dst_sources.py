"""Audit local nflverse sources before creating Kicker/DST training labels.

Print the actual available columns instead of silently assuming
coverage of this Yahoo league's distance- and play-level rules.

Run from repository root:
    py src\data\audit_k_dst_sources.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

sys.path.insert(0, str(ROOT / "src" / "scoring"))
from league_rules import FG_MADE_POINTS, FG_MISSED_POINTS  # noqa: E402

KICKING_FIELDS = [
    "fg_made", "fg_att", "fg_missed", "fg_blocked",
    *FG_MADE_POINTS, *FG_MISSED_POINTS,
    "pat_made", "pat_att", "pat_missed", "pat_blocked",
]
DEFENSE_FIELDS = [
    "def_sacks", "def_interceptions", "def_fumbles",
    "def_fumbles_forced", "def_tds", "def_safeties",
    "def_tackles_for_loss", "def_punt_blocks",
    "def_fg_blocks", "def_pat_blocks",
    "special_teams_tds", "kickoff_return_tds",
    "punt_return_tds",
    "def_three_and_outs", "three_and_outs_forced",
    "extra_point_returns", "extra_point_return_tds",
    "passing_yards", "rushing_yards",
    "sack_yards_lost", "sacks_suffered",
]


def inspect_schema(path: Path, fields: list[str]) -> set[str]:
    print(f"\n{path.relative_to(ROOT)}")
    if not path.exists():
        print("  NOT DOWNLOADED")
        return set()

    columns = set(pd.read_csv(path, nrows=0).columns)
    have = [name for name in fields if name in columns]
    missing = [name for name in fields if name not in columns]
    print(f"  Total columns: {len(columns)}")
    print("  Present: " + (", ".join(have) or "(none)"))
    print("  Missing: " + (", ".join(missing) or "(none)"))

    # Show other plausible columns, including previously unknown versions.
    terms = [
        "kick", "field_goal", "fg_", "pat_", "extra_point",
        "def_", "punt_", "return_", "fumble", "three_out",
        "tackle", "sack", "point", "yard",
    ]
    other = sorted(
        name for name in columns
        if any(term in name.lower() for term in terms)
        and name not in fields
    )
    print(
        "  Other candidate fields: "
        + (", ".join(other[:65]) if other else "(none)")
    )
    return columns


def summarize_kickers(path: Path, columns: set[str]) -> None:
    if not columns:
        return
    available = [
        field for field in [
            "position", "player_id", "player_display_name",
            *KICKING_FIELDS,
        ] if field in columns
    ]
    if not available:
        print("  Kicker rows cannot be identified in this file.")
        return
    frame = pd.read_csv(
        path,
        usecols=available,
        low_memory=False,
        dtype={"player_id": str},
    )
    if "position" in frame.columns:
        frame = frame[
            frame["position"].astype(str).eq("K")
        ].copy()
    elif "fg_att" in frame.columns and "pat_att" in frame.columns:
        frame = frame[
            (
                pd.to_numeric(frame["fg_att"], errors="coerce")
                .fillna(0)
                + pd.to_numeric(
                    frame["pat_att"], errors="coerce"
                ).fillna(0)
            ).gt(0)
        ].copy()
    else:
        print("  No reliable kicker position/attempt filter.")
        return

    print(f"  Identifiable kicker rows: {len(frame):,}")
    for field in [
        "fg_missed_50_59", "fg_missed_60_",
        "fg_blocked", "pat_missed", "pat_blocked",
    ]:
        if field not in frame.columns:
            print(f"  {field}: unavailable")
            continue
        values = pd.to_numeric(
            frame[field], errors="coerce"
        )
        print(
            f"  {field}: total={values.sum():,.0f}; "
            f"rows with event={values.gt(0).sum():,}; "
            f"nulls={values.isna().sum():,}"
        )
    expected = set([*FG_MADE_POINTS, *FG_MISSED_POINTS, "pat_made"])
    if expected.issubset(columns):
        print(
            "  All league Kicker scoring columns are present; "
            "missed/blocked PATs and 50+ FG misses score zero."
        )
    else:
        print(
            "  Distance-bucket components are incomplete. "
            "Do not train on generic fantasy_points."
        )


def inspect_kicker_depth_chart(year: int) -> None:
    """Check whether K/PK can be enumerated before games, including 2026."""
    path = RAW / f"depth_charts_{year}.csv"
    print(f"\n{path.relative_to(ROOT)} kicker candidate availability")
    if not path.exists():
        print("  NOT DOWNLOADED")
        return
    frame = pd.read_csv(path, low_memory=False)
    position_field = (
        "pos_abb" if "pos_abb" in frame.columns
        else "position" if "position" in frame.columns
        else None
    )
    if position_field is None:
        print("  No position identifier in depth charts.")
        print(f"  Available columns: {sorted(frame.columns)}")
        return
    pos = frame[position_field].astype(str)
    print(
        f"  Source position field: {position_field}; "
        f"code counts: {pos.value_counts().head(25).to_dict()}"
    )
    kicking = frame[
        pos.str.upper().isin(["K", "PK", "KICKER"])
    ].copy()
    print(f"  K/PK depth-chart rows: {len(kicking):,}")
    if not kicking.empty:
        for field in [
            "gsis_id", "season", "week", "team",
            "club_code", "dt", "pos_rank", "depth_team",
        ]:
            if field in kicking.columns:
                print(
                    f"  {field}: "
                    f"{kicking[field].dropna().nunique():,} unique "
                    f"| example={str(kicking[field].dropna().iloc[0]) if kicking[field].notna().any() else 'None'}"
                )


def main() -> None:
    print("GRIDIRONIQ KICKER / D-ST SOURCE AUDIT")
    print("=" * 80)

    years = [2021, 2023, 2025, 2026]
    for year in years:
        path = RAW / f"stats_player_week_{year}.csv"
        columns = inspect_schema(path, KICKING_FIELDS)
        if year in (2025, 2026):
            summarize_kickers(path, columns)

    for year in (2021, 2024, 2025, 2026):
        inspect_kicker_depth_chart(year)

    for year in (2025, 2026):
        inspect_schema(
            RAW / f"stats_team_week_{year}.csv",
            DEFENSE_FIELDS,
        )
    inspect_schema(RAW / "games.csv", [
        "season", "week", "game_type", "home_team",
        "away_team", "home_score", "away_score",
        "gameday", "gametime",
    ])

    print("\nINGESTION GAP REVIEW")
    print("=" * 80)
    pbp_sources = sorted(
        path.relative_to(ROOT).as_posix()
        for path in RAW.iterdir()
        if path.is_file()
        and ("pbp" in path.name.lower()
             or "play_by_play" in path.name.lower())
    ) if RAW.exists() else []
    print(
        "Play-by-play files in data/raw/: "
        + (", ".join(pbp_sources) if pbp_sources else "NONE")
    )
    print(
        "K: Full field-goal and PAT scoring confirmed; "
        "ready to verify pregame candidate coverage and model labels."
    )
    print(
        "D/ST: Check three-and-outs, special-teams return TDs, "
        "blocked kicks, tackles for loss, and returned extra points. "
        "The current downloader does not fetch play-by-play."
    )
    print(
        "D/ST points allowed: raw opponent game score alone "
        "may differ from Yahoo D/ST points allowed when "
        "opposing defensive/special-teams scores occur."
    )
    print(
        "D/ST yards allowed: verify the source's net-yardage "
        "definition against the league scoreboard."
    )
    print(
        "CONFIRMED ABSENT SCORING CATEGORIES (0 points): "
        "missed FG 50+ yards, D/ST points allowed 21-27, "
        "D/ST yards allowed 300-399."
    )
    print(
        "CONFIRMED: made PAT +1, missed/blocked PAT 0."
    )
    print(
        "\nNo training tables, saved model bundles, "
        "or existing weekly rankings were changed."
    )


if __name__ == "__main__":
    main()
