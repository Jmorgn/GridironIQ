"""Validate GridironIQ v2 QB1 zero-activity candidates.

The candidate audit found that roughly one in five historical QB1 depth-chart
rows had zero recorded QB activity. Before using those rows as zero-point
training examples, this script checks whether they look like real availability
cases or data-join mistakes.

Checks:
- Did the QB record any offensive snaps despite having zero tracked activity?
- Was the QB on the injury report?
- Was another QB active for the same team/week?
- Show representative zero-activity QB1 examples.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit_qb_candidates import load_depth_candidates, normalize_team

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

PLAYER_FILE = PROCESSED_DIR / "player_weekly_2021_2026.csv"
SNAP_FILE = PROCESSED_DIR / "snap_counts_2021_2026.csv"
PLAYERS_META_FILE = RAW_DIR / "players.csv"
OUTPUT_FILE = PROCESSED_DIR / "qb1_zero_activity_audit.csv"


def load_active_qbs() -> pd.DataFrame:
    players = pd.read_csv(PLAYER_FILE, low_memory=False)
    qbs = players[players["position"].astype(str).eq("QB")].copy()

    def numeric(name: str) -> pd.Series:
        if name not in qbs.columns:
            return pd.Series(0.0, index=qbs.index)
        return pd.to_numeric(qbs[name], errors="coerce").fillna(0)

    activity = numeric("attempts") + numeric("carries") + numeric("receptions")
    qbs = qbs[activity.gt(0)].copy()

    name_col = (
        "player_display_name"
        if "player_display_name" in qbs.columns
        else "player_name"
    )

    active = qbs[
        ["player_id", name_col, "season", "week", "team"]
    ].copy()
    active = active.rename(columns={name_col: "active_player_name"})
    active["team"] = normalize_team(active["team"].astype(str))
    return active.drop_duplicates(
        ["player_id", "season", "week", "team"]
    )


def load_player_names() -> pd.DataFrame:
    meta = pd.read_csv(PLAYERS_META_FILE, low_memory=False)

    candidates = [
        "display_name",
        "full_name",
        "football_name",
        "short_name",
    ]
    name_col = next((c for c in candidates if c in meta.columns), None)

    if name_col is None:
        out = meta[["gsis_id"]].copy()
        out["player_name"] = out["gsis_id"]
        return out

    return (
        meta[["gsis_id", name_col]]
        .rename(columns={name_col: "player_name"})
        .drop_duplicates("gsis_id")
    )


def load_snap_check() -> pd.DataFrame:
    snaps = pd.read_csv(SNAP_FILE, low_memory=False)
    meta = pd.read_csv(PLAYERS_META_FILE, low_memory=False)

    id_map = meta[["gsis_id", "pfr_id"]].dropna().drop_duplicates("gsis_id")

    snaps = snaps[snaps["position"].astype(str).eq("QB")].copy()
    snaps["team"] = normalize_team(snaps["team"].astype(str))
    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"], errors="coerce"
    ).fillna(0)
    snaps["offense_snaps"] = pd.to_numeric(
        snaps["offense_snaps"], errors="coerce"
    ).fillna(0)

    snaps = snaps.merge(
        id_map,
        how="left",
        left_on="pfr_player_id",
        right_on="pfr_id",
    )

    return (
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


def load_injury_check() -> pd.DataFrame:
    frames = []

    for season in range(2021, 2026):
        path = RAW_DIR / f"injuries_{season}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, low_memory=False))

    if not frames:
        return pd.DataFrame(
            columns=[
                "player_id",
                "season",
                "week",
                "team",
                "report_status",
                "practice_status",
            ]
        )

    injuries = pd.concat(frames, ignore_index=True)

    if "season_type" in injuries.columns:
        injuries = injuries[
            injuries["season_type"].astype(str).eq("REG")
        ].copy()

    injuries["team"] = normalize_team(injuries["team"].astype(str))
    injuries["_modified"] = pd.to_datetime(
        injuries.get("date_modified"),
        errors="coerce",
        utc=True,
    )

    injuries = (
        injuries.sort_values("_modified")
        .drop_duplicates(
            ["gsis_id", "season", "week", "team"],
            keep="last",
        )
        .rename(columns={"gsis_id": "player_id"})
    )

    keep = ["player_id", "season", "week", "team"]
    for column in ["report_status", "practice_status"]:
        if column in injuries.columns:
            keep.append(column)
        else:
            injuries[column] = ""
            keep.append(column)

    return injuries[keep]


def main() -> None:
    candidates = load_depth_candidates()
    candidates = candidates[
        candidates["season"].between(2021, 2025)
        & candidates["listed_qb1"].eq(1)
    ].copy()

    active = load_active_qbs()

    candidate_activity = active[
        ["player_id", "season", "week", "team"]
    ].copy()
    candidate_activity["recorded_activity"] = 1

    audit = candidates.merge(
        candidate_activity,
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    audit["recorded_activity"] = (
        audit["recorded_activity"].fillna(0).astype(int)
    )
    zero = audit[audit["recorded_activity"].eq(0)].copy()

    zero = zero.merge(
        load_player_names(),
        how="left",
        left_on="player_id",
        right_on="gsis_id",
    ).drop(columns=["gsis_id"], errors="ignore")

    zero = zero.merge(
        load_snap_check(),
        how="left",
        on=["player_id", "season", "week", "team"],
    )
    zero["offense_snaps"] = zero["offense_snaps"].fillna(0)
    zero["offense_pct"] = zero["offense_pct"].fillna(0)

    zero = zero.merge(
        load_injury_check(),
        how="left",
        on=["player_id", "season", "week", "team"],
    )

    active_names = (
        active.groupby(["season", "week", "team"])["active_player_name"]
        .apply(lambda s: ", ".join(sorted(set(s.dropna().astype(str)))))
        .reset_index(name="other_active_qbs")
    )
    zero = zero.merge(
        active_names,
        how="left",
        on=["season", "week", "team"],
    )

    zero["has_offensive_snaps"] = zero["offense_snaps"].gt(0)
    zero["has_injury_report"] = zero["report_status"].notna()

    report = zero["report_status"].fillna("").astype(str).str.lower()
    zero["reported_out"] = report.str.contains(
        r"\bout\b", regex=True
    )
    zero["reported_doubtful"] = report.str.contains(
        "doubtful", regex=False
    )
    zero["reported_questionable"] = report.str.contains(
        "questionable", regex=False
    )

    total = len(zero)

    print("GRIDIRONIQ V2 QB1 ZERO-ACTIVITY VALIDATION")
    print("=" * 62)
    print(f"Zero-activity QB1 rows checked: {total:,}")

    if total:
        print(
            "Rows that still show offensive snaps: "
            f"{zero['has_offensive_snaps'].mean() * 100:.1f}%"
        )
        print(
            "Rows with an injury-report match:     "
            f"{zero['has_injury_report'].mean() * 100:.1f}%"
        )
        print(
            "Rows officially reported OUT:         "
            f"{zero['reported_out'].mean() * 100:.1f}%"
        )
        print(
            "Rows reported doubtful/questionable: "
            f"{(zero['reported_doubtful'] | zero['reported_questionable']).mean() * 100:.1f}%"
        )
        print(
            "Rows where another QB was active:     "
            f"{zero['other_active_qbs'].notna().mean() * 100:.1f}%"
        )

    output_columns = [
        "player_name",
        "player_id",
        "season",
        "week",
        "team",
        "depth_chart_qb_rank",
        "offense_snaps",
        "offense_pct",
        "report_status",
        "practice_status",
        "other_active_qbs",
    ]

    zero[output_columns].sort_values(
        ["season", "week", "team", "player_name"]
    ).to_csv(OUTPUT_FILE, index=False)

    print("\nSAMPLE ZERO-ACTIVITY QB1 ROWS")
    print(
        zero[output_columns]
        .sort_values(["season", "week", "team"])
        .head(30)
        .to_string(index=False)
    )

    print(f"\nSaved full audit to: {OUTPUT_FILE}")
    print(
        "\nIf offensive-snap mismatches are rare and another QB is usually "
        "active instead, these are useful v2 zero-point training examples "
        "rather than simple join failures."
    )


if __name__ == "__main__":
    main()
