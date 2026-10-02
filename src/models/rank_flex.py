"""Rank FLEX-eligible RBs and WRs using saved GridironIQ v2 projections.

FLEX is a derived ranking, NOT another trained model. Tight ends are not
eligible in this league's FLEX slot. Default output is the full RB/WR
pool; supply --exclude for players already locked into starting RB/WR
slots to see the highest projected available FLEX candidates.

Examples:
    py src\models\rank_flex.py
    py src\models\rank_flex.py --exclude "Jahmyr Gibbs" --exclude "CeeDee Lamb"
    py src\models\rank_flex.py --exclude-id 00-0036322
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
INPUTS = {
    "RB": PROCESSED_DIR / "rb_v2_weekly_rankings.csv",
    "WR": PROCESSED_DIR / "wr_v2_weekly_rankings.csv",
}
FULL_OUTPUT = PROCESSED_DIR / "flex_weekly_rankings.csv"
AVAILABLE_OUTPUT = PROCESSED_DIR / "flex_available_rankings.csv"

OUTPUT_COLUMNS = [
    "rank", "position", "position_rank",
    "player_id", "player_name", "team", "opponent",
    "season", "week", "depth_chart_rank",
    "role_probability", "conditional_fantasy_points",
    "gridironiq_projection", "prediction_low_80",
    "prediction_high_80",
]


def _load_position(position: str) -> pd.DataFrame:
    path = INPUTS[position]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run py run_weekly.py to generate "
            f"current-week {position} rankings first."
        )
    df = pd.read_csv(path, low_memory=False, dtype={"player_id": str})
    if df.empty:
        raise RuntimeError(f"{position} rankings are empty: {path}")

    probability = f"{position.lower()}_role_probability"
    position_rank = f"depth_chart_{position.lower()}_rank"
    required = [
        "player_id", "player_name", "season", "week",
        "team", "opponent", "gridironiq_projection",
        probability, "conditional_fantasy_points",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise RuntimeError(
            f"{position} rankings missing columns: {missing}"
        )

    df["season"] = pd.to_numeric(
        df["season"], errors="raise"
    ).astype(int)
    df["week"] = pd.to_numeric(
        df["week"], errors="raise"
    ).astype(int)
    weeks = df[["season", "week"]].drop_duplicates()
    if len(weeks) != 1:
        raise RuntimeError(
            f"{position} rankings contain multiple season/weeks. "
            "Refresh both position rankings before combining."
        )

    df["position"] = position
    df["role_probability"] = pd.to_numeric(
        df[probability], errors="coerce"
    )
    if position_rank in df.columns:
        df["depth_chart_rank"] = df[position_rank]
    else:
        df["depth_chart_rank"] = pd.NA
    for column in [
        "gridironiq_projection",
        "conditional_fantasy_points",
        "prediction_low_80",
        "prediction_high_80",
    ]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if (
        df["gridironiq_projection"].isna().any()
        or df["role_probability"].isna().any()
    ):
        raise RuntimeError(
            f"{position} rankings contain missing projections "
            "or role probabilities."
        )
    if (
        df["role_probability"].lt(0)
        | df["role_probability"].gt(1)
    ).any():
        raise RuntimeError(
            f"{position} role probabilities must be in [0, 1]."
        )
    if df.duplicated(["player_id", "season", "week"]).any():
        raise RuntimeError(
            f"Duplicate {position} player IDs in the same game week."
        )

    return df


def build_flex_rankings(
    excluded_names: tuple[str, ...] = (),
    excluded_ids: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a combined, deterministic RB/WR ranking.

    Exclusions represent players already locked into the user's lineup;
    do not infer actual roster membership or starting-slot counts.
    """
    rb = _load_position("RB")
    wr = _load_position("WR")
    rb_week = tuple(rb[["season", "week"]].iloc[0])
    wr_week = tuple(wr[["season", "week"]].iloc[0])
    if rb_week != wr_week:
        raise RuntimeError(
            f"RB rankings are for {rb_week} but WR rankings "
            f"are for {wr_week}. Refresh both positions together."
        )

    combined = pd.concat([rb, wr], ignore_index=True)
    combined = combined.sort_values(
        [
            "gridironiq_projection", "role_probability",
            "position", "player_name", "player_id",
        ],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).copy()

    # Position rank is based on each position's actual saved
    # projection, not the player's spot on the depth chart.
    combined["position_rank"] = (
        combined.groupby("position").cumcount() + 1
    )
    combined["player_name_key"] = (
        combined["player_name"]
        .astype(str).str.strip().str.casefold()
    )
    known_ids = set(combined["player_id"].astype(str))
    known_names = set(combined["player_name_key"])

    # An ambiguous player name must be disambiguated with player_id.
    names = {name.strip().casefold() for name in excluded_names}
    ids = {player_id.strip() for player_id in excluded_ids}
    unknown_names = names - known_names
    unknown_ids = ids - known_ids
    if unknown_names or unknown_ids:
        raise ValueError(
            "Excluded player(s) not found in this week's FLEX pool: "
            f"names={sorted(unknown_names)}, ids={sorted(unknown_ids)}"
        )
    for name in names:
        candidates = combined.loc[
            combined["player_name_key"].eq(name), "player_id"
        ].unique()
        if len(candidates) > 1:
            raise ValueError(
                f"Name {name!r} matches multiple players; "
                "use --exclude-id instead."
            )

    locked = (
        combined["player_name_key"].isin(names)
        | combined["player_id"].isin(ids)
    )
    available = combined.loc[~locked].copy()
    available.insert(0, "rank", range(1, len(available) + 1))
    return available[OUTPUT_COLUMNS].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank RB/WR-only FLEX options from official projections."
    )
    parser.add_argument(
        "--exclude", action="append", default=[],
        metavar="PLAYER_NAME",
        help="Exclude a player already starting at RB or WR; repeatable.",
    )
    parser.add_argument(
        "--exclude-id", action="append", default=[],
        metavar="PLAYER_ID",
        help="Exclude by exact player ID if names are ambiguous; repeatable.",
    )
    args = parser.parse_args()
    rankings = build_flex_rankings(
        tuple(args.exclude), tuple(args.exclude_id)
    )
    custom = bool(args.exclude or args.exclude_id)
    output = AVAILABLE_OUTPUT if custom else FULL_OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(output, index=False)

    print("GRIDIRONIQ RB/WR FLEX RANKINGS")
    print("=" * 82)
    if rankings.empty:
        print("No eligible RB/WR candidates remain after exclusions.")
        print(f"Saved: {output}")
        return
    first = rankings.iloc[0]
    print(
        f"Season {int(first['season'])} | Week {int(first['week'])}"
        f" | Eligible RBs/WRs: {len(rankings)}"
    )
    if custom:
        print(
            "Excluded locked starters: "
            + ", ".join([*args.exclude, *args.exclude_id])
        )
    else:
        print(
            "Full RB/WR pool. To see YOUR available FLEX choices, "
            "pass --exclude for each already-starting RB and WR."
        )

    display = rankings[
        [
            "rank", "position", "position_rank",
            "player_name", "team", "opponent",
            "gridironiq_projection", "role_probability",
            "prediction_low_80", "prediction_high_80",
        ]
    ].head(30).copy()
    print(
        display.to_string(
            index=False,
            formatters={
                "gridironiq_projection": "{:.2f}".format,
                "role_probability": "{:.1%}".format,
                "prediction_low_80": "{:.2f}".format,
                "prediction_high_80": "{:.2f}".format,
            },
        )
    )
    print(f"Saved: {output}")
    print(
        "FLEX ranking reuses RB/WR models. It does not create "
        "a separate FLEX model or include tight ends."
    )


if __name__ == "__main__":
    main()
