"""Combine raw yearly NFL CSV files into historical tables.

Reads:
- data/raw/stats_player_week_YYYY.csv
- data/raw/stats_team_week_YYYY.csv
- data/raw/snap_counts_YYYY.csv

Writes:
- data/processed/player_weekly_2021_2026.csv
- data/processed/team_weekly_2021_2026.csv
- data/processed/snap_counts_2021_2026.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

START_SEASON = 2021
END_SEASON = 2026


def combine_yearly_files(prefix: str, output_name: str) -> pd.DataFrame:
    frames = []

    for season in range(START_SEASON, END_SEASON + 1):
        path = RAW_DIR / f"{prefix}_{season}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

        df = pd.read_csv(path)
        frames.append(df)
        print(f"READ  {path.name:<32} rows={len(df):,}")

    combined = pd.concat(frames, ignore_index=True)

    # Keep regular-season rows for the first version of the fantasy model.
    if "season_type" in combined.columns:
        combined = combined[combined["season_type"] == "REG"].copy()

    combined = combined.sort_values(
        [c for c in ["season", "week", "player_id", "team"] if c in combined.columns]
    ).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    destination = PROCESSED_DIR / output_name
    combined.to_csv(destination, index=False)

    print(f"WRITE {destination.name:<32} rows={len(combined):,}")
    return combined


def main() -> None:
    print("Combining player weekly data...")
    players = combine_yearly_files(
        "stats_player_week",
        f"player_weekly_{START_SEASON}_{END_SEASON}.csv",
    )

    print("\nCombining team weekly data...")
    teams = combine_yearly_files(
        "stats_team_week",
        f"team_weekly_{START_SEASON}_{END_SEASON}.csv",
    )

    print("\nCombining snap count data...")
    snaps = combine_yearly_files(
        "snap_counts",
        f"snap_counts_{START_SEASON}_{END_SEASON}.csv",
    )

    print("\nDone.")
    print(f"Player rows: {len(players):,}")
    print(f"Team rows:   {len(teams):,}")
    print(f"Snap rows:   {len(snaps):,}")
    print(f"Processed data: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
