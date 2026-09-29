"""Download the raw nflverse datasets used by GridironIQ.

Phase 1 downloads:
- weekly player stats, 2021-2026
- weekly team stats, 2021-2026
- weekly snap counts, 2021-2026
- game/schedule context (rest, venue/weather, betting lines)
- weekly injury reports, 2021-2026
- player metadata

The files are stored under data/raw/ and are ignored by Git.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import requests


NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
DEFAULT_START_SEASON = 2021
DEFAULT_END_SEASON = 2026
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool = False,
) -> None:
    """Download one file, optionally replacing an existing local copy."""
    if destination.exists() and not force:
        print(f"SKIP  {destination.name} already exists")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    action = "REFRESH" if destination.exists() else "GET"
    print(f"{action:<7}{destination.name}")

    temp_path = destination.with_suffix(destination.suffix + ".tmp")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with temp_path.open("wb") as file_handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)

    temp_path.replace(destination)


def player_stats_url(season: int) -> str:
    return (
        f"{NFLVERSE_RELEASE}/stats_player/"
        f"stats_player_week_{season}.csv"
    )


def team_stats_url(season: int) -> str:
    return (
        f"{NFLVERSE_RELEASE}/stats_team/"
        f"stats_team_week_{season}.csv"
    )


def snap_counts_url(season: int) -> str:
    return (
        f"{NFLVERSE_RELEASE}/snap_counts/"
        f"snap_counts_{season}.csv"
    )


def schedules_url() -> str:
    # nflverse schedule source maintained in the nfldata repository.
    return "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def depth_charts_url(season: int) -> str:
    return (
        f"{NFLVERSE_RELEASE}/depth_charts/"
        f"depth_charts_{season}.csv"
    )


def injuries_url(season: int) -> str:
    return (
        f"{NFLVERSE_RELEASE}/injuries/"
        f"injuries_{season}.csv"
    )


def players_url() -> str:
    return f"{NFLVERSE_RELEASE}/players/players.csv"


def download_datasets(start_season: int, end_season: int) -> None:
    if start_season > end_season:
        raise ValueError("start_season cannot be greater than end_season")

    for season in range(start_season, end_season + 1):
        # Current-season nflverse files change every week. Historical files
        # are effectively immutable, so keep the fast skip behavior for them.
        refresh = season == DEFAULT_END_SEASON

        download_file(
            player_stats_url(season),
            RAW_DIR / f"stats_player_week_{season}.csv",
            force=refresh,
        )
        download_file(
            team_stats_url(season),
            RAW_DIR / f"stats_team_week_{season}.csv",
            force=refresh,
        )
        download_file(
            snap_counts_url(season),
            RAW_DIR / f"snap_counts_{season}.csv",
            force=refresh,
        )
        download_file(
            depth_charts_url(season),
            RAW_DIR / f"depth_charts_{season}.csv",
            force=refresh,
        )
        download_file(
            injuries_url(season),
            RAW_DIR / f"injuries_{season}.csv",
            force=refresh,
        )

    # Schedule scores/statuses also change during the live season.
    download_file(
        schedules_url(),
        RAW_DIR / "games.csv",
        force=True,
    )
    download_file(players_url(), RAW_DIR / "players.csv")
    print(f"\nDone. Raw data is in: {RAW_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GridironIQ nflverse data")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=DEFAULT_END_SEASON)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_datasets(args.start_season, args.end_season)
