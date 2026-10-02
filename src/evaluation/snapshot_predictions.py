"""Save an immutable, timestamped snapshot of all four pregame candidate pools.

Run automatically after run_weekly.py, or manually just after the four
prediction scripts. Only games with an explicit, future kickoff (Eastern
nflverse schedule time) and no recorded final score are captured.
Never reconstruct "pregame" projections from postgame model outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256

import pandas as pd

from tracker_common import (
    POSITIONS, SNAPSHOT_DIR, candidate_file, load_schedule,
    model_file, normalize_team, utc_now,
)


def file_sha256(path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_position(
    position: str,
    *,
    now: datetime,
    max_age_minutes: float,
) -> pd.DataFrame:
    path = candidate_file(position)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run 'py run_weekly.py' before snapshotting."
        )
    model = model_file(position)
    if not model.exists():
        raise FileNotFoundError(
            f"Missing {model}. Train the {position} v2 model first."
        )

    written = datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    age_minutes = (now - written).total_seconds() / 60
    if age_minutes < -5 or age_minutes > max_age_minutes:
        raise RuntimeError(
            f"{path.name} was written {age_minutes:.1f} minutes ago. "
            "Generate fresh predictions before taking a snapshot. "
            "Refusing to relabel stale predictions as current."
        )

    frame = pd.read_csv(
        path, low_memory=False, dtype={"player_id": str}
    )
    required = [
        "player_id", "player_name", "season", "week", "team",
        "opponent", "gridironiq_projection",
        POSITIONS[position]["probability_column"],
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise RuntimeError(
            f"Missing {position} prediction fields: {missing}"
        )
    if frame.empty:
        raise RuntimeError(f"No {position} future candidates found.")

    frame["season"] = pd.to_numeric(
        frame["season"], errors="raise"
    ).astype(int)
    frame["week"] = pd.to_numeric(
        frame["week"], errors="raise"
    ).astype(int)
    frame["team"] = normalize_team(frame["team"])
    frame["opponent"] = normalize_team(frame["opponent"])
    # The prediction scripts write all upcoming weeks; take only
    # the earliest available season/week for the weekly snapshot.
    first = (
        frame[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .iloc[0]
    )
    frame = frame.loc[
        frame["season"].eq(first["season"])
        & frame["week"].eq(first["week"])
    ].copy()

    frame["position"] = position
    frame["role_probability"] = pd.to_numeric(
        frame[POSITIONS[position]["probability_column"]],
        errors="coerce",
    )
    frame["projection"] = pd.to_numeric(
        frame["gridironiq_projection"], errors="coerce"
    )
    frame["model_sha256"] = file_sha256(model)
    frame["prediction_file_written_utc"] = written.isoformat()
    frame["role_definition"] = POSITIONS[position]["role_name"]
    frame["role_threshold"] = POSITIONS[position]["role_threshold"]
    if position == "QB":
        frame["selected_team_qb"] = pd.to_numeric(
            frame.get("selected_team_qb", 0),
            errors="coerce",
        ).fillna(0).astype(int)
    else:
        frame["selected_team_qb"] = pd.NA

    return frame


def take_snapshot(max_age_minutes: float = 60.0) -> pd.DataFrame:
    if max_age_minutes <= 0:
        raise ValueError("--max-age-minutes must be positive.")

    now = utc_now()
    schedule = load_schedule()
    frames = [
        load_position(
            position, now=now, max_age_minutes=max_age_minutes
        )
        for position in POSITIONS
    ]

    weeks = {
        (int(frame["season"].iloc[0]), int(frame["week"].iloc[0]))
        for frame in frames
    }
    if len(weeks) != 1:
        raise RuntimeError(
            "Position models disagree on upcoming season/week: "
            f"{sorted(weeks)}. Refresh all four positions together."
        )

    candidates = pd.concat(frames, ignore_index=True)
    keys = [
        "position", "player_id", "season", "week", "team",
        "opponent",
    ]
    if candidates.duplicated(keys).any():
        raise RuntimeError(
            "Duplicate player/game entries in prediction files. "
            "Resolve these before snapshotting."
        )
    candidates = candidates.merge(
        schedule,
        how="left",
        on=["season", "week", "team", "opponent"],
        validate="many_to_one",
        indicator=True,
    )
    unmatched = candidates["_merge"].ne("both")
    if unmatched.any():
        raise RuntimeError(
            "Predictions cannot be matched to schedule: "
            + str(
                candidates.loc[unmatched, keys]
                .head(8).to_dict("records")
            )
        )
    candidates = candidates.drop(columns="_merge")

    bad_predictions = (
        candidates["projection"].isna()
        | candidates["role_probability"].isna()
    )
    if bad_predictions.any():
        raise RuntimeError(
            "Snapshot has nonnumeric projections or role probabilities."
        )
    if (
        candidates["role_probability"].lt(0)
        | candidates["role_probability"].gt(1)
    ).any():
        raise RuntimeError("Role probabilities must be in [0, 1].")

    before_kickoff = (
        candidates["kickoff_utc"].notna()
        & candidates["kickoff_utc"].gt(pd.Timestamp(now))
        & ~candidates["game_completed"]
    )
    excluded = candidates.loc[~before_kickoff]
    if len(excluded):
        reasons = (
            excluded.assign(
                reason=excluded.apply(
                    lambda row: (
                        "kickoff missing"
                        if pd.isna(row["kickoff_utc"])
                        else "game completed"
                        if row["game_completed"]
                        else "kickoff has passed"
                    ),
                    axis=1,
                )
            )["reason"]
            .value_counts()
            .to_dict()
        )
        print(
            f"Excluded {len(excluded)} non-pregame candidates: {reasons}"
        )

    candidates = candidates.loc[before_kickoff].copy()
    if candidates.empty:
        print(
            "No verified pregame candidates; no snapshot created. "
            "Check schedule gametime/timezone and the next matchup."
        )
        return candidates

    # Rank only games still eligible at THIS snapshot time.
    candidates = candidates.sort_values(
        ["position", "projection", "player_id"],
        ascending=[True, False, True],
    ).copy()
    candidates["rank_at_capture"] = (
        candidates.groupby("position").cumcount() + 1
    )

    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    candidates["snapshot_id"] = stamp
    candidates["captured_at_utc"] = now.isoformat()
    candidates["kickoff_utc"] = candidates[
        "kickoff_utc"
    ].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    output_columns = [
        "snapshot_id", "captured_at_utc",
        "prediction_file_written_utc", "model_sha256",
        "season", "week", "game_id", "kickoff_utc", "position",
        "player_id", "player_name", "team", "opponent",
        "rank_at_capture", "projection",
        "conditional_fantasy_points", "role_probability",
        "role_definition", "role_threshold", "selected_team_qb",
        "prediction_low_80", "prediction_high_80",
        "avg_fp_last_3",
    ]
    for col in output_columns:
        if col not in candidates.columns:
            candidates[col] = pd.NA
    output = candidates[output_columns].copy()

    season, week = next(iter(weeks))
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    destination = (
        SNAPSHOT_DIR
        / f"{season}_w{week:02d}_{stamp}.csv"
    )
    # Exclusive-create: existing snapshots are never overwritten.
    with destination.open("x", encoding="utf-8", newline="") as handle:
        output.to_csv(handle, index=False)

    print("GRIDIRONIQ PROSPECTIVE SNAPSHOT")
    print("=" * 72)
    print(f"Captured (UTC): {now.isoformat()}")
    print(f"Season {season} | Week {week}")
    print(
        "Valid pregame candidates: "
        + ", ".join(
            f"{pos}={int(output['position'].eq(pos).sum())}"
            for pos in POSITIONS
        )
    )
    print(f"Immutable snapshot: {destination}")
    print(
        "These snapshots are local/ignored by Git. "
        "Back up data/snapshots/ if you change computers."
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture all four GridironIQ pregame candidate pools."
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=60.0,
        help="Maximum allowed age of prediction CSVs (default: 60).",
    )
    args = parser.parse_args()
    take_snapshot(max_age_minutes=args.max_age_minutes)


if __name__ == "__main__":
    main()
