"""Evaluate only genuine prekickoff GridironIQ prediction snapshots.

Never regenerate old projections using today's trained model. The tracker
uses the latest VERIFIED prekickoff prediction per player/game by default.
It scores only games with final schedule scores and available team-boxscore
data, with the same custom fantasy function as the modeling pipeline.

Pairwise accuracy is a proxy for start/sit ranking quality among the
projected top K; it does not represent actual user lineup decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from tracker_common import (
    POSITIONS, PROCESSED_DIR, RAW_DIR, REPORT_DIR,
    SNAPSHOT_DIR, load_schedule, normalize_team,
)

# Use the project's original scoring function; do not duplicate or change it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "features"))
from build_qb_model_dataset import custom_fantasy_points  # noqa: E402

KEYS = [
    "position", "player_id", "season", "week", "team", "opponent",
]
ACTUAL_KEYS = ["player_id", "season", "week", "team"]


def read_snapshots(selection: str) -> pd.DataFrame:
    paths = sorted(SNAPSHOT_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No pregame snapshots in {SNAPSHOT_DIR}. "
            "Run py run_weekly.py before kickoff to create one."
        )
    frames = []
    required = [
        *KEYS, "captured_at_utc", "projection",
        "role_probability", "role_threshold",
        "model_sha256", "prediction_low_80",
        "prediction_high_80",
    ]
    for path in paths:
        frame = pd.read_csv(
            path, low_memory=False, dtype={"player_id": str}
        )
        missing = [c for c in required if c not in frame.columns]
        if missing:
            raise RuntimeError(
                f"Snapshot {path.name} is missing {missing}."
            )
        frame["snapshot_file"] = path.name
        frames.append(frame)

    snapshots = pd.concat(frames, ignore_index=True)
    snapshots["season"] = pd.to_numeric(
        snapshots["season"], errors="raise"
    ).astype(int)
    snapshots["week"] = pd.to_numeric(
        snapshots["week"], errors="raise"
    ).astype(int)
    snapshots["team"] = normalize_team(snapshots["team"])
    snapshots["opponent"] = normalize_team(snapshots["opponent"])
    for c in [
        "projection", "role_probability", "role_threshold",
        "prediction_low_80", "prediction_high_80",
        "avg_fp_last_3",
    ]:
        if c in snapshots.columns:
            snapshots[c] = pd.to_numeric(
                snapshots[c], errors="coerce"
            )
    snapshots["captured_at_utc"] = pd.to_datetime(
        snapshots["captured_at_utc"], utc=True, errors="coerce"
    )
    snapshots["prediction_file_written_utc"] = pd.to_datetime(
        snapshots["prediction_file_written_utc"],
        utc=True,
        errors="coerce",
    )
    if snapshots["captured_at_utc"].isna().any():
        raise RuntimeError("A snapshot has an invalid capture time.")
    if snapshots["prediction_file_written_utc"].isna().any():
        raise RuntimeError(
            "A snapshot is missing the original prediction file time."
        )
    if snapshots.duplicated(KEYS + ["captured_at_utc"]).any():
        raise RuntimeError(
            "Duplicate candidates recorded at the same snapshot time."
        )

    # Retain the originally advertised kickoff for provenance, but use
    # the refreshed authoritative schedule for final-game validation.
    # Dropping the original duplicate columns prevents _x/_y suffixes.
    snapshots["captured_schedule_kickoff_utc"] = (
        snapshots["kickoff_utc"]
    )
    snapshots = snapshots.drop(
        columns=["kickoff_utc", "game_id"],
        errors="ignore",
    )
    schedule = load_schedule()
    snapshots = snapshots.merge(
        schedule,
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if snapshots["_merge"].ne("both").any():
        raise RuntimeError(
            "One or more saved predictions have no matching game "
            "in the refreshed regular-season schedule."
        )
    snapshots = snapshots.drop(columns="_merge")

    valid_time = (
        snapshots["kickoff_utc"].notna()
        & snapshots["captured_at_utc"].lt(snapshots["kickoff_utc"])
        & snapshots["prediction_file_written_utc"].le(
            snapshots["captured_at_utc"]
        )
    )
    invalid = int((~valid_time).sum())
    if invalid:
        print(
            f"Ignoring {invalid} snapshots not independently verified "
            "as prekickoff."
        )
    snapshots = snapshots.loc[valid_time].copy()
    if snapshots.empty:
        raise RuntimeError("No verified prekickoff snapshots remain.")

    snapshots = snapshots.sort_values(
        ["captured_at_utc", "snapshot_file"],
        ascending=[selection == "earliest", selection == "earliest"],
    ).drop_duplicates(KEYS, keep="first")

    # Compute a single retrospective evaluation cohort using ONLY
    # saved pregame projections. Do this BEFORE filtering final games.
    snapshots = snapshots.sort_values(
        ["season", "week", "position", "projection", "player_id"],
        ascending=[True, True, True, False, True],
    ).copy()
    snapshots["evaluation_rank"] = (
        snapshots.groupby(["season", "week", "position"])
        .cumcount() + 1
    )
    snapshots["starter_cohort"] = snapshots.apply(
        lambda row: int(
            row["evaluation_rank"]
            <= POSITIONS[row["position"]]["starter_count"]
        ),
        axis=1,
    )
    return snapshots


def load_actual_stats(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load real player fantasy points and independently measured snap roles."""
    seasons = set(snapshots["season"].unique())
    players_path = PROCESSED_DIR / "player_weekly_2021_2026.csv"
    teams_path = PROCESSED_DIR / "team_weekly_2021_2026.csv"
    snaps_path = PROCESSED_DIR / "snap_counts_2021_2026.csv"
    meta_path = RAW_DIR / "players.csv"
    for path in [
        players_path, teams_path, snaps_path, meta_path
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run py run_weekly.py to refresh data."
            )

    teams = pd.read_csv(teams_path, low_memory=False)
    teams = teams[
        teams["season"].isin(seasons)
    ].copy()
    teams["team"] = normalize_team(teams["team"])
    box_coverage = teams[
        ["season", "week", "team"]
    ].drop_duplicates().assign(boxscore_available=1)

    players = pd.read_csv(
        players_path, low_memory=False, dtype={"player_id": str}
    )
    players = players[
        players["season"].isin(seasons)
    ].copy()
    players["team"] = normalize_team(players["team"])
    players["_fantasy_points"] = custom_fantasy_points(players)
    actual = (
        players.groupby(ACTUAL_KEYS, as_index=False)
        .agg(
            actual_fantasy_points=("_fantasy_points", "sum"),
        )
    )
    actual["had_stat_row"] = 1

    snaps = pd.read_csv(snaps_path, low_memory=False)
    snaps = snaps[
        snaps["season"].isin(seasons)
    ].copy()
    snaps["team"] = normalize_team(snaps["team"])
    snap_coverage = snaps[
        ["season", "week", "team"]
    ].drop_duplicates().assign(snap_team_data_available=1)
    snaps["offense_snaps"] = pd.to_numeric(
        snaps["offense_snaps"], errors="coerce"
    ).fillna(0.0)
    snaps["offense_pct"] = pd.to_numeric(
        snaps["offense_pct"], errors="coerce"
    )
    snaps["offense_pct"] = snaps["offense_pct"].where(
        snaps["offense_pct"].le(1.0),
        snaps["offense_pct"] / 100.0,
    )

    meta = pd.read_csv(
        meta_path, low_memory=False,
        dtype={"gsis_id": str, "pfr_id": str},
    )
    ids = meta[
        ["pfr_id", "gsis_id"]
    ].dropna().drop_duplicates()
    ambiguous = ids.duplicated("pfr_id", keep=False)
    if ambiguous.any():
        ids = ids.loc[
            ~ids["pfr_id"].isin(
                ids.loc[ambiguous, "pfr_id"]
            )
        ].copy()

    snaps = snaps.merge(
        ids,
        left_on="pfr_player_id",
        right_on="pfr_id",
        how="left",
        validate="many_to_one",
    ).rename(columns={"gsis_id": "player_id"})
    snap_actual = (
        snaps.dropna(subset=["player_id"])
        .groupby(ACTUAL_KEYS, as_index=False)
        .agg(
            offense_pct=("offense_pct", "max"),
            offense_snaps=("offense_snaps", "max"),
        )
    )
    snap_actual["has_snap_row"] = 1
    return (
        actual,
        snap_actual,
        box_coverage.merge(
            snap_coverage,
            on=["season", "week", "team"],
            how="left",
            validate="one_to_one",
        ),
    )


def attach_actuals(
    snapshots: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Never infer missing scores as zeros unless the team boxscore exists."""
    completed = snapshots[
        snapshots["game_completed"].eq(True)
    ].copy()
    pending = snapshots[
        snapshots["game_completed"].ne(True)
    ].copy()
    if completed.empty:
        return completed, pending

    actual, snaps, coverage = load_actual_stats(snapshots)
    completed = completed.merge(
        coverage,
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    no_box = completed["boxscore_available"].isna()
    if no_box.any():
        missing = completed.loc[no_box].copy()
        missing["not_evaluated_reason"] = (
            "final score posted but team boxscore not downloaded"
        )
        pending = pd.concat([pending, missing], ignore_index=True)
        completed = completed.loc[~no_box].copy()
        print(
            f"Deferred {len(missing)} candidates: "
            "completed schedule without a team boxscore."
        )
    if completed.empty:
        return completed, pending

    completed = completed.merge(
        actual,
        on=ACTUAL_KEYS,
        how="left",
        validate="many_to_one",
    )
    completed = completed.merge(
        snaps,
        on=ACTUAL_KEYS,
        how="left",
        validate="many_to_one",
    )
    completed["had_stat_row"] = completed[
        "had_stat_row"
    ].fillna(0).astype(int)
    completed["actual_fantasy_points"] = completed[
        "actual_fantasy_points"
    ].fillna(0.0)

    # An absent snap row means zero ONLY if the team snap feed exists
    # and the player also has no stat line. Otherwise role is unknown.
    known_zero = (
        completed["has_snap_row"].isna()
        & completed["had_stat_row"].eq(0)
        & completed["snap_team_data_available"].eq(1)
    )
    completed.loc[known_zero, "offense_pct"] = 0.0
    completed.loc[known_zero, "offense_snaps"] = 0.0
    completed["actual_role"] = np.where(
        completed["offense_pct"].notna(),
        (
            completed["offense_pct"]
            >= completed["role_threshold"]
        ).astype(float),
        np.nan,
    )
    completed["played_any_snap"] = np.where(
        completed["offense_snaps"].notna(),
        (
            completed["offense_snaps"].gt(0)
            | completed["had_stat_row"].eq(1)
        ).astype(float),
        np.where(
            completed["had_stat_row"].eq(1),
            1.0,
            np.nan,
        ),
    )
    completed["error"] = (
        completed["projection"]
        - completed["actual_fantasy_points"]
    )
    completed["absolute_error"] = completed["error"].abs()
    completed["squared_error"] = completed["error"] ** 2
    has_bounds = (
        completed["prediction_low_80"].notna()
        & completed["prediction_high_80"].notna()
    )
    completed["interval_covered"] = np.where(
        has_bounds,
        (
            completed["actual_fantasy_points"]
            .ge(completed["prediction_low_80"])
            & completed["actual_fantasy_points"]
            .le(completed["prediction_high_80"])
        ).astype(float),
        np.nan,
    )
    completed["role_brier"] = np.where(
        completed["actual_role"].notna(),
        (
            completed["role_probability"]
            - completed["actual_role"]
        ) ** 2,
        np.nan,
    )
    completed["role_correct_50pct"] = np.where(
        completed["actual_role"].notna(),
        (
            completed["role_probability"].ge(0.50)
            == completed["actual_role"].eq(1)
        ).astype(float),
        np.nan,
    )
    baseline_ok = completed["avg_fp_last_3"].notna()
    completed["last3_absolute_error"] = np.where(
        baseline_ok,
        (
            completed["avg_fp_last_3"]
            - completed["actual_fantasy_points"]
        ).abs(),
        np.nan,
    )
    completed["model_mae_on_baseline_cohort"] = np.where(
        baseline_ok,
        completed["absolute_error"],
        np.nan,
    )
    return completed, pending


def pairwise_top_starters(group: pd.DataFrame) -> tuple[int, float]:
    """Ordering within completed top-K players from the SAME week only.

    Pairwise ranking is a lineup-quality proxy, not recorded user starts.
    Cumulative results pool individual week pairs without comparing
    players across different matchups or weeks.
    """
    correct = 0
    count = 0
    subset = group[
        group["starter_cohort"].eq(1)
    ].copy()
    for _, week_frame in subset.groupby(["season", "week"]):
        preds = week_frame["projection"].to_numpy(dtype=float)
        actual = week_frame[
            "actual_fantasy_points"
        ].to_numpy(dtype=float)
        for i in range(len(week_frame)):
            for j in range(i + 1, len(week_frame)):
                pred_diff = preds[i] - preds[j]
                true_diff = actual[i] - actual[j]
                if pred_diff == 0 or true_diff == 0:
                    continue
                count += 1
                correct += int(
                    (pred_diff > 0) == (true_diff > 0)
                )
    return count, (
        correct / count if count else float("nan")
    )

def metric_row(
    group: pd.DataFrame,
    position: str,
    season: int | str,
    week: int | str,
) -> dict:
    starter = group[
        group["starter_cohort"].eq(1)
    ]
    count_pairs, pair_accuracy = pairwise_top_starters(group)
    return {
        "season": season,
        "week": week,
        "position": position,
        "evaluated": len(group),
        "starter_evaluated": len(starter),
        "mae": group["absolute_error"].mean(),
        "rmse": float(np.sqrt(group["squared_error"].mean())),
        "signed_error_pred_minus_actual": group["error"].mean(),
        "starter_mae": starter["absolute_error"].mean(),
        "role_known": int(group["actual_role"].notna().sum()),
        "role_brier": group["role_brier"].mean(),
        "role_accuracy_at_50pct": group[
            "role_correct_50pct"
        ].mean(),
        "interval_rows": int(
            group["interval_covered"].notna().sum()
        ),
        "interval_coverage": group["interval_covered"].mean(),
        "last3_rows": int(
            group["last3_absolute_error"].notna().sum()
        ),
        "last3_mae": group["last3_absolute_error"].mean(),
        "model_mae_same_last3_rows": group[
            "model_mae_on_baseline_cohort"
        ].mean(),
        "starter_pair_count": count_pairs,
        "starter_pairwise_accuracy": pair_accuracy,
    }


def role_calibration(completed: pd.DataFrame) -> pd.DataFrame:
    valid = completed[
        completed["actual_role"].notna()
        & completed["role_probability"].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame()
    valid["probability_bin"] = pd.cut(
        valid["role_probability"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.00000001],
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        include_lowest=True,
        right=False,
    )
    return (
        valid.groupby(
            ["position", "probability_bin"],
            observed=True,
        )
        .agg(
            rows=("actual_role", "size"),
            mean_probability=("role_probability", "mean"),
            observed_role_rate=("actual_role", "mean"),
            brier=("role_brier", "mean"),
        )
        .reset_index()
    )


def evaluate(selection: str = "latest") -> None:
    print("GRIDIRONIQ PROSPECTIVE PERFORMANCE TRACKER")
    print("=" * 84)
    print(f"Snapshot selection: {selection} verified prekickoff per player/game")
    snapshots = read_snapshots(selection)
    completed, pending = attach_actuals(snapshots)
    print(f"Valid snapshotted candidates: {len(snapshots):,}")
    print(f"Finished with downloaded boxscore: {len(completed):,}")
    print(f"Pending or awaiting boxscore:       {len(pending):,}")
    if completed.empty:
        print(
            "No completed, scorable games yet. "
            "Refresh after the relevant games finish; "
            "saved snapshots remain untouched."
        )
        return

    weekly_rows = []
    for (season, week, position), group in completed.groupby(
        ["season", "week", "position"]
    ):
        weekly_rows.append(
            metric_row(group, position, int(season), int(week))
        )
    weekly = pd.DataFrame(weekly_rows)
    cumulative = pd.DataFrame(
        [
            metric_row(
                group, position, "ALL", "ALL"
            )
            for position, group in completed.groupby("position")
        ]
    )
    calibration = role_calibration(completed)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    files = {
        "prediction_results.csv": completed,
        "weekly_summary.csv": weekly,
        "cumulative_summary.csv": cumulative,
        "role_calibration.csv": calibration,
    }
    for name, frame in files.items():
        frame.to_csv(REPORT_DIR / name, index=False)

    cols = [
        "season", "week", "position", "evaluated",
        "starter_evaluated", "mae", "starter_mae",
        "role_known", "role_brier",
        "interval_rows", "interval_coverage",
        "starter_pair_count", "starter_pairwise_accuracy",
    ]
    print("\nWEEKLY PERFORMANCE")
    print(
        weekly[cols].sort_values(
            ["season", "week", "position"]
        ).to_string(
            index=False, float_format=lambda val: f"{val:.3f}"
        )
    )
    print("\nCUMULATIVE BY POSITION")
    print(
        cumulative[cols[2:]].sort_values(
            "position"
        ).to_string(
            index=False, float_format=lambda val: f"{val:.3f}"
        )
    )
    print("\nRole-calibration table: "
          + str(REPORT_DIR / "role_calibration.csv"))
    print(
        "Starter cohorts: QB top 12, RB/WR top 24, TE top 12 "
        "from saved pregame projections. "
        "Pairwise accuracy compares completed starter-cohort "
        "player pairs and excludes ties; it does not measure "
        "the user's actual start/sit decisions."
    )
    print(
        "Missing snap coverage is excluded from role metrics. "
        "Historical 80% interval coverage on these live games "
        "is measured independently of its calibration sample."
    )
    print(f"Reports written to: {REPORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score immutable GridironIQ prekickoff snapshots."
    )
    parser.add_argument(
        "--selection",
        choices=["latest", "earliest"],
        default="latest",
        help="Select most recent or first verified prekickoff forecast.",
    )
    args = parser.parse_args()
    evaluate(selection=args.selection)


if __name__ == "__main__":
    main()
