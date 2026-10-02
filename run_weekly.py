"""Run the complete weekly GridironIQ QB + RB + WR + TE workflow.

Default weekly workflow:
1. Refresh live nflverse data.
2. Rebuild processed historical tables.
3. Rebuild QB v2 pregame candidate dataset.
4. Rebuild RB v2 pregame candidate dataset.
5. Rebuild WR v2 pregame candidate dataset.
6. Rebuild TE v2 pregame candidate dataset.
7. Generate current-week QB rankings from saved models.
8. Generate current-week RB rankings from saved models.
9. Generate current-week WR rankings from saved models.
10. Generate current-week TE rankings from saved models.
11. Save an immutable prekickoff snapshot of all four candidate pools.

Use --retrain when you intentionally want to refit the official QB, RB,
WR, and TE v2 models before generating rankings.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run_step(
    label: str,
    script: Path,
) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    start = time.perf_counter()

    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
    )

    elapsed = (
        time.perf_counter()
        - start
    )
    print(
        f"\nDONE: {label} "
        f"({elapsed:.1f}s)"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the weekly GridironIQ "
            "QB + RB + WR + TE v2 pipeline."
        )
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Retrain the official QB, RB, "
            "WR and TE v2 models on 2021-2025 "
            "before producing weekly rankings."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help=(
            "Use the current local raw data "
            "instead of refreshing nflverse."
        ),
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help=(
            "Skip saving a timestamped prekickoff prediction "
            "snapshot (automatic by default)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "GRIDIRONIQ WEEKLY PIPELINE"
    )
    print("=" * 78)
    print(
        f"Python: {sys.executable}"
    )
    print(
        f"Project: {ROOT}"
    )
    print(
        "Positions: QB, RB, WR, TE"
    )
    print(
        "Retrain models: "
        f"{'YES' if args.retrain else 'NO'}"
    )

    if not args.skip_download:
        run_step(
            "STEP 1/11 — Refresh live NFL data",
            ROOT
            / "src"
            / "data"
            / "download_nfl_data.py",
        )
    else:
        print(
            "\nSTEP 1/11 — Refresh live NFL "
            "data: SKIPPED"
        )

    run_step(
        "STEP 2/11 — Rebuild processed "
        "historical tables",
        ROOT
        / "src"
        / "data"
        / "build_historical_tables.py",
    )

    run_step(
        "STEP 3/11 — Build QB v2 pregame "
        "candidate dataset",
        ROOT
        / "src"
        / "features"
        / "build_qb_v2_candidate_dataset.py",
    )

    run_step(
        "STEP 4/11 — Build RB v2 pregame "
        "candidate dataset",
        ROOT
        / "src"
        / "features"
        / "build_rb_v2_candidate_dataset.py",
    )

    run_step(
        "STEP 5/11 — Build WR v2 pregame "
        "candidate dataset",
        ROOT
        / "src"
        / "features"
        / "build_wr_v2_candidate_dataset.py",
    )

    run_step(
        "STEP 6/11 — Build TE v2 pregame "
        "candidate dataset",
        ROOT
        / "src"
        / "features"
        / "build_te_v2_candidate_dataset.py",
    )

    if args.retrain:
        run_step(
            "OPTIONAL — Retrain official "
            "QB v2 models",
            ROOT
            / "src"
            / "models"
            / "train_qb_v2.py",
        )
        run_step(
            "OPTIONAL — Retrain official "
            "RB v2 models",
            ROOT
            / "src"
            / "models"
            / "train_rb_v2.py",
        )
        run_step(
            "OPTIONAL — Retrain official "
            "WR v2 models",
            ROOT
            / "src"
            / "models"
            / "train_wr_v2.py",
        )
        run_step(
            "OPTIONAL — Retrain official "
            "TE v2 models",
            ROOT
            / "src"
            / "models"
            / "train_te_v2.py",
        )

    run_step(
        "STEP 7/11 — Generate current-week "
        "QB rankings",
        ROOT
        / "src"
        / "models"
        / "predict_qb_v2.py",
    )

    run_step(
        "STEP 8/11 — Generate current-week "
        "RB rankings",
        ROOT
        / "src"
        / "models"
        / "predict_rb_v2.py",
    )

    run_step(
        "STEP 9/11 — Generate current-week "
        "WR rankings",
        ROOT
        / "src"
        / "models"
        / "predict_wr_v2.py",
    )

    run_step(
        "STEP 10/11 — Generate current-week "
        "TE rankings",
        ROOT
        / "src"
        / "models"
        / "predict_te_v2.py",
    )

    if not args.no_snapshot:
        run_step(
            "STEP 11/11 — Preserve prekickoff predictions",
            ROOT
            / "src"
            / "evaluation"
            / "snapshot_predictions.py",
        )
    else:
        print(
            "\\nSTEP 11/11 — Preserve prekickoff "
            "predictions: SKIPPED (requested)"
        )

    print(
        "\n" + "=" * 78
    )
    print(
        "GRIDIRONIQ WEEKLY PIPELINE COMPLETE"
    )
    print("=" * 78)
    print(
        "QB rankings: "
        "data/processed/"
        "qb_v2_weekly_rankings.csv"
    )
    print(
        "RB rankings: "
        "data/processed/"
        "rb_v2_weekly_rankings.csv"
    )
    print(
        "WR rankings: "
        "data/processed/"
        "wr_v2_weekly_rankings.csv"
    )
    print(
        "TE rankings: "
        "data/processed/"
        "te_v2_weekly_rankings.csv"
    )
    print(
        "Pregame snapshots: "
        "data/snapshots/ (local, immutable)"
    )
    print(
        "After games finish, evaluate with: "
        "py src\\\\evaluation\\\\evaluate_predictions.py"
    )


if __name__ == "__main__":
    main()
