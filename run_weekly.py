"""Run the complete weekly GridironIQ QB v2 workflow.

Default weekly workflow:
1. Refresh live nflverse data.
2. Rebuild processed historical tables.
3. Rebuild the pregame QB v2 candidate dataset.
4. Load the saved QB v2 model bundle and generate current-week rankings.

Use --retrain when you intentionally want to refit the official v2 models
before generating rankings.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def run_step(label: str, script: Path) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)

    start = time.perf_counter()

    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
    )

    elapsed = time.perf_counter() - start
    print(f"\nDONE: {label} ({elapsed:.1f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the weekly GridironIQ QB v2 pipeline."
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help=(
            "Retrain the official QB v2 models on 2021-2025 before "
            "producing weekly rankings."
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the current local raw data instead of refreshing nflverse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("GRIDIRONIQ WEEKLY PIPELINE")
    print("=" * 78)
    print(f"Python: {sys.executable}")
    print(f"Project: {ROOT}")
    print(f"Retrain models: {'YES' if args.retrain else 'NO'}")

    if not args.skip_download:
        run_step(
            "STEP 1/4 — Refresh live NFL data",
            ROOT / "src" / "data" / "download_nfl_data.py",
        )
    else:
        print("\nSTEP 1/4 — Refresh live NFL data: SKIPPED")

    run_step(
        "STEP 2/4 — Rebuild processed historical tables",
        ROOT / "src" / "data" / "build_historical_tables.py",
    )

    run_step(
        "STEP 3/4 — Build QB v2 pregame candidate dataset",
        ROOT / "src" / "features" / "build_qb_v2_candidate_dataset.py",
    )

    if args.retrain:
        run_step(
            "OPTIONAL — Retrain official QB v2 models",
            ROOT / "src" / "models" / "train_qb_v2.py",
        )

    run_step(
        "STEP 4/4 — Generate current-week QB rankings",
        ROOT / "src" / "models" / "predict_qb_v2.py",
    )

    print("\n" + "=" * 78)
    print("GRIDIRONIQ WEEKLY PIPELINE COMPLETE")
    print("=" * 78)
    print(
        "Weekly rankings: data/processed/qb_v2_weekly_rankings.csv"
    )


if __name__ == "__main__":
    main()
