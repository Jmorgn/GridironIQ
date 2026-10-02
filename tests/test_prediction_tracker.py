"""Offline contract tests for GridironIQ's prospective prediction tracker.

Run from repo root:
    py -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

import evaluate_predictions as scoring  # noqa: E402
import snapshot_predictions as capture  # noqa: E402
from tracker_common import POSITIONS, kickoff_utc  # noqa: E402

NOW = datetime(2026, 10, 2, 19, 0, tzinfo=timezone.utc)
KICKOFF = pd.Timestamp("2026-10-04T17:00:00Z")
PAST = pd.Timestamp("2026-10-01T23:00:00Z")


def mock_schedule() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "season": 2026, "week": 4,
            "team": "ARI", "opponent": "NYG",
            "game_id": "future", "kickoff_utc": KICKOFF,
            "game_completed": False,
        },
        {
            "season": 2026, "week": 4,
            "team": "PIT", "opponent": "CLE",
            "game_id": "past", "kickoff_utc": PAST,
            "game_completed": True,
        },
    ])


class ScheduleTests(unittest.TestCase):
    def test_eastern_schedule_kickoff_becomes_utc(self):
        # The first Sunday in October is daylight saving time.
        kickoff = kickoff_utc("2026-10-04", "13:00")
        self.assertEqual(kickoff, KICKOFF)

    def test_unknown_kickoff_is_never_guessed(self):
        self.assertTrue(pd.isna(kickoff_utc("2026-10-04", "TBD")))
        self.assertTrue(pd.isna(kickoff_utc("2026-10-04", None)))


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.files = {}
        self.models = {}
        for position, config in POSITIONS.items():
            path = self.base / f"{position.lower()}_future.csv"
            model = self.base / f"{position.lower()}_v2_bundle.joblib"
            model.write_bytes(position.encode("utf-8"))
            self.models[position] = model
            self.files[position] = path
            probabilities = config["probability_column"]
            frame = pd.DataFrame([
                {
                    "season": 2026, "week": 4,
                    "player_id": f"{position}-good",
                    "player_name": f"{position} Good",
                    "team": "ARI", "opponent": "NYG",
                    "gridironiq_projection": 10.0,
                    probabilities: 0.8,
                    "selected_team_qb": 1 if position == "QB" else np.nan,
                    "prediction_low_80": 2.0,
                    "prediction_high_80": 17.0,
                    "avg_fp_last_3": 8.0,
                },
                {
                    "season": 2026, "week": 4,
                    "player_id": f"{position}-past",
                    "player_name": f"{position} Past",
                    "team": "PIT", "opponent": "CLE",
                    "gridironiq_projection": 20.0,
                    probabilities: 0.9,
                    "selected_team_qb": 1 if position == "QB" else np.nan,
                },
            ])
            frame.to_csv(path, index=False)
            generated = NOW.timestamp() - 30
            os.utime(path, (generated, generated))

        self.patchers = [
            patch.object(capture, "SNAPSHOT_DIR", self.base / "snapshots"),
            patch.object(
                capture, "candidate_file",
                side_effect=lambda pos: self.files[pos],
            ),
            patch.object(
                capture, "model_file",
                side_effect=lambda pos: self.models[pos],
            ),
            patch.object(capture, "load_schedule", side_effect=mock_schedule),
            patch.object(capture, "utc_now", return_value=NOW),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_capture_only_future_unfinished_games(self):
        result = capture.take_snapshot()
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result["position"]), set(POSITIONS))
        self.assertTrue(result["team"].eq("ARI").all())
        self.assertTrue(result["captured_at_utc"].eq(NOW.isoformat()).all())
        files = list((self.base / "snapshots").glob("*.csv"))
        self.assertEqual(len(files), 1)
        self.assertEqual(len(pd.read_csv(files[0])), 4)

        # Same ID/timestamp must not silently overwrite history.
        with self.assertRaises(FileExistsError):
            capture.take_snapshot()
        self.assertEqual(len(pd.read_csv(files[0])), 4)

    def test_stale_prediction_file_rejected(self):
        stale = NOW.timestamp() - 7200
        os.utime(self.files["QB"], (stale, stale))
        with self.assertRaisesRegex(RuntimeError, "stale predictions"):
            # The precise error says to refuse relabeling stale outputs.
            capture.take_snapshot()


class EvaluationTests(unittest.TestCase):
    def test_latest_snapshot_excludes_postkickoff_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            forecasts = [
                ("2026-10-02T19:00:00Z", 4.0),
                ("2026-10-03T18:00:00Z", 6.0),
                ("2026-10-04T18:00:00Z", 100.0),
            ]
            for index, (capture_time, projected) in enumerate(forecasts):
                captured = pd.Timestamp(capture_time)
                pd.DataFrame([{
                    "position": "TE",
                    "player_id": "TE-X",
                    "player_name": "Example TE",
                    "season": 2026,
                    "week": 4,
                    "team": "ARI",
                    "opponent": "NYG",
                    "game_id": "future",
                    "kickoff_utc": KICKOFF.isoformat(),
                    "captured_at_utc": captured.isoformat(),
                    "prediction_file_written_utc": (
                        captured - pd.Timedelta(minutes=1)
                    ).isoformat(),
                    "projection": projected,
                    "role_probability": 0.8,
                    "role_threshold": 0.5,
                    "model_sha256": "test-model",
                    "prediction_low_80": 1.0,
                    "prediction_high_80": 15.0,
                    "avg_fp_last_3": 5.0,
                }]).to_csv(folder / f"snapshot_{index}.csv", index=False)

            with patch.object(
                scoring, "SNAPSHOT_DIR", folder
            ), patch.object(
                scoring, "load_schedule", side_effect=mock_schedule
            ):
                latest = scoring.read_snapshots("latest")
                earliest = scoring.read_snapshots("earliest")

        self.assertEqual(len(latest), 1)
        self.assertEqual(float(latest["projection"].iloc[0]), 6.0)
        self.assertEqual(float(earliest["projection"].iloc[0]), 4.0)
        self.assertTrue(
            latest["captured_at_utc"].iloc[0] < KICKOFF
        )

    def test_pairwise_comparisons_do_not_cross_weeks(self):
        group = pd.DataFrame([
            {
                "season": 2026, "week": 4,
                "starter_cohort": 1,
                "projection": 10.0,
                "actual_fantasy_points": 9.0,
            },
            {
                "season": 2026, "week": 4,
                "starter_cohort": 1,
                "projection": 8.0,
                "actual_fantasy_points": 2.0,
            },
            {
                "season": 2026, "week": 5,
                "starter_cohort": 1,
                "projection": 10.0,
                "actual_fantasy_points": 0.0,
            },
            {
                "season": 2026, "week": 5,
                "starter_cohort": 1,
                "projection": 8.0,
                "actual_fantasy_points": 9.0,
            },
        ])
        count, accuracy = scoring.pairwise_top_starters(group)
        self.assertEqual(count, 2)
        self.assertEqual(accuracy, 0.5)

    def test_completed_game_without_stat_row_scores_zero(self):
        snapshot = pd.DataFrame([
            {
                "season": 2026, "week": 4, "position": "TE",
                "player_id": "TE-A", "team": "ARI",
                "opponent": "NYG", "game_completed": True,
                "projection": 7.0, "role_probability": 0.8,
                "role_threshold": 0.5,
                "prediction_low_80": 1.0,
                "prediction_high_80": 11.0,
                "avg_fp_last_3": 6.0,
                "starter_cohort": 1,
            },
            {
                "season": 2026, "week": 4, "position": "TE",
                "player_id": "TE-B", "team": "ARI",
                "opponent": "NYG", "game_completed": True,
                "projection": 1.0, "role_probability": 0.2,
                "role_threshold": 0.5,
                "prediction_low_80": 0.0,
                "prediction_high_80": 4.0,
                "avg_fp_last_3": 1.0,
                "starter_cohort": 0,
            },
        ])
        actual = pd.DataFrame([
            {
                "player_id": "TE-A", "season": 2026,
                "week": 4, "team": "ARI",
                "actual_fantasy_points": 6.0,
                "had_stat_row": 1,
            }
        ])
        snaps = pd.DataFrame([
            {
                "player_id": "TE-A", "season": 2026,
                "week": 4, "team": "ARI",
                "offense_pct": 0.4, "offense_snaps": 20,
                "has_snap_row": 1,
            }
        ])
        coverage = pd.DataFrame([
            {
                "season": 2026, "week": 4, "team": "ARI",
                "boxscore_available": 1,
                "snap_team_data_available": 1,
            }
        ])
        with patch.object(
            scoring, "load_actual_stats",
            return_value=(actual, snaps, coverage),
        ):
            result, pending = scoring.attach_actuals(snapshot)
        self.assertEqual(len(pending), 0)
        self.assertEqual(len(result), 2)
        a = result.loc[result["player_id"].eq("TE-A")].iloc[0]
        b = result.loc[result["player_id"].eq("TE-B")].iloc[0]
        self.assertEqual(float(a["actual_fantasy_points"]), 6.0)
        self.assertEqual(float(a["actual_role"]), 0.0)
        self.assertEqual(float(b["actual_fantasy_points"]), 0.0)
        self.assertEqual(float(b["actual_role"]), 0.0)

    def test_pending_boxscore_cannot_be_treated_as_zero(self):
        snapshot = pd.DataFrame([
            {
                "season": 2026, "week": 4, "position": "WR",
                "player_id": "WR-A", "team": "ARI",
                "opponent": "NYG", "game_completed": True,
            }
        ])
        empty = pd.DataFrame(columns=[
            "player_id", "season", "week", "team",
            "actual_fantasy_points", "had_stat_row",
        ])
        coverage = pd.DataFrame([
            {
                "season": 2026, "week": 4, "team": "ARI",
                "boxscore_available": np.nan,
                "snap_team_data_available": np.nan,
            }
        ])
        with patch.object(
            scoring, "load_actual_stats",
            return_value=(empty, empty, coverage),
        ):
            result, pending = scoring.attach_actuals(snapshot)
        self.assertTrue(result.empty)
        self.assertEqual(len(pending), 1)


if __name__ == "__main__":
    unittest.main()
