"""Contract tests for leakage-safe Kicker v1 candidate construction."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import build_k_v1_candidate_dataset as builder  # noqa: E402
import walk_forward_k_v1 as benchmark  # noqa: E402


class PregameDepthTests(unittest.TestCase):
    def test_2025_postkickoff_snapshot_never_enters_candidates(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            for season in range(2021, 2025):
                pd.DataFrame(columns=[
                    "position", "gsis_id", "season",
                    "week", "club_code", "depth_team",
                ]).to_csv(
                    raw / f"depth_charts_{season}.csv",
                    index=False,
                )
            for season in (2025, 2026):
                chart = pd.DataFrame(columns=[
                    "pos_abb", "gsis_id", "team",
                    "dt", "pos_rank",
                ])
                if season == 2025:
                    chart = pd.DataFrame([
                        {
                            "pos_abb": "K", "gsis_id": "K-OLD",
                            "team": "ARI",
                            "dt": "2025-10-05T14:00:00Z",
                            "pos_rank": 1,
                        },
                        {
                            "pos_abb": "K", "gsis_id": "K-VALID",
                            "team": "ARI",
                            "dt": "2025-10-05T16:30:00Z",
                            "pos_rank": 1,
                        },
                        {
                            "pos_abb": "K", "gsis_id": "K-AFTER",
                            "team": "ARI",
                            "dt": "2025-10-05T18:00:00Z",
                            "pos_rank": 1,
                        },
                    ])
                chart.to_csv(
                    raw / f"depth_charts_{season}.csv",
                    index=False,
                )
            schedule = pd.DataFrame([
                {
                    "season": 2025, "week": 5,
                    "team": "ARI", "opponent": "NYG",
                    "kickoff_utc": pd.Timestamp(
                        "2025-10-05T17:00:00Z"
                    ),
                    "game_completed": 1,
                    "home_away": "home",
                }
            ])
            with patch.object(builder, "RAW", raw):
                result = builder.depth_chart_candidates(schedule)
        self.assertEqual(
            result["player_id"].tolist(), ["K-VALID"]
        )
        self.assertEqual(
            result["chart_pregame_verified"].tolist(), [1]
        )

    def test_unverified_weekly_chart_is_not_marked_pregame(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            pd.DataFrame([
                {
                    "position": "K", "gsis_id": "K-2021",
                    "season": 2021, "week": 1,
                    "club_code": "ARI", "depth_team": 1,
                }
            ]).to_csv(
                raw / "depth_charts_2021.csv",
                index=False,
            )
            for season in (2022, 2023, 2024):
                pd.DataFrame(columns=[
                    "position", "gsis_id", "season",
                    "week", "club_code", "depth_team",
                ]).to_csv(
                    raw / f"depth_charts_{season}.csv",
                    index=False,
                )
            for season in (2025, 2026):
                pd.DataFrame(columns=[
                    "pos_abb", "gsis_id", "team",
                    "dt", "pos_rank",
                ]).to_csv(
                    raw / f"depth_charts_{season}.csv",
                    index=False,
                )
            schedule = pd.DataFrame([
                {
                    "season": 2021, "week": 1,
                    "team": "ARI", "opponent": "NYG",
                    "kickoff_utc": pd.Timestamp(
                        "2021-09-12T17:00:00Z"
                    ),
                    "game_completed": 1,
                    "home_away": "home",
                }
            ])
            with patch.object(builder, "RAW", raw):
                result = builder.depth_chart_candidates(schedule)
        self.assertEqual(
            result["player_id"].tolist(), ["K-2021"]
        )
        self.assertEqual(
            result["chart_pregame_verified"].tolist(), [0]
        )


class CandidateOutcomeTests(unittest.TestCase):
    def test_inactive_scorable_zero_but_future_unlabeled(self):
        candidates = pd.DataFrame([
            {
                "player_id": "K-A", "season": 2025,
                "week": 4, "team": "ARI",
                "game_completed": 1,
            },
            {
                "player_id": "K-A", "season": 2026,
                "week": 4, "team": "ARI",
                "game_completed": 0,
            },
        ])
        actual_cols = [
            *builder.KEY,
            "actual_fantasy_points",
            "actual_fg_att", "actual_fg_made",
            "actual_pat_att", "actual_pat_made",
            "actual_fg_40plus_made",
            "actual_fg_50plus_made",
            "actual_long_misses",
            "actual_pat_missed",
            "actual_pat_blocked",
            "actual_kick_attempts",
            "actual_active_kicker",
            "had_stat_row",
        ]
        actual = pd.DataFrame(columns=actual_cols)
        coverage = pd.DataFrame([
            {
                "season": 2025, "week": 4,
                "team": "ARI", "boxscore_available": 1,
            }
        ])
        result = builder.add_actuals(
            candidates, actual, coverage
        )
        old = result[result["season"].eq(2025)].iloc[0]
        future = result[result["season"].eq(2026)].iloc[0]
        self.assertEqual(
            float(old["actual_fantasy_points"]),
            0.0,
        )
        self.assertEqual(float(old["actual_active_kicker"]), 0.0)
        self.assertTrue(
            np.isnan(future["actual_fantasy_points"])
        )
        self.assertTrue(
            np.isnan(future["actual_active_kicker"])
        )

    def test_player_rolling_features_exclude_current_game(self):
        records = []
        for week, fp in [
            (1, 3.0), (2, 12.0), (3, 99.0),
        ]:
            row = {
                "player_id": "K-A", "season": 2025,
                "week": week,
            }
            for field in builder.PLAYER_METRICS:
                row[field] = fp
            records.append(row)
        data = pd.DataFrame(records)
        result = builder.add_player_history(data)
        week3 = result[
            result["week"].eq(3)
        ].iloc[0]
        self.assertEqual(week3["previous_fp"], 12.0)
        self.assertEqual(
            week3["avg_fp_last_3"], 7.5
        )


class BenchmarkContractTests(unittest.TestCase):
    def test_no_current_game_target_is_a_model_feature(self):
        for field in benchmark.FEATURES:
            self.assertFalse(field.startswith("actual_"))
            self.assertNotIn("game_completed", field)
            self.assertNotIn("has_pat_miss_or_block", field)
            self.assertNotIn(field, [
                "kickoff_utc", "game_wind", "game_temp",
                "chart_pregame_verified",
            ])
        self.assertIn("team_avg_kicker_fp_last_3",
                      benchmark.FEATURES)
        self.assertIn("opp_avg_fg_att_allowed_last_3",
                      benchmark.FEATURES)


if __name__ == "__main__":
    unittest.main()
