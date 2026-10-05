"""Contract tests for Kicker v2 market features and experiment design."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "features"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import build_k_v1_candidate_dataset as builder  # noqa: E402
import walk_forward_k_v2 as experiment  # noqa: E402


class KickerMarketTests(unittest.TestCase):
    def test_implied_points_formula_uses_team_spread_perspective(self):
        base = pd.DataFrame([
            {
                "season": 2026, "week": 4,
                "team": "ARI", "opponent": "NYG",
                "game_total_line": 47.0,
                "team_spread_line": -3.0,
            }
        ])
        official = pd.DataFrame([
            {
                "season": 2026, "week": 4,
                "team": "ARI", "opponent": "NYG",
                "kickoff_utc": pd.Timestamp(
                    "2026-10-04T20:00:00Z"
                ),
            }
        ])
        with patch.object(
            builder, "build_schedule_team_weeks",
            return_value=base,
        ), patch.object(
            builder, "load_schedule",
            return_value=official,
        ):
            result = builder.schedule_context()
        self.assertEqual(
            float(result["team_implied_points"].iloc[0]),
            25.0,
        )
        self.assertEqual(
            float(result["opponent_implied_points"].iloc[0]),
            22.0,
        )

    def test_missing_market_does_not_invent_implied_points(self):
        base = pd.DataFrame([
            {
                "season": 2026, "week": 4,
                "team": "ARI", "opponent": "NYG",
                "game_total_line": None,
                "team_spread_line": -3.0,
            }
        ])
        official = pd.DataFrame([
            {
                "season": 2026, "week": 4,
                "team": "ARI", "opponent": "NYG",
                "kickoff_utc": pd.Timestamp(
                    "2026-10-04T20:00:00Z"
                ),
            }
        ])
        with patch.object(
            builder, "build_schedule_team_weeks",
            return_value=base,
        ), patch.object(
            builder, "load_schedule",
            return_value=official,
        ):
            result = builder.schedule_context()
        self.assertTrue(
            pd.isna(result["team_implied_points"].iloc[0])
        )
        self.assertTrue(
            pd.isna(
                result["opponent_implied_points"].iloc[0]
            )
        )


class KickerV2ContractTests(unittest.TestCase):
    def test_feature_groups_never_include_current_game_outcomes(self):
        forbidden = {
            experiment.TARGET,
            experiment.ROLE_TARGET,
            "game_completed",
            "actual_fg_att",
            "actual_pat_att",
            "has_pat_miss_or_block",
            "kickoff_utc",
            "chart_pregame_verified",
            "game_temp",
            "game_wind",
        }
        self.assertFalse(
            forbidden.intersection(
                set(experiment.FULL_FEATURES)
            )
        )
        self.assertIn(
            "team_implied_points",
            experiment.FULL_FEATURES,
        )
        self.assertIn(
            "opponent_implied_points",
            experiment.FULL_FEATURES,
        )

    def test_2026_is_not_a_walk_forward_selection_fold(self):
        self.assertEqual(
            [fold[2] for fold in experiment.FOLDS],
            [2023, 2024, 2025],
        )


if __name__ == "__main__":
    unittest.main()
