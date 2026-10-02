"""Offline tests for RB/WR-only GridironIQ FLEX rankings.

Run from repository root:
    py -m unittest discover -s tests -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
import rank_flex  # noqa: E402


class FlexTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        self.rb_path = folder / "rb.csv"
        self.wr_path = folder / "wr.csv"
        self.inputs = {
            "RB": self.rb_path,
            "WR": self.wr_path,
        }
        self.patcher = patch.object(rank_flex, "INPUTS", self.inputs)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

        rb = pd.DataFrame([
            {
                "rank": 1,
                "player_id": "00-RB-A",
                "player_name": "Running Back A",
                "season": 2026,
                "week": 4,
                "team": "ARI",
                "opponent": "NYG",
                "depth_chart_rb_rank": 1,
                "rb_role_probability": 0.75,
                "conditional_fantasy_points": 20.0,
                "gridironiq_projection": 15.0,
                "prediction_low_80": 5.0,
                "prediction_high_80": 25.0,
            },
            {
                "rank": 2,
                "player_id": "00-RB-B",
                "player_name": "Running Back B",
                "season": 2026,
                "week": 4,
                "team": "SF",
                "opponent": "DEN",
                "depth_chart_rb_rank": 2,
                "rb_role_probability": 0.80,
                "conditional_fantasy_points": 10.0,
                "gridironiq_projection": 8.0,
                "prediction_low_80": 1.0,
                "prediction_high_80": 16.0,
            },
        ])
        wr = pd.DataFrame([
            {
                "rank": 1,
                "player_id": "00-WR-A",
                "player_name": "Receiver A",
                "season": 2026,
                "week": 4,
                "team": "SEA",
                "opponent": "LAC",
                "depth_chart_wr_rank": 1,
                "wr_role_probability": 0.90,
                "conditional_fantasy_points": 16.0,
                "gridironiq_projection": 14.4,
                "prediction_low_80": 4.0,
                "prediction_high_80": 24.0,
            },
            {
                "rank": 2,
                "player_id": "00-WR-B",
                "player_name": "Receiver B",
                "season": 2026,
                "week": 4,
                "team": "GB",
                "opponent": "TB",
                "depth_chart_wr_rank": 2,
                "wr_role_probability": 0.70,
                "conditional_fantasy_points": 15.0,
                "gridironiq_projection": 10.5,
                "prediction_low_80": 2.0,
                "prediction_high_80": 19.0,
            },
        ])
        rb.to_csv(self.rb_path, index=False)
        wr.to_csv(self.wr_path, index=False)

    def test_combines_only_rb_and_wr_ranked_by_projection(self):
        result = rank_flex.build_flex_rankings()
        self.assertEqual(
            result["position"].tolist(), ["RB", "WR", "WR", "RB"]
        )
        self.assertEqual(result["rank"].tolist(), [1, 2, 3, 4])
        self.assertEqual(
            result["position_rank"].tolist(), [1, 1, 2, 2]
        )
        self.assertEqual(
            result["gridironiq_projection"].tolist(),
            [15.0, 14.4, 10.5, 8.0],
        )

    def test_excluding_locked_starters_preserves_position_rank(self):
        result = rank_flex.build_flex_rankings(
            excluded_names=("Running Back A",),
            excluded_ids=("00-WR-A",),
        )
        self.assertEqual(
            result["player_name"].tolist(),
            ["Receiver B", "Running Back B"],
        )
        self.assertEqual(result["rank"].tolist(), [1, 2])
        self.assertEqual(result["position_rank"].tolist(), [2, 2])

    def test_unknown_exclusion_fails_instead_of_silently_ignoring(self):
        with self.assertRaisesRegex(ValueError, "not found"):
            rank_flex.build_flex_rankings(
                excluded_names=("Misspelled Player",)
            )

    def test_rejects_week_mismatch(self):
        wr = pd.read_csv(self.wr_path)
        wr["week"] = 5
        wr.to_csv(self.wr_path, index=False)
        with self.assertRaisesRegex(RuntimeError, "Refresh both"):
            rank_flex.build_flex_rankings()

    def test_rejects_duplicate_player_rows(self):
        rb = pd.read_csv(self.rb_path)
        rb = pd.concat([rb, rb.head(1)], ignore_index=True)
        rb.to_csv(self.rb_path, index=False)
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            rank_flex.build_flex_rankings()


if __name__ == "__main__":
    unittest.main()
