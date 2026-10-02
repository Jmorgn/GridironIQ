"""Offline tests for screenshot-confirmed kicker and D/ST scoring.

Run from the repository root:
    py -m unittest discover -s tests -v
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scoring"))

from league_rules import (  # noqa: E402
    FG_MADE_POINTS,
    FG_MISSED_POINTS,
    DST_EVENT_POINTS,
    dst_points_allowed_bonus,
    dst_yards_allowed_bonus,
    kicker_known_components,
)


class KickerScoringTests(unittest.TestCase):
    def kicker_row(self) -> dict[str, float]:
        row = {
            column: 0 for column in [
                *FG_MADE_POINTS,
                *FG_MISSED_POINTS,
                "pat_made",
            ]
        }
        return row

    def test_made_fg_distance_buckets_and_pat(self):
        row = self.kicker_row()
        row.update({
            "fg_made_0_19": 1,
            "fg_made_20_29": 1,
            "fg_made_30_39": 1,
            "fg_made_40_49": 1,
            "fg_made_50_59": 1,
            "fg_made_60_": 1,
            "pat_made": 2,
        })
        result = kicker_known_components(pd.DataFrame([row]))
        self.assertEqual(
            float(result["confirmed_component_points"].iloc[0]),
            25.0,
        )
        self.assertEqual(
            float(result["long_misses_no_penalty"].iloc[0]), 0.0
        )

    def test_missed_short_fg_penalties(self):
        row = self.kicker_row()
        row.update({
            "fg_missed_0_19": 1,
            "fg_missed_20_29": 1,
            "fg_missed_30_39": 1,
            "fg_missed_40_49": 1,
        })
        result = kicker_known_components(pd.DataFrame([row]))
        self.assertEqual(
            float(result["confirmed_component_points"].iloc[0]),
            -3.5,
        )

    def test_missed_50plus_has_zero_penalty(self):
        row = self.kicker_row()
        row["fg_missed_50_59"] = 1
        row["fg_missed_60_"] = 2
        result = kicker_known_components(pd.DataFrame([row]))
        self.assertEqual(
            float(result["long_misses_no_penalty"].iloc[0]), 3.0
        )
        self.assertEqual(
            float(result["confirmed_component_points"].iloc[0]), 0.0
        )

    def test_missing_input_column_is_error_not_zero(self):
        row = self.kicker_row()
        del row["fg_made_50_59"]
        with self.assertRaisesRegex(ValueError, "missing nflverse"):
            kicker_known_components(pd.DataFrame([row]))


class DSTScoringTests(unittest.TestCase):
    def test_points_allowed_visible_brackets(self):
        expected = {
            0: 10.0,
            1: 7.0, 6: 7.0,
            7: 4.0, 13: 4.0,
            14: 1.0, 20: 1.0,
            21: 0.0, 24: 0.0, 27: 0.0,
            28: -1.0, 34: -1.0,
            35: -4.0, 60: -4.0,
        }
        for value, points in expected.items():
            with self.subTest(points_allowed=value):
                self.assertEqual(
                    dst_points_allowed_bonus(value), points
                )

    def test_yards_allowed_visible_brackets(self):
        expected = {
            -4: 4.0,
            0: 3.0, 99: 3.0,
            100: 2.0, 199: 2.0,
            200: 1.0, 299: 1.0,
            300: 0.0, 350: 0.0, 399: 0.0,
            400: -1.0, 499: -1.0,
            500: -2.0, 620: -2.0,
        }
        for value, points in expected.items():
            with self.subTest(yards_allowed=value):
                self.assertEqual(
                    dst_yards_allowed_bonus(value), points
                )

    def test_defensive_and_special_teams_event_values(self):
        expected = {
            "sack": 0.5,
            "interception": 2.0,
            "fumble_recovery": 2.0,
            "touchdown": 6.0,
            "safety": 2.0,
            "blocked_kick": 2.0,
            "kickoff_return_touchdown": 6.0,
            "punt_return_touchdown": 6.0,
            "tackle_for_loss": 0.5,
            "three_and_out_forced": 1.0,
            "extra_point_returned": 2.0,
        }
        self.assertEqual(DST_EVENT_POINTS, expected)


if __name__ == "__main__":
    unittest.main()
