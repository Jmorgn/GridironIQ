"""Contract tests for the roster-aware GridironIQ lineup optimizer."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "lineup"))

import optimize_lineup as lineup  # noqa: E402


def player(
    player_id: str,
    name: str,
    position: str,
    projection: float,
    low: float = 0.0,
    high: float = 20.0,
) -> dict:
    return {
        "player_id": player_id,
        "player_name": name,
        "player_name_key": name.casefold(),
        "position": position,
        "season": 2026,
        "week": 5,
        "team": "AAA",
        "opponent": "BBB",
        "gridironiq_projection": projection,
        "prediction_low_80": low,
        "prediction_high_80": high,
        "key_positives": "",
        "key_negatives": "",
    }


class OptimizationTests(unittest.TestCase):
    def test_global_optimizer_handles_rb_wr_flex(self):
        roster = pd.DataFrame([
            player("RB1", "RB One", "RB", 20.0),
            player("RB2", "RB Two", "RB", 19.0),
            player("RB3", "RB Three", "RB", 3.0),
            player("WR1", "WR One", "WR", 30.0),
            player("WR2", "WR Two", "WR", 18.0),
            player("WR3", "WR Three", "WR", 4.0),
        ])
        slots = lineup.build_slots(
            qb=0, rb=2, wr=2, te=0, flex=1, k=0
        )
        result = lineup.optimize(roster, slots)
        starters = {
            str(value["player_id"])
            for value in result.values()
        }
        self.assertEqual(
            starters,
            {"RB1", "RB2", "WR1", "WR2", "WR3"},
        )
        total = sum(
            float(value["gridironiq_projection"])
            for value in result.values()
        )
        self.assertEqual(total, 91.0)

    def test_flex_never_accepts_kicker_or_tight_end(self):
        roster = pd.DataFrame([
            player("K1", "Huge K", "K", 99.0),
            player("TE1", "Huge TE", "TE", 98.0),
            player("RB1", "Eligible RB", "RB", 5.0),
            player("WR1", "Eligible WR", "WR", 4.0),
        ])
        slots = lineup.build_slots(
            qb=0, rb=0, wr=0, te=0, flex=1, k=0
        )
        result = lineup.optimize(roster, slots)
        self.assertEqual(
            str(result["FLEX"]["player_id"]), "RB1"
        )

    def test_one_player_cannot_fill_rb_and_flex(self):
        roster = pd.DataFrame([
            player("RB1", "Only RB", "RB", 20.0),
        ])
        slots = lineup.build_slots(
            qb=0, rb=1, wr=0, te=0, flex=1, k=0
        )
        with self.assertRaisesRegex(
            RuntimeError, "No legal lineup"
        ):
            lineup.optimize(roster, slots)

    def test_bench_margin_and_interval_overlap(self):
        roster = pd.DataFrame([
            player(
                "QB1", "Starter QB", "QB",
                20.0, low=10.0, high=30.0,
            ),
            player(
                "QB2", "Bench QB", "QB",
                18.0, low=12.0, high=25.0,
            ),
        ])
        slots = lineup.build_slots(
            qb=1, rb=0, wr=0, te=0, flex=0, k=0
        )
        assignment = lineup.optimize(roster, slots)
        starters, bench = lineup.build_outputs(
            roster, slots, assignment
        )
        row = starters.iloc[0]
        self.assertEqual(
            row["best_bench_alternative"], "Bench QB"
        )
        self.assertEqual(
            float(row["projection_margin"]), 2.0
        )
        self.assertTrue(
            bool(row["individual_80_ranges_overlap"])
        )
        self.assertEqual(
            bench["player_name"].tolist(), ["Bench QB"]
        )


class RosterResolutionTests(unittest.TestCase):
    def test_case_insensitive_name_and_dst_are_reported(self):
        pool = pd.DataFrame([
            player("QB1", "Dak Prescott", "QB", 21.0),
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roster.csv"
            pd.DataFrame([
                {
                    "player_name": "dak prescott",
                    "position": "qb",
                },
                {
                    "player_name": "My Defense",
                    "position": "D/ST",
                },
            ]).to_csv(path, index=False)
            matched, unavailable = lineup.load_roster(
                path, pool
            )

        self.assertEqual(
            matched["player_id"].tolist(), ["QB1"]
        )
        self.assertEqual(len(unavailable), 1)
        self.assertEqual(
            unavailable["reason"].iloc[0],
            "D/ST intentionally unsupported",
        )

    def test_strict_unmatched_fails(self):
        pool = pd.DataFrame([
            player("QB1", "Dak Prescott", "QB", 21.0),
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "roster.csv"
            pd.DataFrame([
                {
                    "player_name": "Not A Player",
                    "position": "QB",
                },
            ]).to_csv(path, index=False)
            with self.assertRaisesRegex(
                ValueError, "unavailable/unmatched"
            ):
                lineup.load_roster(
                    path, pool, strict_unmatched=True
                )


class ProjectionPoolTests(unittest.TestCase):
    def test_position_files_must_share_week(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            files = {}
            for index, position in enumerate(
                lineup.SUPPORTED_POSITIONS
            ):
                path = base / f"{position}.csv"
                pd.DataFrame([
                    {
                        "player_id": f"{position}1",
                        "player_name": f"{position} Player",
                        "season": 2026,
                        "week": 6 if position == "K" else 5,
                        "team": "AAA",
                        "opponent": "BBB",
                        "gridironiq_projection": 10 + index,
                        "prediction_low_80": 2,
                        "prediction_high_80": 20,
                    }
                ]).to_csv(path, index=False)
                files[position] = path

            with patch.object(
                lineup, "POSITION_FILES", files
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "disagree on season/week"
                ):
                    lineup.load_projection_pool()


if __name__ == "__main__":
    unittest.main()
