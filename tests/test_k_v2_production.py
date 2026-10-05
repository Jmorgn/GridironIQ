"""Production-selection contracts for official Kicker v2."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))

import train_k_v2  # noqa: E402
import walk_forward_k_v2 as research  # noqa: E402
from tracker_common import POSITIONS  # noqa: E402


class KickerProductionTests(unittest.TestCase):
    def test_production_uses_predeclared_all_no_ids_features(self):
        self.assertEqual(
            train_k_v2.FEATURES,
            research.FEATURE_GROUPS["All no IDs"],
        )
        self.assertNotIn("team", train_k_v2.FEATURES)
        self.assertNotIn("opponent", train_k_v2.FEATURES)
        self.assertIn(
            "team_implied_points", train_k_v2.FEATURES
        )

    def test_tracker_knows_kicker_has_no_role_model(self):
        config = POSITIONS["K"]
        self.assertFalse(config["has_role_model"])
        self.assertIsNone(config["probability_column"])
        self.assertIsNone(config["role_threshold"])
        self.assertEqual(config["starter_count"], 12)


if __name__ == "__main__":
    unittest.main()
