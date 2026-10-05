"""Generate current-week GridironIQ Kicker v2 rankings.

Production K v2 is direct Gradient Boosting on the "All no IDs"
feature set. There is no activity/role classifier or probability gate.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from uncertainty import apply_residual_intervals

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "data" / "processed" / "k_v1_candidate_dataset.csv"
BUNDLE_FILE = ROOT / "models" / "k_v2_bundle.joblib"
ALL_FUTURE_FILE = (
    ROOT / "data" / "processed" / "k_v2_all_future_candidates.csv"
)
RANKINGS_FILE = (
    ROOT / "data" / "processed" / "k_v2_weekly_rankings.csv"
)


def safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_context(row: pd.Series) -> tuple[str, str]:
    positives: list[str] = []
    negatives: list[str] = []

    implied = safe_float(row.get("team_implied_points"))
    total = safe_float(row.get("game_total_line"))
    team_fg = safe_float(row.get("team_avg_fg_att_last_3"))
    team_pat = safe_float(row.get("team_avg_pat_att_last_3"))
    player_fp = safe_float(row.get("avg_fp_last_3"))
    opp_fg = safe_float(row.get("opp_avg_fg_att_allowed_last_3"))
    roof = str(row.get("roof", "")).lower()

    if implied is not None and implied >= 25:
        positives.append(
            f"Strong implied team scoring ({implied:.1f})"
        )
    elif implied is not None and implied <= 20:
        negatives.append(
            f"Low implied team scoring ({implied:.1f})"
        )

    if total is not None and total >= 47:
        positives.append(f"High game total ({total:.1f})")
    elif total is not None and total <= 40:
        negatives.append(f"Low game total ({total:.1f})")

    if team_fg is not None and team_fg >= 2.2:
        positives.append(
            f"Team averaging {team_fg:.1f} FG attempts recently"
        )
    elif team_fg is not None and team_fg <= 1.2:
        negatives.append(
            f"Team averaging only {team_fg:.1f} FG attempts recently"
        )

    if team_pat is not None and team_pat >= 2.5:
        positives.append(
            f"Frequent PAT opportunity ({team_pat:.1f}/game)"
        )

    if player_fp is not None and player_fp >= 10:
        positives.append(
            f"Strong recent K scoring ({player_fp:.1f} FP)"
        )
    elif player_fp is not None and player_fp <= 5:
        negatives.append(
            f"Low recent K scoring ({player_fp:.1f} FP)"
        )

    if opp_fg is not None and opp_fg >= 2.2:
        positives.append(
            f"Opponent allowing {opp_fg:.1f} FG attempts/game recently"
        )

    if roof in {"dome", "closed"}:
        positives.append("Indoor/closed-roof kicking environment")

    return (
        "; ".join(positives[:3]) or "No major positive signal",
        "; ".join(negatives[:3]) or "No major negative signal",
    )


def main() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run the K candidate builder first."
        )
    if not BUNDLE_FILE.is_file():
        raise FileNotFoundError(
            f"Missing {BUNDLE_FILE}. Run train_k_v2.py once first."
        )

    df = pd.read_csv(
        DATA_FILE, low_memory=False,
        dtype={"player_id": str},
    )
    bundle = joblib.load(BUNDLE_FILE)
    if bundle.get("version") != "k_v2":
        raise RuntimeError(
            "Unexpected Kicker model bundle version."
        )
    features = bundle["features"]
    model = bundle["regressor"]
    uncertainty = bundle.get("uncertainty")
    if uncertainty is None:
        raise RuntimeError(
            "K v2 bundle has no uncertainty calibration. Retrain it."
        )

    missing = [
        feature for feature in features
        if feature not in df.columns
    ]
    if missing:
        raise RuntimeError(
            "Current K candidate dataset is missing features: "
            + ", ".join(missing)
        )

    future = df[
        df["game_completed"].eq(0)
    ].copy()
    if future.empty:
        print("No future/unplayed Kicker candidates found.")
        return

    projection = np.maximum(
        model.predict(future[features]), 0.0
    )
    confidence = np.ones(len(future), dtype=float)
    low, high = apply_residual_intervals(
        projection, confidence, uncertainty
    )

    keep = [
        "player_id", "player_name",
        "season", "week", "team", "opponent",
        "home_away", "depth_chart_k_rank", "listed_k1",
        "team_spread_line", "game_total_line",
        "team_implied_points", "opponent_implied_points",
        "avg_fp_last_3", "avg_fg_att_last_3",
        "avg_pat_att_last_3",
        "team_avg_kicker_fp_last_3",
        "team_avg_fg_att_last_3",
        "team_avg_pat_att_last_3",
        "opp_avg_fg_att_allowed_last_3",
        "opp_avg_pat_att_allowed_last_3",
        "roof", "surface",
    ]
    keep = [col for col in keep if col in future.columns]
    output = future[keep].copy()
    output["gridironiq_projection"] = projection
    output["prediction_low_80"] = low
    output["prediction_high_80"] = high
    # Explicit field for the prospective tracker: K v2 has no role
    # classifier, so this is metadata rather than a probability.
    output["k_has_role_model"] = 0

    contexts = [
        build_context(future.iloc[index])
        for index in range(len(future))
    ]
    output["key_positives"] = [x[0] for x in contexts]
    output["key_negatives"] = [x[1] for x in contexts]

    output = output.sort_values(
        ["season", "week", "gridironiq_projection", "player_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    ALL_FUTURE_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )
    output.to_csv(ALL_FUTURE_FILE, index=False)

    first = (
        output[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .iloc[0]
    )
    rankings = output[
        output["season"].eq(first["season"])
        & output["week"].eq(first["week"])
    ].copy()
    rankings = rankings.sort_values(
        ["gridironiq_projection", "player_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    rankings.insert(0, "rank", rankings.index + 1)
    rankings.to_csv(RANKINGS_FILE, index=False)

    print("GRIDIRONIQ WEEKLY KICKER RANKINGS")
    print("=" * 92)
    print(
        f"Season {int(first['season'])} | "
        f"Week {int(first['week'])} | "
        f"{len(rankings)} pregame K candidates"
    )
    print(
        "Production model: direct Gradient Boosting, All no IDs. "
        "No activity/role gate.\n"
    )
    display = rankings[
        [
            "rank", "player_name", "team", "opponent",
            "gridironiq_projection",
            "prediction_low_80", "prediction_high_80",
            "team_implied_points", "game_total_line",
        ]
    ].head(32)
    print(
        display.to_string(
            index=False,
            formatters={
                "gridironiq_projection": "{:.2f}".format,
                "prediction_low_80": "{:.2f}".format,
                "prediction_high_80": "{:.2f}".format,
                "team_implied_points": "{:.1f}".format,
                "game_total_line": "{:.1f}".format,
            },
        )
    )
    print("\nWHY GRIDIRONIQ LIKES / DISLIKES THE TOP 10")
    print("=" * 92)
    for _, row in rankings.head(10).iterrows():
        print(
            f"\n#{int(row['rank'])} {row['player_name']} "
            f"({row['team']} vs {row['opponent']}) — "
            f"{row['gridironiq_projection']:.2f} FP"
        )
        print(f"  + {row['key_positives']}")
        print(f"  - {row['key_negatives']}")

    print(f"\nSaved all future K candidates: {ALL_FUTURE_FILE}")
    print(f"Saved weekly K rankings:       {RANKINGS_FILE}")


if __name__ == "__main__":
    main()
