"""Compare two running backs using current GridironIQ RB v2 rankings.

Usage:
    py compare_rbs.py "Bucky Irving" "Rachaad White"

The official RB projection is:
    P(35%+ offensive snaps) * conditional fantasy points
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
RANKINGS_FILE = (
    ROOT / "data" / "processed" / "rb_v2_weekly_rankings.csv"
)
ALL_FUTURE_FILE = (
    ROOT / "data" / "processed" / "rb_v2_all_future_candidates.csv"
)


def safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def load_current_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not RANKINGS_FILE.exists():
        raise FileNotFoundError(
            "Missing rb_v2_weekly_rankings.csv. "
            "Run 'py run_weekly.py' first."
        )

    rankings = pd.read_csv(RANKINGS_FILE, low_memory=False)

    if ALL_FUTURE_FILE.exists():
        future = pd.read_csv(ALL_FUTURE_FILE, low_memory=False)
    else:
        future = rankings.copy()

    return rankings, future


def find_player(
    query: str,
    rankings: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.Series:
    query_clean = query.strip().lower()

    if not query_clean:
        raise ValueError("Player name cannot be empty.")

    for frame in (rankings, future):
        names = frame["player_name"].astype(str)

        exact = frame[names.str.lower().eq(query_clean)]
        if len(exact) == 1:
            return exact.iloc[0]

        partial = frame[
            names.str.lower().str.contains(
                query_clean,
                regex=False,
            )
        ]
        if len(partial) == 1:
            return partial.iloc[0]

    unique_names = sorted(
        set(rankings["player_name"].astype(str).tolist())
        | set(future["player_name"].astype(str).tolist())
    )
    match = get_close_matches(
        query,
        unique_names,
        n=1,
        cutoff=0.55,
    )

    if match:
        matched = future[
            future["player_name"].astype(str).eq(match[0])
        ]
        if len(matched):
            print(
                f"Using closest match for '{query}': {match[0]}"
            )
            return matched.iloc[0]

    raise ValueError(
        f"Could not find a current-week RB matching '{query}'. "
        "Run 'py run_weekly.py' to refresh the weekly rankings."
    )


def current_rank(row: pd.Series, rankings: pd.DataFrame) -> str:
    match = rankings[
        rankings["player_id"].astype(str).eq(
            str(row.get("player_id", ""))
        )
    ]

    if len(match):
        rank = safe_float(match.iloc[0].get("rank"))
        return f"RB{int(rank)}" if rank is not None else "Ranked"

    return "Not ranked"


def projection(row: pd.Series) -> float:
    value = safe_float(row.get("gridironiq_projection"))
    return value if value is not None else 0.0


def conditional_points(row: pd.Series) -> float | None:
    return safe_float(row.get("conditional_fantasy_points"))


def role_probability(row: pd.Series) -> float | None:
    return safe_float(row.get("rb_role_probability"))


def prediction_interval(row: pd.Series) -> tuple[float | None, float | None]:
    return (
        safe_float(row.get("prediction_low_80")),
        safe_float(row.get("prediction_high_80")),
    )


def value(row: pd.Series, column: str) -> float | None:
    return safe_float(row.get(column))


def choose_higher(
    a: pd.Series,
    b: pd.Series,
    column: str,
    tolerance: float = 0.0,
) -> str:
    av = value(a, column)
    bv = value(b, column)

    if av is None or bv is None:
        return "N/A"

    if abs(av - bv) <= tolerance:
        return "Similar"

    return str(a["player_name"]) if av > bv else str(b["player_name"])


def choose_lower(
    a: pd.Series,
    b: pd.Series,
    column: str,
    tolerance: float = 0.0,
) -> str:
    av = value(a, column)
    bv = value(b, column)

    if av is None or bv is None:
        return "N/A"

    if abs(av - bv) <= tolerance:
        return "Similar"

    return str(a["player_name"]) if av < bv else str(b["player_name"])


def recommendation_label(gap: float) -> str:
    gap = abs(gap)
    if gap < 1.0:
        return "Essentially a toss-up"
    if gap < 2.5:
        return "Slight lean"
    if gap < 5.0:
        return "Clear lean"
    return "Strong lean"


def print_player_summary(
    row: pd.Series,
    rankings: pd.DataFrame,
) -> None:
    proj = projection(row)
    conditional = conditional_points(row)
    role = role_probability(row)
    rank = current_rank(row, rankings)
    depth_rank = safe_float(row.get("depth_chart_rb_rank"))

    print(
        f"{row['player_name']} "
        f"({row['team']} vs {row['opponent']})"
    )
    print(f"  Rank:              {rank}")
    print(f"  Projection:        {proj:.2f} FP")
    low, high = prediction_interval(row)
    if low is not None and high is not None:
        print(f"  80% range:         {low:.2f} to {high:.2f} FP")
    print(
        "  Role probability: "
        + (f"{role:.1%}" if role is not None else "N/A")
    )
    print(
        "  Conditional FP:    "
        + (
            f"{conditional:.2f}"
            if conditional is not None
            else "N/A"
        )
    )
    print(
        "  Depth chart:       "
        + (
            f"RB{int(depth_rank)}"
            if depth_rank is not None
            else "N/A"
        )
    )
    print(f"  Location:          {str(row.get('home_away', 'N/A')).title()}")
    print(
        "  Positives:         "
        + str(
            row.get(
                "key_positives",
                "No explanation available",
            )
        )
    )
    print(
        "  Negatives:         "
        + str(
            row.get(
                "key_negatives",
                "No explanation available",
            )
        )
    )

    if role is not None and conditional is not None:
        role_discount = conditional - proj
        print(
            f"  Workload discount: -{role_discount:.2f} FP "
            f"({conditional:.2f} x {role:.1%} = {proj:.2f})"
        )


def print_model_attribution(row: pd.Series) -> None:
    """Print SHAP attribution for the conditional RB points model."""
    from src.models.explain_rb_v2 import (
        explain_conditional_prediction,
    )

    try:
        result = explain_conditional_prediction(
            player_id=str(row["player_id"]),
            season=int(safe_float(row.get("season")) or 0),
            week=int(safe_float(row.get("week")) or 0),
            team=str(row["team"]),
            top_n=4,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"  SHAP unavailable: {exc}")
        return

    print(
        f"  Conditional-model baseline: "
        f"{result['baseline']:.2f} FP"
    )
    print(
        f"  Conditional-model output:   "
        f"{result['pipeline_prediction']:.2f} FP"
    )

    print("  Model factors pushing UP:")
    for item in result["positive"]:
        print(
            f"    + {item['label']:<38} "
            f"{item['contribution']:+.2f} FP "
            f"(value={item['value']})"
        )

    if not result["positive"]:
        print("    + No positive SHAP contributions")

    print("  Model factors pushing DOWN:")
    for item in result["negative"]:
        print(
            f"    - {item['label']:<38} "
            f"{item['contribution']:+.2f} FP "
            f"(value={item['value']})"
        )

    if not result["negative"]:
        print("    - No negative SHAP contributions")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two RBs using current GridironIQ projections."
    )
    parser.add_argument(
        "rb1",
        help='First RB, e.g. "Bucky Irving"',
    )
    parser.add_argument(
        "rb2",
        help='Second RB, e.g. "Rachaad White"',
    )
    args = parser.parse_args()

    rankings, future = load_current_data()
    a = find_player(args.rb1, rankings, future)
    b = find_player(args.rb2, rankings, future)

    season = int(
        safe_float(rankings.iloc[0].get("season")) or 0
    )
    week = int(
        safe_float(rankings.iloc[0].get("week")) or 0
    )

    print("GRIDIRONIQ RB START / SIT COMPARISON")
    print("=" * 88)
    print(f"Season {season} | Week {week}\n")

    print_player_summary(a, rankings)
    print()
    print_player_summary(b, rankings)

    diff = projection(a) - projection(b)

    print("\n" + "=" * 88)
    print("SIDE-BY-SIDE SIGNALS")
    print("=" * 88)

    categories = [
        (
            "Recent opportunities",
            choose_higher(
                a,
                b,
                "avg_opportunities_last_3",
                tolerance=1.0,
            ),
            fmt(
                value(a, "avg_opportunities_last_3"),
                1,
                " opp",
            ),
            fmt(
                value(b, "avg_opportunities_last_3"),
                1,
                " opp",
            ),
        ),
        (
            "Recent snap share",
            choose_higher(
                a,
                b,
                "avg_offense_pct_last_3",
                tolerance=0.05,
            ),
            fmt_pct(value(a, "avg_offense_pct_last_3")),
            fmt_pct(value(b, "avg_offense_pct_last_3")),
        ),
        (
            "Recent fantasy form",
            choose_higher(
                a,
                b,
                "avg_fp_last_3",
                tolerance=1.0,
            ),
            fmt(value(a, "avg_fp_last_3"), 1, " FP"),
            fmt(value(b, "avg_fp_last_3"), 1, " FP"),
        ),
        (
            "Rushing matchup",
            choose_higher(
                a,
                b,
                "opp_avg_rush_yards_allowed_last_3",
                tolerance=10.0,
            ),
            fmt(
                value(
                    a,
                    "opp_avg_rush_yards_allowed_last_3",
                ),
                0,
                " yds",
            ),
            fmt(
                value(
                    b,
                    "opp_avg_rush_yards_allowed_last_3",
                ),
                0,
                " yds",
            ),
        ),
        (
            "RB-room competition",
            choose_lower(
                a,
                b,
                "rb_room_max_other_prior_opportunities",
                tolerance=1.0,
            ),
            fmt(
                value(
                    a,
                    "rb_room_max_other_prior_opportunities",
                ),
                1,
                " opp",
            ),
            fmt(
                value(
                    b,
                    "rb_room_max_other_prior_opportunities",
                ),
                1,
                " opp",
            ),
        ),
        (
            "Game environment",
            choose_higher(
                a,
                b,
                "game_total_line",
                tolerance=1.0,
            ),
            fmt(value(a, "game_total_line"), 1),
            fmt(value(b, "game_total_line"), 1),
        ),
        (
            "Workload confidence",
            choose_higher(
                a,
                b,
                "rb_role_probability",
                tolerance=0.03,
            ),
            fmt_pct(role_probability(a)),
            fmt_pct(role_probability(b)),
        ),
    ]

    print(
        f"{'Category':<23}"
        f"{str(a['player_name']):>18}"
        f"{str(b['player_name']):>18}"
        f"{'Edge':>24}"
    )
    print("-" * 88)

    for label, edge, avalue, bvalue in categories:
        print(
            f"{label:<23}"
            f"{avalue:>18}"
            f"{bvalue:>18}"
            f"{edge:>24}"
        )

    print("\n" + "=" * 88)
    print("MODEL ATTRIBUTION (SHAP)")
    print("=" * 88)
    print(
        "SHAP explains the conditional Gradient Boosting fantasy-points "
        "model. The separate workload classifier then scales that output "
        "by P(35%+ offensive snaps) to create the official projection."
    )

    print(f"\n{a['player_name']}")
    print_model_attribution(a)

    print(f"\n{b['player_name']}")
    print_model_attribution(b)

    print("\n" + "=" * 88)
    print("GRIDIRONIQ LEAN")
    print("=" * 88)

    if diff >= 0:
        preferred = a
        other = b
    else:
        preferred = b
        other = a

    gap = abs(diff)
    label = recommendation_label(gap)

    print(
        f"{label}: START {preferred['player_name']} over "
        f"{other['player_name']}."
    )
    print(
        f"Projection gap: {gap:.2f} FP "
        f"({projection(preferred):.2f} vs "
        f"{projection(other):.2f})."
    )

    preferred_low, preferred_high = prediction_interval(preferred)
    other_low, other_high = prediction_interval(other)
    if (
        preferred_low is not None
        and preferred_high is not None
        and other_low is not None
        and other_high is not None
    ):
        overlap = max(
            0.0,
            min(preferred_high, other_high)
            - max(preferred_low, other_low),
        )
        if overlap > 0:
            print(
                "Uncertainty note: the historical 80% prediction ranges "
                f"overlap by {overlap:.2f} FP, so the point-estimate edge "
                "is not a guarantee."
            )

    preferred_role = role_probability(preferred)
    other_role = role_probability(other)

    if (
        preferred_role is not None
        and preferred_role < 0.65
    ):
        print(
            "Caution: the preferred RB has below-65% probability of "
            "reaching the model's 35% snap-share role threshold."
        )

    if (
        preferred_role is not None
        and other_role is not None
        and abs(preferred_role - other_role) >= 0.20
    ):
        print(
            "Workload confidence is a major part of this gap: "
            f"{preferred['player_name']} {preferred_role:.1%} vs "
            f"{other['player_name']} {other_role:.1%}."
        )

    print(
        "\nThe context table is descriptive. SHAP attribution applies "
        "only to conditional fantasy points; workload probability is a "
        "separate model stage. The 80% range is calibrated from historical "
        "2023-2025 walk-forward residuals and is not a guaranteed bound."
    )


if __name__ == "__main__":
    main()
