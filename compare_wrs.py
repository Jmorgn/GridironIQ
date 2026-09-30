"""Compare two wide receivers using current GridironIQ WR v2 rankings.

Usage:
    py compare_wrs.py "Amon-Ra St. Brown" "Puka Nacua"

The official WR projection is:
    P(65%+ offensive snaps) * conditional fantasy points
"""

from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import argparse

import pandas as pd

ROOT = Path(__file__).resolve().parent
RANKINGS_FILE = (
    ROOT
    / "data"
    / "processed"
    / "wr_v2_weekly_rankings.csv"
)
ALL_FUTURE_FILE = (
    ROOT
    / "data"
    / "processed"
    / "wr_v2_all_future_candidates.csv"
)


def safe_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(
    value: float | None,
    digits: int = 1,
    suffix: str = "",
) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def fmt_pct(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1%}"


def load_current_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not RANKINGS_FILE.exists():
        raise FileNotFoundError(
            "Missing wr_v2_weekly_rankings.csv. "
            "Run 'py run_weekly.py' first."
        )

    rankings = pd.read_csv(
        RANKINGS_FILE,
        low_memory=False,
    )

    if ALL_FUTURE_FILE.exists():
        future = pd.read_csv(
            ALL_FUTURE_FILE,
            low_memory=False,
        )
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
        raise ValueError(
            "Player name cannot be empty."
        )

    for frame in (
        rankings,
        future,
    ):
        names = frame[
            "player_name"
        ].astype(str)

        exact = frame[
            names.str.lower().eq(
                query_clean
            )
        ]
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
        set(
            rankings[
                "player_name"
            ].astype(str).tolist()
        )
        | set(
            future[
                "player_name"
            ].astype(str).tolist()
        )
    )

    match = get_close_matches(
        query,
        unique_names,
        n=1,
        cutoff=0.55,
    )

    if match:
        matched = future[
            future[
                "player_name"
            ].astype(str).eq(
                match[0]
            )
        ]
        if len(matched):
            print(
                "Using closest match for "
                f"'{query}': {match[0]}"
            )
            return matched.iloc[0]

    raise ValueError(
        "Could not find a current-week WR "
        f"matching '{query}'. Run "
        "'py run_weekly.py' to refresh "
        "the weekly rankings."
    )


def current_rank(
    row: pd.Series,
    rankings: pd.DataFrame,
) -> str:
    match = rankings[
        rankings[
            "player_id"
        ].astype(str).eq(
            str(
                row.get(
                    "player_id",
                    "",
                )
            )
        )
    ]

    if len(match):
        rank = safe_float(
            match.iloc[0].get(
                "rank"
            )
        )
        return (
            f"WR{int(rank)}"
            if rank is not None
            else "Ranked"
        )

    return "Not ranked"


def projection(
    row: pd.Series,
) -> float:
    value = safe_float(
        row.get(
            "gridironiq_projection"
        )
    )
    return (
        value
        if value is not None
        else 0.0
    )


def conditional_points(
    row: pd.Series,
) -> float | None:
    return safe_float(
        row.get(
            "conditional_fantasy_points"
        )
    )


def role_probability(
    row: pd.Series,
) -> float | None:
    return safe_float(
        row.get(
            "wr_role_probability"
        )
    )


def prediction_interval(
    row: pd.Series,
) -> tuple[
    float | None,
    float | None,
]:
    return (
        safe_float(
            row.get(
                "prediction_low_80"
            )
        ),
        safe_float(
            row.get(
                "prediction_high_80"
            )
        ),
    )


def value(
    row: pd.Series,
    column: str,
) -> float | None:
    return safe_float(
        row.get(column)
    )


def choose_higher(
    a: pd.Series,
    b: pd.Series,
    column: str,
    tolerance: float = 0.0,
) -> str:
    av = value(
        a,
        column,
    )
    bv = value(
        b,
        column,
    )

    if (
        av is None
        or bv is None
    ):
        return "N/A"

    if abs(av - bv) <= tolerance:
        return "Similar"

    return (
        str(a["player_name"])
        if av > bv
        else str(b["player_name"])
    )


def choose_lower(
    a: pd.Series,
    b: pd.Series,
    column: str,
    tolerance: float = 0.0,
) -> str:
    av = value(
        a,
        column,
    )
    bv = value(
        b,
        column,
    )

    if (
        av is None
        or bv is None
    ):
        return "N/A"

    if abs(av - bv) <= tolerance:
        return "Similar"

    return (
        str(a["player_name"])
        if av < bv
        else str(b["player_name"])
    )


def recommendation_label(
    gap: float,
) -> str:
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
    conditional = (
        conditional_points(row)
    )
    role = role_probability(row)
    rank = current_rank(
        row,
        rankings,
    )
    depth_rank = safe_float(
        row.get(
            "depth_chart_wr_rank"
        )
    )

    print(
        f"{row['player_name']} "
        f"({row['team']} vs "
        f"{row['opponent']})"
    )
    print(
        f"  Rank:              "
        f"{rank}"
    )
    print(
        f"  Projection:        "
        f"{proj:.2f} FP"
    )

    low, high = prediction_interval(
        row
    )
    if (
        low is not None
        and high is not None
    ):
        print(
            f"  80% range:         "
            f"{low:.2f} to "
            f"{high:.2f} FP"
        )

    print(
        "  Role probability: "
        + (
            f"{role:.1%}"
            if role is not None
            else "N/A"
        )
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
            f"WR{int(depth_rank)}"
            if depth_rank is not None
            else "N/A"
        )
    )
    print(
        "  Location:          "
        f"{str(row.get('home_away', 'N/A')).title()}"
    )
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

    if (
        role is not None
        and conditional is not None
    ):
        role_discount = (
            conditional - proj
        )
        print(
            "  Workload discount: "
            f"-{role_discount:.2f} FP "
            f"({conditional:.2f} x "
            f"{role:.1%} = "
            f"{proj:.2f})"
        )


def print_model_attribution(
    row: pd.Series,
) -> None:
    from src.models.explain_wr_v2 import (
        explain_conditional_prediction,
    )

    try:
        result = (
            explain_conditional_prediction(
                player_id=str(
                    row["player_id"]
                ),
                season=int(
                    safe_float(
                        row.get("season")
                    )
                    or 0
                ),
                week=int(
                    safe_float(
                        row.get("week")
                    )
                    or 0
                ),
                team=str(
                    row["team"]
                ),
                top_n=4,
            )
        )
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(
            f"  SHAP unavailable: "
            f"{exc}"
        )
        return

    print(
        "  Conditional-model baseline: "
        f"{result['baseline']:.2f} FP"
    )
    print(
        "  Conditional-model output:   "
        f"{result['pipeline_prediction']:.2f} FP"
    )

    print(
        "  Model factors pushing UP:"
    )
    for item in result[
        "positive"
    ]:
        print(
            "    + "
            f"{item['label']:<38} "
            f"{item['contribution']:+.2f} FP "
            f"(value={item['value']})"
        )

    if not result["positive"]:
        print(
            "    + No positive "
            "SHAP contributions"
        )

    print(
        "  Model factors pushing DOWN:"
    )
    for item in result[
        "negative"
    ]:
        print(
            "    - "
            f"{item['label']:<38} "
            f"{item['contribution']:+.2f} FP "
            f"(value={item['value']})"
        )

    if not result["negative"]:
        print(
            "    - No negative "
            "SHAP contributions"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two WRs using current "
            "GridironIQ projections."
        )
    )
    parser.add_argument(
        "wr1",
        help='First WR, e.g. "Amon-Ra St. Brown"',
    )
    parser.add_argument(
        "wr2",
        help='Second WR, e.g. "Puka Nacua"',
    )
    args = parser.parse_args()

    rankings, future = (
        load_current_data()
    )
    a = find_player(
        args.wr1,
        rankings,
        future,
    )
    b = find_player(
        args.wr2,
        rankings,
        future,
    )

    season = int(
        safe_float(
            rankings.iloc[0].get(
                "season"
            )
        )
        or 0
    )
    week = int(
        safe_float(
            rankings.iloc[0].get(
                "week"
            )
        )
        or 0
    )

    print(
        "GRIDIRONIQ WR START / "
        "SIT COMPARISON"
    )
    print("=" * 92)
    print(
        f"Season {season} | "
        f"Week {week}\n"
    )

    print_player_summary(
        a,
        rankings,
    )
    print()
    print_player_summary(
        b,
        rankings,
    )

    diff = (
        projection(a)
        - projection(b)
    )

    print(
        "\n" + "=" * 92
    )
    print(
        "SIDE-BY-SIDE SIGNALS"
    )
    print("=" * 92)

    categories = [
        (
            "Recent targets",
            choose_higher(
                a,
                b,
                "avg_targets_last_3",
                tolerance=0.75,
            ),
            fmt(
                value(
                    a,
                    "avg_targets_last_3",
                ),
                1,
                " tgt",
            ),
            fmt(
                value(
                    b,
                    "avg_targets_last_3",
                ),
                1,
                " tgt",
            ),
        ),
        (
            "Recent target share",
            choose_higher(
                a,
                b,
                "avg_target_share_last_3",
                tolerance=0.03,
            ),
            fmt_pct(
                value(
                    a,
                    "avg_target_share_last_3",
                )
            ),
            fmt_pct(
                value(
                    b,
                    "avg_target_share_last_3",
                )
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
            fmt_pct(
                value(
                    a,
                    "avg_offense_pct_last_3",
                )
            ),
            fmt_pct(
                value(
                    b,
                    "avg_offense_pct_last_3",
                )
            ),
        ),
        (
            "Recent fantasy form",
            choose_higher(
                a,
                b,
                "avg_fp_last_3",
                tolerance=1.0,
            ),
            fmt(
                value(
                    a,
                    "avg_fp_last_3",
                ),
                1,
                " FP",
            ),
            fmt(
                value(
                    b,
                    "avg_fp_last_3",
                ),
                1,
                " FP",
            ),
        ),
        (
            "Receiving matchup",
            choose_higher(
                a,
                b,
                "opp_avg_rec_yards_allowed_last_3",
                tolerance=15.0,
            ),
            fmt(
                value(
                    a,
                    "opp_avg_rec_yards_allowed_last_3",
                ),
                0,
                " yds",
            ),
            fmt(
                value(
                    b,
                    "opp_avg_rec_yards_allowed_last_3",
                ),
                0,
                " yds",
            ),
        ),
        (
            "WR-room competition",
            choose_lower(
                a,
                b,
                "wr_room_max_other_prior_targets",
                tolerance=0.75,
            ),
            fmt(
                value(
                    a,
                    "wr_room_max_other_prior_targets",
                ),
                1,
                " tgt",
            ),
            fmt(
                value(
                    b,
                    "wr_room_max_other_prior_targets",
                ),
                1,
                " tgt",
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
            fmt(
                value(
                    a,
                    "game_total_line",
                ),
                1,
            ),
            fmt(
                value(
                    b,
                    "game_total_line",
                ),
                1,
            ),
        ),
        (
            "Role confidence",
            choose_higher(
                a,
                b,
                "wr_role_probability",
                tolerance=0.03,
            ),
            fmt_pct(
                role_probability(a)
            ),
            fmt_pct(
                role_probability(b)
            ),
        ),
    ]

    print(
        f"{'Category':<23}"
        f"{str(a['player_name']):>20}"
        f"{str(b['player_name']):>20}"
        f"{'Edge':>27}"
    )
    print("-" * 92)

    for (
        label,
        edge,
        avalue,
        bvalue,
    ) in categories:
        print(
            f"{label:<23}"
            f"{avalue:>20}"
            f"{bvalue:>20}"
            f"{edge:>27}"
        )

    print(
        "\n" + "=" * 92
    )
    print(
        "MODEL ATTRIBUTION (SHAP)"
    )
    print("=" * 92)
    print(
        "SHAP explains the conditional "
        "Gradient Boosting fantasy-points "
        "model. The separate role classifier "
        "then scales that output by P(65%+ "
        "offensive snaps) to create the "
        "official projection."
    )

    print(
        f"\n{a['player_name']}"
    )
    print_model_attribution(a)

    print(
        f"\n{b['player_name']}"
    )
    print_model_attribution(b)

    print(
        "\n" + "=" * 92
    )
    print("GRIDIRONIQ LEAN")
    print("=" * 92)

    if diff >= 0:
        preferred = a
        other = b
    else:
        preferred = b
        other = a

    gap = abs(diff)
    label = recommendation_label(
        gap
    )

    print(
        f"{label}: START "
        f"{preferred['player_name']} over "
        f"{other['player_name']}."
    )
    print(
        f"Projection gap: {gap:.2f} FP "
        f"({projection(preferred):.2f} vs "
        f"{projection(other):.2f})."
    )

    (
        preferred_low,
        preferred_high,
    ) = prediction_interval(
        preferred
    )
    (
        other_low,
        other_high,
    ) = prediction_interval(
        other
    )

    if (
        preferred_low is not None
        and preferred_high is not None
        and other_low is not None
        and other_high is not None
    ):
        overlap = max(
            0.0,
            min(
                preferred_high,
                other_high,
            )
            - max(
                preferred_low,
                other_low,
            ),
        )
        if overlap > 0:
            print(
                "Uncertainty note: the "
                "historical 80% prediction "
                "ranges overlap by "
                f"{overlap:.2f} FP, so the "
                "point-estimate edge is not "
                "a guarantee."
            )

    preferred_role = (
        role_probability(
            preferred
        )
    )
    other_role = (
        role_probability(
            other
        )
    )

    if (
        preferred_role is not None
        and preferred_role < 0.65
    ):
        print(
            "Caution: the preferred WR has "
            "below-65% probability of "
            "reaching the model's 65% "
            "snap-share role threshold."
        )

    if (
        preferred_role is not None
        and other_role is not None
        and abs(
            preferred_role
            - other_role
        ) >= 0.20
    ):
        print(
            "Role confidence is a major "
            "part of this gap: "
            f"{preferred['player_name']} "
            f"{preferred_role:.1%} vs "
            f"{other['player_name']} "
            f"{other_role:.1%}."
        )

    print(
        "\nThe context table is descriptive. "
        "SHAP attribution applies only to "
        "conditional fantasy points; role "
        "probability is a separate model "
        "stage. The 80% range is calibrated "
        "from historical 2023-2025 "
        "walk-forward residuals and is not "
        "a guaranteed bound."
    )


if __name__ == "__main__":
    main()
