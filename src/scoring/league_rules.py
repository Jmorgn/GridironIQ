"""GridironIQ league-specific Kicker and D/ST scoring from screenshots.

Scored categories were read from league screenshots. The user confirmed
that omitted long-FG misses, 21-27 points allowed and 300-399 yards
allowed have no configured scoring category and therefore score zero.
Missed/blocked extra-point penalties have not been confirmed separately.

Screenshots: IMG_5534.png through IMG_5537.png, provided 2026-10-02.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# Field-goal distance buckets match the nflverse weekly kicking columns.
FG_MADE_POINTS: Mapping[str, float] = {
    "fg_made_0_19": 3.0,
    "fg_made_20_29": 3.0,
    "fg_made_30_39": 3.0,
    "fg_made_40_49": 4.0,
    "fg_made_50_59": 5.0,
    "fg_made_60_": 5.0,
}
FG_MISSED_POINTS: Mapping[str, float] = {
    "fg_missed_0_19": -1.0,
    "fg_missed_20_29": -1.0,
    "fg_missed_30_39": -1.0,
    "fg_missed_40_49": -0.5,
    # Confirmed: no 50+ miss-penalty category in this league.
    "fg_missed_50_59": 0.0,
    "fg_missed_60_": 0.0,
}
PAT_MADE_POINTS = 1.0
# Screenshot does not state a missed/blocked PAT penalty.
PAT_MISSED_POINTS: float | None = None
PAT_BLOCKED_POINTS: float | None = None

DST_EVENT_POINTS: Mapping[str, float] = {
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

# Inclusive integer brackets: (minimum, maximum, points).
# The user confirmed the two unlisted 21-27 and 300-399
# categories award zero points.
DST_POINTS_ALLOWED = (
    (0, 0, 10.0),
    (1, 6, 7.0),
    (7, 13, 4.0),
    (14, 20, 1.0),
    (21, 27, 0.0),
    (28, 34, -1.0),
    (35, None, -4.0),
)
DST_YARDS_ALLOWED = (
    (None, -1, 4.0),
    (0, 99, 3.0),
    (100, 199, 2.0),
    (200, 299, 1.0),
    (300, 399, 0.0),
    (400, 499, -1.0),
    (500, None, -2.0),
)


def bracket_points(
    value: int,
    brackets: tuple[tuple[int | None, int | None, float | None], ...],
    *,
    label: str,
) -> float:
    """Look up an integer score, refusing undocumented brackets."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be a whole number.")
    for minimum, maximum, points in brackets:
        if (minimum is None or value >= minimum) and (
            maximum is None or value <= maximum
        ):
            if points is None:
                raise ValueError(
                    f"{label}={value} falls into a scoring bracket "
                    "not visible in the supplied league screenshots. "
                    "Verify the setting before scoring this game."
                )
            return points
    raise ValueError(f"No {label} scoring bracket for {value}.")


def dst_points_allowed_bonus(points: int) -> float:
    return bracket_points(
        points, DST_POINTS_ALLOWED, label="D/ST points allowed"
    )


def dst_yards_allowed_bonus(yards: int) -> float:
    return bracket_points(
        yards, DST_YARDS_ALLOWED, label="D/ST yards allowed"
    )


def kicker_known_components(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate only kicker scoring components confirmed by screenshots.

    Never fill missing NFL source columns as zero. The returned
    long_misses_no_penalty column records long missed FG attempts that
    correctly carry zero penalty. Missed and blocked PAT penalties
    remain unconfirmed, so this function reports the scored FG and
    made-PAT components rather than claiming other categories are exact.
    """
    required = [
        *FG_MADE_POINTS, *FG_MISSED_POINTS, "pat_made",
    ]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(
            "Cannot reconstruct verified kicker scoring: "
            f"missing nflverse columns {missing}"
        )

    values = frame[required].apply(
        pd.to_numeric, errors="coerce"
    )
    if values.isna().any().any():
        raise ValueError(
            "Kicking input has null/nonnumeric required stats. "
            "Verify source completeness instead of zero-filling."
        )
    if values.lt(0).any().any():
        raise ValueError("Kicking count statistics cannot be negative.")

    points = (
        values["pat_made"] * PAT_MADE_POINTS
    ).astype(float)
    for column, weight in FG_MADE_POINTS.items():
        points += values[column] * weight
    for column, weight in FG_MISSED_POINTS.items():
        points += values[column] * weight

    long_misses_no_penalty = (
        values["fg_missed_50_59"]
        + values["fg_missed_60_"]
    )
    return pd.DataFrame({
        "confirmed_component_points": points,
        "long_misses_no_penalty": long_misses_no_penalty,
    }, index=frame.index)
