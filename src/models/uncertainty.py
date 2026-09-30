"""Walk-forward residual calibration utilities for GridironIQ.

Prediction intervals in GridironIQ are empirical out-of-sample residual
intervals. They are calibrated from the 2023-2025 walk-forward folds using the
same production scoring rule as the corresponding position model.

The interval is intentionally described as a historical 80% interval rather
than a guaranteed bound: future seasons can differ from the calibration sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COVERAGE = 0.80
ROLE_BUCKETS = (
    ("low", 0.0, 0.50),
    ("medium", 0.50, 0.80),
    ("high", 0.80, 1.01),
)


def role_bucket(probability: float) -> str:
    value = float(probability)
    for name, lower, upper in ROLE_BUCKETS:
        if lower <= value < upper:
            return name
    return "high" if value >= 0.80 else "low"


def _calibrate_group(
    residuals: pd.Series,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict:
    values = pd.to_numeric(
        residuals,
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        raise ValueError("Cannot calibrate an empty residual group.")

    lower = float(
        np.quantile(
            values,
            lower_quantile,
            method="linear",
        )
    )
    upper = float(
        np.quantile(
            values,
            upper_quantile,
            method="linear",
        )
    )
    coverage = float(
        np.mean((values >= lower) & (values <= upper))
    )

    return {
        "n": int(len(values)),
        "lower_residual": lower,
        "upper_residual": upper,
        "empirical_coverage": coverage,
    }


def build_residual_calibration(
    actual: np.ndarray | pd.Series,
    prediction: np.ndarray | pd.Series,
    role_probability: np.ndarray | pd.Series,
    *,
    nominal_coverage: float = DEFAULT_COVERAGE,
    min_bucket_rows: int = 100,
) -> dict:
    """Build asymmetric residual intervals, optionally by role-confidence bin."""
    if not 0.0 < nominal_coverage < 1.0:
        raise ValueError("nominal_coverage must be between 0 and 1.")

    alpha = 1.0 - nominal_coverage
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    frame = pd.DataFrame(
        {
            "actual": pd.to_numeric(actual, errors="coerce"),
            "prediction": pd.to_numeric(
                prediction,
                errors="coerce",
            ),
            "role_probability": pd.to_numeric(
                role_probability,
                errors="coerce",
            ),
        }
    ).dropna()

    if frame.empty:
        raise ValueError("No valid out-of-fold rows for uncertainty calibration.")

    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["role_bucket"] = frame["role_probability"].map(role_bucket)

    all_group = _calibrate_group(
        frame["residual"],
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )

    buckets: dict[str, dict] = {}
    for name, _, _ in ROLE_BUCKETS:
        group = frame.loc[
            frame["role_bucket"].eq(name),
            "residual",
        ]

        if len(group) >= min_bucket_rows:
            result = _calibrate_group(
                group,
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )
            result["uses_fallback"] = False
        else:
            result = dict(all_group)
            result["source_rows"] = int(len(group))
            result["uses_fallback"] = True

        buckets[name] = result

    return {
        "method": "walk_forward_empirical_residual_quantiles",
        "nominal_coverage": float(nominal_coverage),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "min_bucket_rows": int(min_bucket_rows),
        "all": all_group,
        "buckets": buckets,
    }


def apply_residual_intervals(
    prediction: np.ndarray | pd.Series,
    role_probability: np.ndarray | pd.Series,
    calibration: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stored residual quantiles to point predictions."""
    predictions = np.asarray(prediction, dtype=float)
    probabilities = np.asarray(role_probability, dtype=float)

    if predictions.shape != probabilities.shape:
        raise ValueError(
            "prediction and role_probability must have the same shape."
        )

    lower = np.empty_like(predictions, dtype=float)
    upper = np.empty_like(predictions, dtype=float)

    buckets = calibration.get("buckets", {})
    fallback = calibration.get("all")

    if not fallback:
        raise ValueError("Uncertainty calibration is missing its fallback group.")

    for index, (point, probability) in enumerate(
        zip(predictions, probabilities)
    ):
        bucket_name = role_bucket(probability)
        bucket = buckets.get(bucket_name, fallback)

        lower[index] = point + float(
            bucket["lower_residual"]
        )
        upper[index] = point + float(
            bucket["upper_residual"]
        )

    return lower, upper


def calibration_summary(calibration: dict) -> str:
    coverage = calibration.get("nominal_coverage", DEFAULT_COVERAGE)
    all_group = calibration["all"]

    return (
        f"{coverage:.0%} historical walk-forward interval | "
        f"n={all_group['n']:,} | "
        f"calibration coverage={all_group['empirical_coverage']:.1%} | "
        f"residual offsets "
        f"{all_group['lower_residual']:+.2f}/"
        f"{all_group['upper_residual']:+.2f} FP"
    )
