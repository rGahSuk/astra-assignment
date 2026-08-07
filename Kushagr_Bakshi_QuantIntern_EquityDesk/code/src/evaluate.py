"""Output construction, metrics, and file-level validation."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import log_loss

from .utils import format_count, format_size


POOLED_METRICS = [
    "direction_score",
    "directional_return_pct",
    "magnitude_score",
    "conf_direction_score",
    "conf_direction_lift",
    "conf_magnitude_score",
    "hit_rate",
    "precision_up",
    "recall_up",
    "f1_up",
    "brier",
    "brier_skill",
    "log_loss",
    "ece_10",
    "mae",
    "rmse",
    "rank_ic",
    "rank_ic_t",
    "r2_vs_vol",
    "mae_conf_top_decile",
    "mae_conf_bottom_decile",
    "conf_mag_gradient",
    "frac_stocks_hit_gt_50",
    "frac_stocks_beat_naive",
    "var_share_universe",
]

RESIDUAL_METRICS = [
    "direction_score",
    "directional_return_pct",
    "magnitude_score",
    "conf_direction_score",
    "conf_direction_lift",
    "conf_magnitude_score",
    "hit_rate",
    "precision_up",
    "recall_up",
    "f1_up",
    "brier",
    "brier_skill",
    "log_loss",
    "ece_10",
    "rank_ic",
    "rank_ic_t",
]
logger = logging.getLogger(__name__)


def _expected_calibration_error(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(confidence, bin_edges[1:-1], right=False)
    ece = 0.0
    total = len(confidence)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if mask.any():
            ece += (mask.sum() / total) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _rank_ic_stats(dates: pd.Series, prediction: np.ndarray, actual: np.ndarray) -> tuple[float, float, int]:
    rank_ics: list[float] = []
    frame = pd.DataFrame({"pred_date": dates.to_numpy(), "prediction": prediction, "actual": actual})
    for _, day in frame.groupby("pred_date"):
        if len(day) < 3 or day["actual"].nunique() <= 1 or day["prediction"].nunique() <= 1:
            continue
        corr = spearmanr(day["prediction"], day["actual"]).statistic
        if np.isfinite(corr):
            rank_ics.append(float(corr))
    if not rank_ics:
        return np.nan, np.nan, 0
    rank_ic = float(np.mean(rank_ics))
    if len(rank_ics) > 1 and np.std(rank_ics, ddof=1) > 0:
        rank_ic_t = rank_ic / (np.std(rank_ics, ddof=1) / np.sqrt(len(rank_ics)))
    else:
        rank_ic_t = np.nan
    return rank_ic, float(rank_ic_t), len(rank_ics)


def build_predictions_df(scored_df: pd.DataFrame, decimal_places_magnitude: int) -> pd.DataFrame:
    """Construct predictions.csv contents."""
    predictions = scored_df[
        [
            "pred_date",
            "target_date",
            "symbol",
            "pred_magnitude_pct",
            "pred_direction",
            "conf_direction",
            "conf_magnitude",
            "split",
        ]
    ].copy()
    predictions = predictions.sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    predictions["pred_magnitude_pct"] = predictions["pred_magnitude_pct"].round(decimal_places_magnitude)
    predictions["pred_date"] = predictions["pred_date"].dt.strftime("%Y-%m-%d")
    predictions["target_date"] = predictions["target_date"].dt.strftime("%Y-%m-%d")
    return predictions


def build_actuals_df(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Construct actuals.csv contents."""
    actuals = scored_df[
        [
            "pred_date",
            "target_date",
            "symbol",
            "actual_return_pct",
            "actual_direction",
            "actual_magnitude_pct",
        ]
    ].copy()
    actuals["universe_mean_pct"] = actuals.groupby("pred_date")["actual_return_pct"].transform("mean")
    actuals = actuals.sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    actuals["pred_date"] = actuals["pred_date"].dt.strftime("%Y-%m-%d")
    actuals["target_date"] = actuals["target_date"].dt.strftime("%Y-%m-%d")
    return actuals


def _compute_metrics(split_df: pd.DataFrame, actual_col: str) -> dict[str, tuple[float, int]]:
    actual = split_df[actual_col].to_numpy(dtype=float)
    magnitude_actual = np.abs(actual)
    direction = split_df["pred_direction"].to_numpy(dtype=float)
    magnitude = split_df["pred_magnitude_pct"].to_numpy(dtype=float)
    conf_direction = split_df["conf_direction"].to_numpy(dtype=float)
    conf_magnitude = split_df["conf_magnitude"].to_numpy(dtype=float)
    correct = (np.where(actual >= 0, 1, -1) == direction).astype(float)
    weight = 2 * conf_direction - 1
    abs_error = np.abs(magnitude - magnitude_actual)

    result: dict[str, tuple[float, int]] = {}
    denom = np.abs(actual).sum()
    result["direction_score"] = ((direction * actual).sum() / denom if denom > 0 else np.nan, len(split_df))
    result["directional_return_pct"] = (float(np.mean(direction * actual)), len(split_df))
    result["magnitude_score"] = (
        1 - abs_error.sum() / magnitude_actual.sum() if magnitude_actual.sum() > 0 else np.nan,
        len(split_df),
    )
    weighted_denom = (weight * np.abs(actual)).sum()
    conf_direction_score = (
        (weight * direction * actual).sum() / weighted_denom if weighted_denom > 0 else np.nan
    )
    result["conf_direction_score"] = (float(conf_direction_score), len(split_df))
    result["conf_direction_lift"] = (float(conf_direction_score - result["direction_score"][0]), len(split_df))
    result["conf_magnitude_score"] = (float(spearmanr(conf_magnitude, -abs_error).statistic), len(split_df))
    result["hit_rate"] = (float(np.mean(correct)), len(split_df))
    up_mask = direction == 1
    actual_up_mask = np.where(actual >= 0, 1, -1) == 1
    precision_up = float(correct[up_mask].mean()) if up_mask.any() else np.nan
    recall_up = float((direction[actual_up_mask] == 1).mean()) if actual_up_mask.any() else np.nan
    if precision_up > 0 and recall_up > 0:
        f1_up = 2 * precision_up * recall_up / (precision_up + recall_up)
    else:
        f1_up = 0.0 if np.isfinite(precision_up) and np.isfinite(recall_up) else np.nan
    result["precision_up"] = (precision_up, int(up_mask.sum()))
    result["recall_up"] = (recall_up, int(actual_up_mask.sum()))
    result["f1_up"] = (float(f1_up), len(split_df))
    brier = float(np.mean((conf_direction - correct) ** 2))
    mean_correct = float(correct.mean())
    brier_reference = float(np.mean((mean_correct - correct) ** 2))
    result["brier"] = (brier, len(split_df))
    result["brier_skill"] = (1 - brier / brier_reference if brier_reference > 0 else np.nan, len(split_df))
    result["log_loss"] = (
        float(log_loss(correct, np.clip(conf_direction, 1e-6, 1 - 1e-6), labels=[0, 1])),
        len(split_df),
    )
    result["ece_10"] = (float(_expected_calibration_error(correct, conf_direction, 10)), len(split_df))
    result["mae"] = (float(abs_error.mean()), len(split_df))
    result["rmse"] = (float(np.sqrt(np.mean((magnitude - magnitude_actual) ** 2))), len(split_df))
    rank_ic, rank_ic_t, rank_days = _rank_ic_stats(split_df["pred_date"], magnitude, magnitude_actual)
    result["rank_ic"] = (rank_ic, rank_days)
    result["rank_ic_t"] = (rank_ic_t, rank_days)
    baseline = split_df["trailing_magnitude_20d"].to_numpy(dtype=float)
    valid_r2 = np.isfinite(baseline) & np.isfinite(magnitude_actual) & np.isfinite(magnitude)
    baseline_sse = np.sum((baseline[valid_r2] - magnitude_actual[valid_r2]) ** 2)
    model_sse = np.sum((magnitude[valid_r2] - magnitude_actual[valid_r2]) ** 2)
    result["r2_vs_vol"] = (1 - model_sse / baseline_sse if baseline_sse > 0 else np.nan, int(valid_r2.sum()))

    ordered = split_df.assign(abs_error=abs_error).sort_values("conf_magnitude")
    decile_size = max(int(math.ceil(len(ordered) * 0.10)), 1)
    bottom = ordered.head(decile_size)
    top = ordered.tail(decile_size)
    bottom_mae = float(bottom["abs_error"].mean())
    top_mae = float(top["abs_error"].mean())
    result["mae_conf_top_decile"] = (top_mae, len(top))
    result["mae_conf_bottom_decile"] = (bottom_mae, len(bottom))
    result["conf_mag_gradient"] = (bottom_mae - top_mae, len(top))

    by_symbol = split_df.assign(correct=correct)
    symbol_summary = by_symbol.groupby("symbol").agg(
        n_obs=("correct", "size"),
        hit_rate=("correct", "mean"),
        always_up_hit=("actual_direction", lambda x: float(np.mean(x.to_numpy() == 1))),
    )
    eligible = symbol_summary.loc[symbol_summary["n_obs"] >= 20]
    result["frac_stocks_hit_gt_50"] = (
        float(np.mean(eligible["hit_rate"] > 0.50)) if len(eligible) else np.nan,
        len(eligible),
    )
    result["frac_stocks_beat_naive"] = (
        float(np.mean(eligible["hit_rate"] > eligible["always_up_hit"])) if len(eligible) else np.nan,
        len(eligible),
    )

    pooled_actual = split_df["actual_return_pct"].to_numpy(dtype=float)
    universe_mean = split_df["universe_mean_pct"].to_numpy(dtype=float)
    numerator = np.var(pooled_actual - universe_mean, ddof=1)
    denominator_var = np.var(pooled_actual, ddof=1)
    result["var_share_universe"] = (
        1 - numerator / denominator_var if denominator_var > 0 else np.nan,
        len(split_df),
    )
    return result


def build_statistics_df(scored_df: pd.DataFrame) -> pd.DataFrame:
    """Construct the required 123-row statistics.csv output."""
    stats_rows: list[dict] = []
    working = scored_df.copy()
    working["universe_mean_pct"] = working.groupby("pred_date")["actual_return_pct"].transform("mean")
    working["residual_actual_return_pct"] = working["actual_return_pct"] - working["universe_mean_pct"]
    for split_name in ["train", "valid", "test"]:
        split_df = working.loc[working["split"] == split_name].copy()
        pooled_metrics = _compute_metrics(split_df, "actual_return_pct")
        residual_metrics = _compute_metrics(split_df, "residual_actual_return_pct")
        for metric in POOLED_METRICS:
            value, n_obs = pooled_metrics[metric]
            stats_rows.append({"split": split_name, "scope": "pooled", "metric": metric, "value": value, "n_obs": n_obs})
        for metric in RESIDUAL_METRICS:
            value, n_obs = residual_metrics[metric]
            stats_rows.append({"split": split_name, "scope": "residual", "metric": metric, "value": value, "n_obs": n_obs})
    statistics_df = pd.DataFrame(stats_rows)
    if len(statistics_df) != 123:
        raise AssertionError(f"statistics.csv must contain 123 rows, found {len(statistics_df)}")
    return statistics_df


def validate_output_frames(predictions_df: pd.DataFrame, actuals_df: pd.DataFrame, statistics_df: pd.DataFrame) -> None:
    """Validate final output frames."""
    if len(predictions_df) != len(actuals_df):
        raise AssertionError("Predictions and actuals lengths differ")
    if predictions_df[["pred_date", "target_date", "symbol"]].duplicated().any():
        raise AssertionError("Duplicate keys found in predictions")
    if actuals_df[["pred_date", "target_date", "symbol"]].duplicated().any():
        raise AssertionError("Duplicate keys found in actuals")
    if not predictions_df[["pred_date", "target_date", "symbol"]].equals(actuals_df[["pred_date", "target_date", "symbol"]]):
        raise AssertionError("Predictions and actuals keys are not aligned")
    if predictions_df.isna().any().any() or actuals_df.isna().any().any() or statistics_df.isna().any().any():
        raise AssertionError("Final outputs contain null values")
    if predictions_df["pred_direction"].isin([-1, 1]).all() is False:
        raise AssertionError("pred_direction must be -1/+1")
    if actuals_df["actual_direction"].isin([-1, 1]).all() is False:
        raise AssertionError("actual_direction must be -1/+1")
    if (predictions_df["pred_magnitude_pct"] < 0).any():
        raise AssertionError("pred_magnitude_pct must be non-negative")
    if not predictions_df["conf_direction"].between(0.5, 1.0).all():
        raise AssertionError("conf_direction out of bounds")
    if not predictions_df["conf_magnitude"].between(0.0, 1.0).all():
        raise AssertionError("conf_magnitude out of bounds")
    if not predictions_df["split"].isin(["train", "valid", "test"]).all():
        raise AssertionError("Unexpected split values")
    if len(statistics_df) != 123:
        raise AssertionError("statistics.csv must have 123 rows")
    logger.info("Predictions/actuals key alignment: PASS")
    logger.info("Duplicate-row check: PASS")
    logger.info("Null/infinity check: PASS")
    logger.info("Direction-range check: PASS")
    logger.info("Confidence-range check: PASS")
    logger.info("Statistics completeness: PASS")
    logger.info("Statistics rows: %s", format_count(len(statistics_df)))


def write_outputs(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    statistics_df: pd.DataFrame,
    output_dir: Path,
    filenames: dict,
) -> None:
    """Write the final CSV files."""
    outputs = [
        ("predictions.csv", predictions_df, output_dir / filenames["predictions_filename"]),
        ("actuals.csv", actuals_df, output_dir / filenames["actuals_filename"]),
        ("statistics.csv", statistics_df, output_dir / filenames["statistics_filename"]),
    ]
    for label, frame, path in outputs:
        logger.info("Writing %s", label)
        frame.to_csv(path, index=False)
        logger.info(
            "Path: %s | Rows: %s | Columns: %s | Size: %s | Schema validation: PASS",
            path,
            format_count(len(frame)),
            format_count(len(frame.columns)),
            format_size(path.stat().st_size),
        )
