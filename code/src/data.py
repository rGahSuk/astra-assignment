"""Data loading, validation, targets, and split construction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import format_count


DAILY_REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
MINUTE_REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
logger = logging.getLogger(__name__)


def resolve_repo_root(config_path: str | Path) -> Path:
    """Resolve the repository root from the config location."""
    config_path = Path(config_path).resolve()
    for candidate in [Path.cwd().resolve(), *config_path.parents]:
        if (candidate / "data").exists() and (candidate / "code").exists():
            return candidate
    return config_path.parent.parent


def resolve_path(root: Path, path_value: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def load_daily_file(file_path: Path) -> pd.DataFrame:
    """Load one daily parquet file and add symbol."""
    frame = pd.read_parquet(file_path).copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = set(DAILY_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{file_path.name} missing daily columns: {sorted(missing)}")
    frame = frame[DAILY_REQUIRED_COLUMNS].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["symbol"] = file_path.stem
    return frame


def load_minute_file(file_path: Path) -> pd.DataFrame:
    """Load one minute parquet file and add symbol."""
    frame = pd.read_parquet(file_path).copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    missing = set(MINUTE_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{file_path.name} missing minute columns: {sorted(missing)}")
    frame = frame[MINUTE_REQUIRED_COLUMNS].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame["symbol"] = file_path.stem
    return frame


def load_all_daily_data(daily_dir: Path) -> pd.DataFrame:
    """Load and concatenate all daily files."""
    daily_files = sorted(daily_dir.glob("*.parquet"))
    if not daily_files:
        raise FileNotFoundError(f"No daily parquet files found in {daily_dir}")
    logger.info("Daily directory: %s", daily_dir)
    logger.info("Found %s daily files", format_count(len(daily_files)))
    logger.info(
        "Daily symbols span: first=%s last=%s",
        daily_files[0].stem,
        daily_files[-1].stem,
    )
    frames = [load_daily_file(file_path) for file_path in daily_files]
    daily_df = pd.concat(frames, ignore_index=True)
    daily_df = daily_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("Loaded %s daily rows", format_count(len(daily_df)))
    logger.info(
        "Daily range: %s to %s",
        daily_df["date"].min().date(),
        daily_df["date"].max().date(),
    )
    return daily_df[["date", "symbol", "open", "high", "low", "close", "volume"]]


def list_minute_files(minute_dir: Path) -> list[Path]:
    """List minute parquet files."""
    files = sorted(minute_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No minute parquet files found in {minute_dir}")
    logger.info("Minute directory: %s", minute_dir)
    logger.info("Found %s minute files", format_count(len(files)))
    logger.info(
        "Minute symbols span: first=%s last=%s",
        files[0].stem,
        files[-1].stem,
    )
    return files


def validate_daily_data(daily_df: pd.DataFrame, expected_symbol_count: int) -> None:
    """Validate combined daily data."""
    duplicate_count = int(daily_df.duplicated(["symbol", "date"]).sum())
    non_positive_prices = int((daily_df[["open", "high", "low", "close"]] <= 0).sum().sum())
    negative_volume = int((daily_df["volume"] < 0).sum())
    high_below_low = int((daily_df["high"] < daily_df["low"]).sum())
    logger.info("Daily duplicate symbol-date rows: %s", format_count(duplicate_count))
    logger.info("Daily non-positive price cells: %s", format_count(non_positive_prices))
    logger.info("Daily negative-volume rows: %s", format_count(negative_volume))
    logger.info("Daily high-below-low rows: %s", format_count(high_below_low))
    if daily_df["symbol"].nunique() != expected_symbol_count:
        raise AssertionError(
            f"Expected {expected_symbol_count} daily symbols, found {daily_df['symbol'].nunique()}"
        )
    if daily_df[DAILY_REQUIRED_COLUMNS].isna().any().any():
        raise AssertionError("Daily data contains missing required values")
    if daily_df.duplicated(["symbol", "date"]).any():
        raise AssertionError("Duplicate symbol-date rows found")
    if (daily_df[["open", "high", "low", "close"]] <= 0).any().any():
        raise AssertionError("Daily prices must be positive")
    if (daily_df["volume"] < 0).any():
        raise AssertionError("Daily volume must be non-negative")
    if (daily_df["high"] < daily_df["low"]).any():
        raise AssertionError("Daily high below low detected")


def validate_minute_universe(minute_files: list[Path], daily_symbols: set[str], expected_symbol_count: int) -> None:
    """Validate minute symbol coverage against daily symbols."""
    minute_symbols = {file_path.stem for file_path in minute_files}
    logger.info("Minute symbol count: %s", format_count(len(minute_symbols)))
    if len(minute_symbols) != expected_symbol_count:
        raise AssertionError(
            f"Expected {expected_symbol_count} minute symbols, found {len(minute_symbols)}"
        )
    if minute_symbols != daily_symbols:
        raise AssertionError("Daily and minute symbol universes do not match")
    logger.info("Daily/minute symbol universes match")


def validate_minute_file(frame: pd.DataFrame, symbol: str) -> None:
    """Validate one minute dataframe."""
    duplicate_count = int(frame.duplicated(["symbol", "timestamp"]).sum())
    non_positive_prices = int((frame[["open", "high", "low", "close"]] <= 0).sum().sum())
    negative_volume = int((frame["volume"] < 0).sum())
    high_below_low = int((frame["high"] < frame["low"]).sum())
    if frame[MINUTE_REQUIRED_COLUMNS].isna().any().any():
        raise AssertionError(f"{symbol} minute data contains missing required values")
    if duplicate_count:
        raise AssertionError(f"{symbol} minute data contains duplicate timestamps")
    if non_positive_prices:
        raise AssertionError(f"{symbol} minute prices must be positive")
    if negative_volume:
        raise AssertionError(f"{symbol} minute volume must be non-negative")
    if high_below_low:
        raise AssertionError(f"{symbol} minute high below low detected")


def construct_targets(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Construct next-session targets from daily data."""
    target_df = daily_df.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    grouped = target_df.groupby("symbol", sort=False)
    target_df["target_date"] = grouped["date"].shift(-1)
    target_df["next_open"] = grouped["open"].shift(-1)
    target_df["actual_return_pct"] = ((target_df["next_open"] / target_df["close"]) - 1.0) * 100.0
    target_df["actual_direction"] = np.where(target_df["actual_return_pct"] >= 0, 1, -1)
    target_df["actual_magnitude_pct"] = target_df["actual_return_pct"].abs()
    target_df["calendar_gap_days"] = (target_df["target_date"] - target_df["date"]).dt.days
    rows_before_drop = len(target_df)
    target_df = target_df.dropna(
        subset=[
            "target_date",
            "next_open",
            "actual_return_pct",
            "actual_direction",
            "actual_magnitude_pct",
        ]
    ).copy()
    removed_rows = rows_before_drop - len(target_df)
    target_df = target_df.rename(columns={"date": "pred_date"})
    target_df["actual_direction"] = target_df["actual_direction"].astype("int8")
    if not (target_df["target_date"] > target_df["pred_date"]).all():
        raise AssertionError("Each target_date must be after pred_date")
    logger.info("Target rows retained: %s", format_count(len(target_df)))
    logger.info("Rows removed without next session: %s", format_count(removed_rows))
    return target_df


def apply_chronological_splits(
    frame: pd.DataFrame,
    validation_boundary: str,
    test_boundary: str,
    embargo_sessions: int,
) -> tuple[pd.DataFrame, pd.Index, pd.Index]:
    """Assign train/valid/test/embargo splits from available prediction dates."""
    split_df = frame.sort_values(["pred_date", "symbol"]).reset_index(drop=True).copy()
    valid_start = pd.Timestamp(validation_boundary)
    test_start = pd.Timestamp(test_boundary)
    available_pred_dates = pd.Index(split_df["pred_date"].dropna().sort_values().unique())
    first_embargo_dates = available_pred_dates[available_pred_dates >= valid_start][:embargo_sessions]
    second_embargo_dates = available_pred_dates[available_pred_dates >= test_start][:embargo_sessions]
    actual_valid_start = available_pred_dates[available_pred_dates > first_embargo_dates[-1]][0]
    actual_test_start = available_pred_dates[available_pred_dates > second_embargo_dates[-1]][0]

    split_df["split"] = "unused"
    split_df.loc[split_df["pred_date"] < valid_start, "split"] = "train"
    split_df.loc[split_df["pred_date"].isin(first_embargo_dates), "split"] = "embargo"
    split_df.loc[
        (split_df["pred_date"] >= actual_valid_start) & (split_df["pred_date"] < test_start),
        "split",
    ] = "valid"
    split_df.loc[split_df["pred_date"].isin(second_embargo_dates), "split"] = "embargo"
    split_df.loc[split_df["pred_date"] >= actual_test_start, "split"] = "test"

    train_end = split_df.loc[split_df["split"] == "train", "pred_date"].max()
    valid_start_actual = split_df.loc[split_df["split"] == "valid", "pred_date"].min()
    valid_end = split_df.loc[split_df["split"] == "valid", "pred_date"].max()
    test_start_actual = split_df.loc[split_df["split"] == "test", "pred_date"].min()
    if not (train_end < valid_start_actual and valid_end < test_start_actual):
        raise AssertionError("Split chronology failed")
    split_summary = split_df.groupby("split").agg(
        rows=("pred_date", "size"),
        dates=("pred_date", "nunique"),
        start=("pred_date", "min"),
        end=("pred_date", "max"),
    )
    logger.info("Chronological split complete")
    for split_name in ["train", "embargo", "valid", "test"]:
        row = split_summary.loc[split_name]
        if split_name == "embargo":
            logger.info(
                "%s | rows=%s | dates=%s",
                split_name,
                format_count(int(row["rows"])),
                format_count(int(row["dates"])),
            )
        else:
            logger.info(
                "%s | rows=%s | dates=%s | start=%s | end=%s",
                split_name,
                format_count(int(row["rows"])),
                format_count(int(row["dates"])),
                row["start"].date(),
                row["end"].date(),
            )
    logger.info("Embargo checks passed")
    logger.info("Split integrity checks passed")
    logger.info("No date overlap found")
    return split_df, first_embargo_dates, second_embargo_dates
