"""Feature engineering logic reproduced from the research notebooks."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_minute_file, validate_minute_file
from .utils import format_count, format_duration


DAILY_FEATURE_COLUMNS = [
    "lagged_overnight_return_1d",
    "return_1d",
    "return_5d",
    "return_20d",
    "daily_volatility_20d",
    "overnight_std_20d",
    "daily_volatility_5d",
    "gap_mean_20d",
    "gap_positive_fraction_20d",
    "volume_zscore_20d",
    "volume_trend_5d_20d",
    "calendar_gap_days",
    "day_of_week",
]

MINUTE_FEATURE_COLUMNS = [
    "intraday_realized_volatility_pct",
    "close_auction_return_concentration",
    "close_auction_volume_concentration",
    "morning_vs_afternoon_return_pct",
    "close_vwap_deviation_pct",
    "amihud_illiquidity",
]

MINUTE_QUALITY_COLUMNS = ["minute_bar_count", "is_complete_session"]

CROSS_SECTIONAL_FEATURE_COLUMNS = [
    "return_1d_rank_pct",
    "return_1d_zscore",
    "rolling_beta_to_universe_60d",
    "cross_sectional_return_dispersion",
    "market_breadth",
    "aggregate_universe_volatility",
]

DIRECTION_NUMERIC_FEATURE_COLUMNS = (
    DAILY_FEATURE_COLUMNS + MINUTE_FEATURE_COLUMNS + CROSS_SECTIONAL_FEATURE_COLUMNS
)
DIRECTION_FEATURE_COLUMNS = DIRECTION_NUMERIC_FEATURE_COLUMNS + ["symbol"]
logger = logging.getLogger(__name__)


def create_daily_features(target_df: pd.DataFrame) -> pd.DataFrame:
    """Create the V1 causal daily features."""
    logger.info(
        "Daily features | count=%s | input_rows=%s",
        format_count(len(DAILY_FEATURE_COLUMNS)),
        format_count(len(target_df)),
    )
    frame = target_df.sort_values(["symbol", "pred_date"]).reset_index(drop=True).copy()
    grouped = frame.groupby("symbol", sort=False, group_keys=False)

    frame["lagged_overnight_return_1d"] = grouped["actual_return_pct"].shift(1)
    frame["gap_mean_20d"] = frame.groupby("symbol")["lagged_overnight_return_1d"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    frame["overnight_std_20d"] = frame.groupby("symbol")["lagged_overnight_return_1d"].transform(
        lambda series: series.rolling(window=20, min_periods=20).std()
    )
    frame["gap_positive_fraction_20d"] = frame.groupby("symbol")["lagged_overnight_return_1d"].transform(
        lambda series: series.ge(0).where(series.notna()).rolling(window=20, min_periods=20).mean()
    )
    frame["return_1d"] = grouped["close"].pct_change(1) * 100
    frame["return_5d"] = grouped["close"].pct_change(5) * 100
    frame["return_20d"] = grouped["close"].pct_change(20) * 100
    frame["daily_volatility_5d"] = frame.groupby("symbol")["return_1d"].transform(
        lambda series: series.rolling(window=5, min_periods=5).std()
    )
    frame["daily_volatility_20d"] = frame.groupby("symbol")["return_1d"].transform(
        lambda series: series.rolling(window=20, min_periods=20).std()
    )
    frame["volume_mean_20d_lagged"] = grouped["volume"].transform(
        lambda series: series.shift(1).rolling(window=20, min_periods=20).mean()
    )
    frame["volume_std_20d_lagged"] = grouped["volume"].transform(
        lambda series: series.shift(1).rolling(window=20, min_periods=20).std()
    )
    frame["volume_zscore_20d"] = (
        frame["volume"] - frame["volume_mean_20d_lagged"]
    ) / frame["volume_std_20d_lagged"]
    frame["volume_mean_5d"] = grouped["volume"].transform(
        lambda series: series.rolling(window=5, min_periods=5).mean()
    )
    frame["volume_mean_20d"] = grouped["volume"].transform(
        lambda series: series.rolling(window=20, min_periods=20).mean()
    )
    frame["volume_trend_5d_20d"] = frame["volume_mean_5d"] / frame["volume_mean_20d"]
    frame["day_of_week"] = frame["pred_date"].dt.dayofweek
    frame[DAILY_FEATURE_COLUMNS] = frame[DAILY_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    logger.info(
        "Daily features complete | output_rows=%s | symbols=%s",
        format_count(len(frame)),
        format_count(frame["symbol"].nunique()),
    )
    return frame


def create_minute_daily_features(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert one symbol's minute bars into one row per prediction date."""
    minute_df = frame.copy()
    minute_df.columns = [str(column).strip().lower() for column in minute_df.columns]
    minute_df["timestamp"] = pd.to_datetime(minute_df["timestamp"], errors="raise")
    minute_df = minute_df.sort_values("timestamp").reset_index(drop=True)
    minute_df["pred_date"] = minute_df["timestamp"].dt.normalize()

    daily_rows: list[dict] = []
    for pred_date, session_df in minute_df.groupby("pred_date", sort=True):
        session_df = session_df.sort_values("timestamp").reset_index(drop=True)
        minute_bar_count = len(session_df)
        is_complete_session = minute_bar_count == 375

        session_df["minute_return"] = session_df["close"].pct_change()
        valid_minute_returns = session_df["minute_return"].dropna()
        if len(valid_minute_returns) > 0:
            intraday_realized_volatility_pct = np.sqrt(np.square(valid_minute_returns).sum()) * 100
        else:
            intraday_realized_volatility_pct = np.nan

        session_open = session_df["open"].iloc[0]
        session_close = session_df["close"].iloc[-1]
        full_session_return = session_close / session_open - 1 if session_open > 0 else np.nan

        last_30_rows = session_df.tail(min(30, minute_bar_count)).copy()
        if len(last_30_rows) > 0 and last_30_rows["open"].iloc[0] > 0:
            last_30_return = last_30_rows["close"].iloc[-1] / last_30_rows["open"].iloc[0] - 1
        else:
            last_30_return = np.nan

        if pd.notna(full_session_return) and abs(full_session_return) > 1e-12:
            close_auction_return_concentration = last_30_return / full_session_return
        else:
            close_auction_return_concentration = np.nan

        full_session_volume = session_df["volume"].sum()
        last_30_volume = last_30_rows["volume"].sum()
        if full_session_volume > 0:
            close_auction_volume_concentration = last_30_volume / full_session_volume
        else:
            close_auction_volume_concentration = np.nan

        session_midpoint = minute_bar_count // 2
        morning_session = session_df.iloc[:session_midpoint]
        afternoon_session = session_df.iloc[session_midpoint:]
        if len(morning_session) > 0 and morning_session["open"].iloc[0] > 0:
            morning_return = morning_session["close"].iloc[-1] / morning_session["open"].iloc[0] - 1
        else:
            morning_return = np.nan
        if len(afternoon_session) > 0 and afternoon_session["open"].iloc[0] > 0:
            afternoon_return = afternoon_session["close"].iloc[-1] / afternoon_session["open"].iloc[0] - 1
        else:
            afternoon_return = np.nan
        if pd.notna(morning_return) and pd.notna(afternoon_return):
            morning_vs_afternoon_return_pct = (morning_return - afternoon_return) * 100
        else:
            morning_vs_afternoon_return_pct = np.nan

        typical_price = (session_df["high"] + session_df["low"] + session_df["close"]) / 3
        vwap_denominator = session_df["volume"].sum()
        if vwap_denominator > 0:
            session_vwap = (typical_price * session_df["volume"]).sum() / vwap_denominator
        else:
            session_vwap = np.nan
        if pd.notna(session_vwap) and session_vwap != 0:
            close_vwap_deviation_pct = (session_close / session_vwap - 1) * 100
        else:
            close_vwap_deviation_pct = np.nan

        valid_amihud_rows = session_df[session_df["minute_return"].notna() & (session_df["volume"] > 0)]
        if len(valid_amihud_rows) > 0:
            amihud_illiquidity = (
                valid_amihud_rows["minute_return"].abs() / valid_amihud_rows["volume"]
            ).mean()
        else:
            amihud_illiquidity = np.nan

        daily_rows.append(
            {
                "pred_date": pred_date,
                "symbol": symbol,
                "intraday_realized_volatility_pct": intraday_realized_volatility_pct,
                "close_auction_return_concentration": close_auction_return_concentration,
                "close_auction_volume_concentration": close_auction_volume_concentration,
                "morning_vs_afternoon_return_pct": morning_vs_afternoon_return_pct,
                "close_vwap_deviation_pct": close_vwap_deviation_pct,
                "amihud_illiquidity": amihud_illiquidity,
                "minute_bar_count": minute_bar_count,
                "is_complete_session": is_complete_session,
            }
        )

    return pd.DataFrame(daily_rows)


def build_minute_feature_panel(minute_files: list[Path], logging_config: dict | None = None) -> pd.DataFrame:
    """Process minute files one symbol at a time."""
    logging_config = logging_config or {}
    progress_every = int(logging_config.get("progress_every_symbols", 10))
    log_every_symbol = str(logging_config.get("console_level", "INFO")).upper() == "DEBUG"
    logger.info(
        "Minute-derived features | feature_count=%s | symbols=%s",
        format_count(len(MINUTE_FEATURE_COLUMNS)),
        format_count(len(minute_files)),
    )
    frames: list[pd.DataFrame] = []
    zero_volume_bars_total = 0
    zero_volume_symbols: set[str] = set()
    incomplete_sessions_total = 0
    start_time = time.perf_counter()
    for file_path in minute_files:
        symbol_start = time.perf_counter()
        minute_df = load_minute_file(file_path)
        validate_minute_file(minute_df, file_path.stem)
        zero_volume_bars = int((minute_df["volume"] == 0).sum())
        zero_volume_bars_total += zero_volume_bars
        if zero_volume_bars > 0:
            zero_volume_symbols.add(file_path.stem)
        symbol_features = create_minute_daily_features(minute_df, file_path.stem)
        incomplete_sessions = int((~symbol_features["is_complete_session"]).sum()) if not symbol_features.empty else 0
        incomplete_sessions_total += incomplete_sessions
        frames.append(symbol_features)
        completed = len(frames)
        elapsed = time.perf_counter() - start_time
        symbol_elapsed = time.perf_counter() - symbol_start
        eta_seconds = (elapsed / completed) * (len(minute_files) - completed) if completed else 0.0
        if log_every_symbol or completed == 1 or completed % progress_every == 0 or completed == len(minute_files):
            logger.info(
                "Minute features [%s/%s] %s | sessions=%s | incomplete_sessions=%s | symbol_elapsed=%s | elapsed=%s | ETA=%s",
                format_count(completed),
                format_count(len(minute_files)),
                file_path.stem,
                format_count(len(symbol_features)),
                format_count(incomplete_sessions),
                format_duration(symbol_elapsed),
                format_duration(elapsed),
                format_duration(eta_seconds),
            )
    minute_features = pd.concat(frames, ignore_index=True)
    minute_features = minute_features.sort_values(["symbol", "pred_date"]).reset_index(drop=True)
    minute_features[MINUTE_FEATURE_COLUMNS] = minute_features[MINUTE_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    logger.info(
        "Minute-derived features complete | rows=%s | zero_volume_bars=%s | incomplete_sessions=%s",
        format_count(len(minute_features)),
        format_count(zero_volume_bars_total),
        format_count(incomplete_sessions_total),
    )
    if zero_volume_bars_total > 0:
        logger.warning(
            "Found %s zero-volume minute bars across %s symbols",
            format_count(zero_volume_bars_total),
            format_count(len(zero_volume_symbols)),
        )
    return minute_features


def add_cross_sectional_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """Add the V1 cross-sectional features."""
    logger.info(
        "Cross-sectional features | count=%s | input_rows=%s | dates=%s",
        format_count(len(CROSS_SECTIONAL_FEATURE_COLUMNS)),
        format_count(len(model_df)),
        format_count(model_df["pred_date"].nunique()),
    )
    frame = model_df.sort_values(["pred_date", "symbol"]).reset_index(drop=True).copy()
    frame["return_1d_rank_pct"] = frame.groupby("pred_date")["return_1d"].rank(method="average", pct=True)
    return_mean_by_date = frame.groupby("pred_date")["return_1d"].transform("mean")
    return_std_by_date = frame.groupby("pred_date")["return_1d"].transform("std")
    frame["return_1d_zscore"] = (frame["return_1d"] - return_mean_by_date) / return_std_by_date
    frame["universe_mean_return_1d"] = frame.groupby("pred_date")["return_1d"].transform("mean")
    frame["cross_sectional_return_dispersion"] = frame.groupby("pred_date")["return_1d"].transform("std")
    frame["market_breadth"] = frame["return_1d"].gt(0).groupby(frame["pred_date"]).transform("mean")
    frame["aggregate_universe_volatility"] = frame.groupby("pred_date")["daily_volatility_20d"].transform("median")
    frame["stock_market_product"] = frame["return_1d"] * frame["universe_mean_return_1d"]
    frame["market_return_squared"] = frame["universe_mean_return_1d"] ** 2
    frame["rolling_mean_stock_return_60d"] = frame.groupby("symbol")["return_1d"].transform(
        lambda series: series.rolling(window=60, min_periods=40).mean()
    )
    frame["rolling_mean_market_return_60d"] = frame.groupby("symbol")["universe_mean_return_1d"].transform(
        lambda series: series.rolling(window=60, min_periods=40).mean()
    )
    frame["rolling_mean_stock_market_product_60d"] = frame.groupby("symbol")["stock_market_product"].transform(
        lambda series: series.rolling(window=60, min_periods=40).mean()
    )
    frame["rolling_mean_market_squared_60d"] = frame.groupby("symbol")["market_return_squared"].transform(
        lambda series: series.rolling(window=60, min_periods=40).mean()
    )
    frame["rolling_market_variance_60d"] = (
        frame["rolling_mean_market_squared_60d"] - frame["rolling_mean_market_return_60d"] ** 2
    )
    frame["rolling_stock_market_covariance_60d"] = (
        frame["rolling_mean_stock_market_product_60d"]
        - frame["rolling_mean_stock_return_60d"] * frame["rolling_mean_market_return_60d"]
    )
    frame["rolling_beta_to_universe_60d"] = (
        frame["rolling_stock_market_covariance_60d"] / frame["rolling_market_variance_60d"]
    )
    frame[CROSS_SECTIONAL_FEATURE_COLUMNS] = frame[CROSS_SECTIONAL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    logger.info("Cross-sectional features complete | output_rows=%s", format_count(len(frame)))
    return frame


def add_trailing_magnitude(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the trailing-magnitude baseline used by the magnitude model."""
    output = frame.sort_values(["symbol", "pred_date"]).reset_index(drop=True).copy()
    output["trailing_magnitude_20d"] = output.groupby("symbol")["actual_magnitude_pct"].transform(
        lambda series: series.shift(1).rolling(window=20, min_periods=20).mean()
    )
    return output


def build_feature_panel(target_df: pd.DataFrame, minute_files: list[Path], logging_config: dict | None = None) -> pd.DataFrame:
    """Build the full modelling panel from targets and raw minute files."""
    daily_features = create_daily_features(target_df)
    minute_features = build_minute_feature_panel(minute_files, logging_config=logging_config)
    merged = daily_features.merge(minute_features, on=["symbol", "pred_date"], how="left", validate="one_to_one")
    merged = add_cross_sectional_features(merged)
    magnitude_feature_count = len(
        [
            column
            for column in merged.columns
            if column == "symbol" or pd.api.types.is_numeric_dtype(merged[column])
        ]
    )
    logger.info(
        "Final modelling panel | shape=(%s, %s) | symbols=%s | date_range=%s to %s | duplicates=%s | direction_features=%s | magnitude_candidate_columns=%s",
        format_count(len(merged)),
        format_count(len(merged.columns)),
        format_count(merged["symbol"].nunique()),
        merged["pred_date"].min().date(),
        merged["pred_date"].max().date(),
        format_count(int(merged.duplicated(["symbol", "pred_date"]).sum())),
        format_count(len(DIRECTION_FEATURE_COLUMNS)),
        format_count(magnitude_feature_count),
    )
    if logging_config and logging_config.get("log_feature_missingness", True):
        missing_counts = merged[DIRECTION_NUMERIC_FEATURE_COLUMNS].isna().sum().sort_values(ascending=False).head(5)
        if missing_counts.max() > 0:
            logger.info("Top feature missingness:")
            for feature_name, missing_count in missing_counts.items():
                logger.info("%s: %s", feature_name, format_count(int(missing_count)))
    return merged


def assert_no_feature_leakage(feature_columns: list[str]) -> None:
    """Reject obviously leaky feature names."""
    suspicious_terms = ["actual_", "target_", "next_open", "future_", "forward_", "label"]
    bad_columns = [
        column
        for column in feature_columns
        if any(term in column.lower() for term in suspicious_terms)
    ]
    if bad_columns:
        raise AssertionError(f"Leaky feature columns detected: {bad_columns}")
