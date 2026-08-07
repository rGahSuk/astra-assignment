"""Utility helpers for the production pipeline."""

from __future__ import annotations

import contextlib
import logging
import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

import lightgbm
import pyarrow
import scipy
import sklearn

import numpy as np
import pandas as pd
import yaml


DEFAULT_LOGGING_CONFIG = {
    "console_level": "INFO",
    "file_level": "DEBUG",
    "save_to_file": True,
    "log_dir": "logs",
    "progress_every_symbols": 10,
    "show_timestamps": True,
    "log_feature_missingness": True,
    "log_model_parameters": True,
    "fail_on_benchmark_drift": False,
}
DEFAULT_BENCHMARKS = {
    "direction_score": {"reference": 0.29481723269301396, "tolerance": 0.005},
    "magnitude_score": {"reference": 0.254340, "tolerance": 0.005},
    "rank_ic": {"reference": 0.245931, "tolerance": 0.01},
    "conf_direction_score": {"reference": 0.268893, "tolerance": 0.01},
    "conf_magnitude_spearman": {"reference": 0.299122, "tolerance": 0.02},
}


def load_yaml(path: str | Path) -> dict:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return the resolved path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_logging_config(config: dict) -> dict:
    """Return logging config with defaults applied."""
    resolved = dict(DEFAULT_LOGGING_CONFIG)
    resolved.update(config.get("logging", {}))
    return resolved


def get_benchmark_config(config: dict) -> dict:
    """Return validation benchmark references with defaults applied."""
    resolved = {name: values.copy() for name, values in DEFAULT_BENCHMARKS.items()}
    for metric_name, overrides in config.get("benchmarks", {}).items():
        if metric_name in resolved:
            resolved[metric_name].update(overrides)
        else:
            resolved[metric_name] = dict(overrides)
    return resolved


def _coerce_log_level(value: str | int) -> int:
    """Map a config value to a logging level."""
    if isinstance(value, int):
        return value
    return getattr(logging, str(value).upper(), logging.INFO)


def setup_logging(config: dict, project_root: Path) -> logging.Logger:
    """Configure central console and file logging for the pipeline."""
    logging_config = get_logging_config(config)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    console_format = "%(asctime)s | %(levelname)-7s | %(message)s"
    file_format = "%(asctime)s | %(levelname)s | %(name)s | %(funcName)s | %(message)s"
    console_datefmt = "%H:%M:%S" if logging_config.get("show_timestamps", True) else None

    console_handler = logging.StreamHandler()
    console_handler.setLevel(_coerce_log_level(logging_config["console_level"]))
    console_handler.setFormatter(logging.Formatter(console_format, datefmt=console_datefmt))
    logger.addHandler(console_handler)

    log_file_path: Path | None = None
    if logging_config.get("save_to_file", True):
        log_dir = ensure_dir(project_root / logging_config["log_dir"])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir / f"pipeline_{timestamp}.log"
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setLevel(_coerce_log_level(logging_config["file_level"]))
        file_handler.setFormatter(logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    logger.log_file_path = log_file_path  # type: ignore[attr-defined]
    logger.logging_config = logging_config  # type: ignore[attr-defined]
    return logger


def set_global_seed(seed: int) -> None:
    """Seed supported random generators."""
    random.seed(seed)
    np.random.seed(seed)


@contextlib.contextmanager
def log_stage(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log the lifecycle of a pipeline stage."""
    start = time.perf_counter()
    logger.info("Starting: %s", label)
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - start
        logger.error(
            "Failed: %s | elapsed=%s | error=%s",
            label,
            format_duration(elapsed),
            exc,
        )
        raise
    elapsed = time.perf_counter() - start
    logger.info("Completed: %s | elapsed=%s", label, format_duration(elapsed))


@contextlib.contextmanager
def timed_block(label: str, logger: logging.Logger) -> Iterator[None]:
    """Backward-compatible alias for stage timing."""
    with log_stage(logger, label):
        yield


def format_duration(seconds: float) -> str:
    """Format elapsed seconds for logs."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, rem_minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(rem_minutes)}m {rem:.0f}s"


def format_count(value: int) -> str:
    """Format row or file counts with separators."""
    return f"{int(value):,}"


def format_size(num_bytes: int) -> str:
    """Format file sizes for logs."""
    size = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0


def summarize_series(series: pd.Series) -> dict[str, float]:
    """Return simple summary stats for logging."""
    valid = series.dropna()
    if valid.empty:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(valid.mean()),
        "min": float(valid.min()),
        "max": float(valid.max()),
    }


def package_versions() -> dict[str, str]:
    """Return package versions for startup logs."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "lightgbm": lightgbm.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def log_benchmark_result(
    logger: logging.Logger,
    metric_name: str,
    observed: float,
    reference: float,
    tolerance: float,
    fail_on_drift: bool,
) -> None:
    """Log a benchmark comparison and optionally fail on drift."""
    delta = observed - reference
    status = "PASS" if abs(delta) <= tolerance else "WARN"
    log_fn = logger.info if status == "PASS" else logger.warning
    log_fn(
        "%s | observed=%.6f | reference=%.6f | delta=%+.6f | tolerance=%.6f | %s",
        metric_name,
        observed,
        reference,
        delta,
        tolerance,
        status,
    )
    if status != "PASS" and fail_on_drift:
        raise RuntimeError(
            f"Benchmark drift for {metric_name}: observed={observed:.6f} reference={reference:.6f}"
        )


def assert_no_infinite_values(
    frame: pd.DataFrame,
    columns: list[str],
    context: str,
) -> None:
    """Ensure the selected numeric columns are finite or null."""
    if not columns:
        return
    values = frame[columns].to_numpy(dtype=float, copy=False)
    inf_mask = np.isinf(values)
    if inf_mask.any():
        raise AssertionError(f"Infinite values found in {context}")


def assert_between(
    series: pd.Series,
    lower: float,
    upper: float,
    context: str,
) -> None:
    """Validate a bounded numeric series."""
    valid = series.dropna()
    if not valid.between(lower, upper).all():
        raise AssertionError(f"{context} must be between {lower} and {upper}")
