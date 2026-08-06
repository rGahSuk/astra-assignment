"""Model factories and train/score workflows."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMDeprecationWarning
from sklearn.linear_model import LogisticRegression

from .features import DIRECTION_FEATURE_COLUMNS, DIRECTION_NUMERIC_FEATURE_COLUMNS
from .utils import format_count


MAGNITUDE_EXCLUDED_COLUMNS = {
    "pred_date",
    "target_date",
    "split",
    "actual_return_pct",
    "actual_direction",
    "actual_magnitude_pct",
    "log_actual_magnitude",
    "universe_mean_pct",
    "next_open",
    "next_open_price",
    "target_return",
    "target_direction",
    "target_magnitude",
}
MAGNITUDE_SUSPICIOUS_TERMS = ["actual_", "target_", "next_open", "future_", "forward_", "label"]
logger = logging.getLogger(__name__)


@dataclass
class ValidationArtifacts:
    """Validation-stage outputs used for regression checks."""

    direction_score: float
    magnitude_score: float
    rank_ic: float
    conf_direction_score: float
    conf_magnitude_spearman: float
    magnitude_best_iteration: int


def make_direction_model(config: dict) -> lgb.LGBMClassifier:
    """Create the final direction classifier."""
    return lgb.LGBMClassifier(**config["direction"]["hyperparameters"])


def make_magnitude_model(config: dict, n_estimators: int | None = None) -> lgb.LGBMRegressor:
    """Create the magnitude regressor."""
    params = dict(config["magnitude"]["hyperparameters"])
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return lgb.LGBMRegressor(**params)


def make_direction_confidence_model(config: dict) -> LogisticRegression:
    """Create the final direction-confidence model."""
    return LogisticRegression(**config["direction_confidence"]["hyperparameters"])


def make_magnitude_confidence_model(config: dict) -> lgb.LGBMRegressor:
    """Create the final magnitude-confidence model."""
    return lgb.LGBMRegressor(**config["magnitude_confidence"]["hyperparameters"])


def build_magnitude_feature_list(model_df: pd.DataFrame) -> list[str]:
    """Reproduce the notebook's programmatic magnitude feature selection."""
    feature_columns: list[str] = []
    for column in model_df.columns:
        if column in MAGNITUDE_EXCLUDED_COLUMNS:
            continue
        if column == "symbol":
            feature_columns.append(column)
            continue
        if any(term in column.lower() for term in MAGNITUDE_SUSPICIOUS_TERMS):
            continue
        if pd.api.types.is_numeric_dtype(model_df[column]):
            feature_columns.append(column)
    feature_columns = list(dict.fromkeys(feature_columns))
    if "trailing_magnitude_20d" not in feature_columns:
        raise AssertionError("Magnitude features must include trailing_magnitude_20d")
    return feature_columns


def align_symbol_categories(train_frame: pd.DataFrame, *other_frames: pd.DataFrame) -> list[pd.DataFrame]:
    """Apply a common categorical dtype for symbol."""
    categories = sorted(train_frame["symbol"].astype(str).unique())
    dtype = pd.CategoricalDtype(categories=categories)
    aligned = []
    for frame in (train_frame, *other_frames):
        updated = frame.copy()
        updated["symbol"] = updated["symbol"].astype(str).astype(dtype)
        aligned.append(updated)
    return aligned


def prepare_direction_frames(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare warm-up filtered direction splits."""
    direction_df = model_df.loc[model_df["split"].isin(["train", "valid", "test"])].copy()
    direction_df = direction_df.dropna(subset=[
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
    ]).reset_index(drop=True)
    train_df = direction_df.loc[direction_df["split"] == "train"].copy()
    valid_df = direction_df.loc[direction_df["split"] == "valid"].copy()
    test_df = direction_df.loc[direction_df["split"] == "test"].copy()
    return train_df, valid_df, test_df


def fit_validation_magnitude_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, config: dict) -> tuple[lgb.LGBMRegressor, int]:
    """Fit the validation-stage magnitude model with early stopping."""
    magnitude_features = build_magnitude_feature_list(pd.concat([train_df, valid_df], ignore_index=True))
    x_train, x_valid = align_symbol_categories(
        train_df[magnitude_features],
        valid_df[magnitude_features],
    )
    y_train = np.log1p(train_df["actual_magnitude_pct"])
    y_valid = np.log1p(valid_df["actual_magnitude_pct"])
    logger.info(
        "Training validation magnitude model | rows=%s | valid_rows=%s | features=%s | categorical_features=1",
        format_count(len(train_df)),
        format_count(len(valid_df)),
        format_count(len(magnitude_features)),
    )
    logger.info(
        "Validation magnitude target summary | train_mean=%.6f | valid_mean=%.6f",
        float(y_train.mean()),
        float(y_valid.mean()),
    )
    model = make_magnitude_model(config)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=LGBMDeprecationWarning,
        )
        model.fit(
            x_train,
            y_train,
            categorical_feature=["symbol"],
            eval_set=[(x_valid, y_valid)],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
    logger.info("Magnitude early stopping selected iteration: %s", model.best_iteration_)
    return model, int(model.best_iteration_)


def score_direction_model(model: lgb.LGBMClassifier, feature_frame: pd.DataFrame, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Score direction probabilities and emitted directions."""
    probability_up = model.predict_proba(feature_frame)[:, 1]
    pred_direction = np.where(probability_up >= threshold, 1, -1).astype("int8")
    return probability_up, pred_direction


def score_magnitude_model(model: lgb.LGBMRegressor, feature_frame: pd.DataFrame) -> np.ndarray:
    """Score magnitude predictions and map back to percent units."""
    return np.clip(np.expm1(model.predict(feature_frame)), 0, None)


def create_direction_confidence_features(
    base_df: pd.DataFrame,
    raw_probability_up: np.ndarray,
    pred_direction: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """Create the direction-confidence feature frame."""
    frame = base_df[["symbol", "pred_date"]].copy().reset_index(drop=True)
    frame["raw_probability_up"] = raw_probability_up
    frame["pred_direction"] = pred_direction
    frame["probability_margin"] = np.abs(frame["raw_probability_up"] - 0.64)
    frame["probability_distance_from_half"] = np.abs(frame["raw_probability_up"] - 0.50)
    frame["emitted_probability"] = np.where(
        frame["pred_direction"] == 1,
        frame["raw_probability_up"],
        1 - frame["raw_probability_up"],
    )
    context_features = [
        "market_breadth",
        "aggregate_universe_volatility",
        "cross_sectional_return_dispersion",
        "day_of_week",
        "calendar_gap_days",
        "return_1d",
        "return_5d",
        "return_20d",
        "daily_volatility_20d",
        "volume_zscore_20d",
    ]
    for column in context_features:
        frame[column] = base_df[column].to_numpy()
    feature_columns = [
        "raw_probability_up",
        "probability_margin",
        "probability_distance_from_half",
        "emitted_probability",
        "pred_direction",
        *context_features,
    ]
    return frame, feature_columns


def create_magnitude_confidence_features(
    base_df: pd.DataFrame,
    pred_magnitude_pct: np.ndarray,
) -> tuple[pd.DataFrame, list[str]]:
    """Create the magnitude-confidence feature frame."""
    frame = base_df[["symbol", "pred_date", "trailing_magnitude_20d"]].copy().reset_index(drop=True)
    frame["pred_magnitude_pct"] = pred_magnitude_pct
    frame["prediction_vs_baseline_ratio"] = (
        frame["pred_magnitude_pct"] / frame["trailing_magnitude_20d"].replace(0, np.nan)
    )
    frame["prediction_baseline_difference"] = frame["pred_magnitude_pct"] - frame["trailing_magnitude_20d"]
    frame["log_predicted_magnitude"] = np.log1p(frame["pred_magnitude_pct"].clip(lower=0))
    context_features = [
        "return_1d",
        "return_5d",
        "return_20d",
        "daily_volatility_20d",
        "volume_zscore_20d",
        "volume_trend_5d_20d",
        "market_breadth",
        "aggregate_universe_volatility",
        "cross_sectional_return_dispersion",
        "day_of_week",
        "calendar_gap_days",
    ]
    for column in context_features:
        frame[column] = base_df[column].to_numpy()
    feature_columns = [
        "pred_magnitude_pct",
        "trailing_magnitude_20d",
        "log_predicted_magnitude",
        "prediction_vs_baseline_ratio",
        "prediction_baseline_difference",
        *context_features,
    ]
    return frame, feature_columns


def _fill_medians(train_frame: pd.DataFrame, score_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Replace infinities then apply train-only median fills."""
    fill_values = train_frame.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    filled_train = train_frame.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    filled_score = score_frame.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    return filled_train, filled_score, fill_values


def expected_error_to_confidence(expected_error: np.ndarray, reference_errors: np.ndarray) -> np.ndarray:
    """Convert predicted expected error into an empirical confidence percentile."""
    percentile = np.searchsorted(np.asarray(reference_errors, dtype=float), np.asarray(expected_error, dtype=float), side="right")
    percentile = percentile / len(reference_errors)
    return np.clip(1.0 - percentile, 0.0, 1.0)


def run_shadow_workflow(model_df: pd.DataFrame, config: dict) -> dict:
    """Create out-of-sample validation labels for the confidence models."""
    train_df, valid_df, _ = prepare_direction_frames(model_df)

    direction_train_x, direction_valid_x = align_symbol_categories(
        train_df[DIRECTION_FEATURE_COLUMNS],
        valid_df[DIRECTION_FEATURE_COLUMNS],
    )
    logger.info(
        "Training shadow direction model | rows=%s | features=%s | positive_target_fraction=%.4f | threshold=%.2f",
        format_count(len(train_df)),
        format_count(len(DIRECTION_FEATURE_COLUMNS)),
        float(train_df["actual_direction"].eq(1).mean()),
        config["direction"]["threshold"],
    )
    shadow_direction_model = make_direction_model(config)
    shadow_direction_model.fit(
        direction_train_x,
        train_df["actual_direction"].eq(1).astype("int8"),
        categorical_feature=["symbol"],
    )
    valid_probability_up, valid_pred_direction = score_direction_model(
        shadow_direction_model,
        direction_valid_x,
        config["direction"]["threshold"],
    )
    logger.info(
        "Shadow direction validation correctness rate: %.6f",
        float(np.mean(valid_pred_direction == valid_df["actual_direction"].to_numpy())),
    )

    magnitude_features = build_magnitude_feature_list(model_df)
    magnitude_train_x, magnitude_valid_x = align_symbol_categories(
        train_df[magnitude_features],
        valid_df[magnitude_features],
    )
    logger.info(
        "Training shadow magnitude model | rows=%s | features=%s",
        format_count(len(train_df)),
        format_count(len(magnitude_features)),
    )
    shadow_magnitude_model = make_magnitude_model(
        config,
        n_estimators=config["magnitude"]["final_best_iteration"],
    )
    shadow_magnitude_model.fit(
        magnitude_train_x,
        np.log1p(train_df["actual_magnitude_pct"]),
        categorical_feature=["symbol"],
    )
    valid_pred_magnitude_pct = score_magnitude_model(shadow_magnitude_model, magnitude_valid_x)
    magnitude_error = np.abs(valid_pred_magnitude_pct - valid_df["actual_magnitude_pct"].to_numpy())
    logger.info(
        "Shadow magnitude validation absolute-error summary | mean=%.6f | median=%.6f | max=%.6f",
        float(np.mean(magnitude_error)),
        float(np.median(magnitude_error)),
        float(np.max(magnitude_error)),
    )

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "shadow_direction_probability_up": valid_probability_up,
        "shadow_pred_direction": valid_pred_direction,
        "shadow_pred_magnitude_pct": valid_pred_magnitude_pct,
    }


def fit_final_primary_models(model_df: pd.DataFrame, config: dict) -> dict:
    """Fit the frozen primary models on train+valid and score train/valid/test."""
    train_df, valid_df, test_df = prepare_direction_frames(model_df)
    final_fit_df = pd.concat([train_df, valid_df], ignore_index=True)
    scored_df = pd.concat([train_df, valid_df, test_df], ignore_index=True).sort_values(["pred_date", "symbol"]).reset_index(drop=True)

    direction_fit_x, direction_score_x = align_symbol_categories(
        final_fit_df[DIRECTION_FEATURE_COLUMNS],
        scored_df[DIRECTION_FEATURE_COLUMNS],
    )
    logger.info(
        "Training final direction model | rows=%s | features=%s | categorical_features=1 | positive_target_fraction=%.4f | threshold=%.2f",
        format_count(len(final_fit_df)),
        format_count(len(DIRECTION_FEATURE_COLUMNS)),
        float(final_fit_df["actual_direction"].eq(1).mean()),
        config["direction"]["threshold"],
    )
    final_direction_model = make_direction_model(config)
    final_direction_model.fit(
        direction_fit_x,
        final_fit_df["actual_direction"].eq(1).astype("int8"),
        categorical_feature=["symbol"],
    )
    raw_probability_up, pred_direction = score_direction_model(
        final_direction_model,
        direction_score_x,
        config["direction"]["threshold"],
    )

    magnitude_features = build_magnitude_feature_list(model_df)
    magnitude_fit_x, magnitude_score_x = align_symbol_categories(
        final_fit_df[magnitude_features],
        scored_df[magnitude_features],
    )
    logger.info(
        "Training final magnitude model | rows=%s | features=%s | selected_tree_count=%s",
        format_count(len(final_fit_df)),
        format_count(len(magnitude_features)),
        format_count(config["magnitude"]["final_best_iteration"]),
    )
    final_magnitude_model = make_magnitude_model(
        config,
        n_estimators=config["magnitude"]["final_best_iteration"],
    )
    final_magnitude_model.fit(
        magnitude_fit_x,
        np.log1p(final_fit_df["actual_magnitude_pct"]),
        categorical_feature=["symbol"],
    )
    pred_magnitude_pct = score_magnitude_model(final_magnitude_model, magnitude_score_x)

    scored_df = scored_df.copy()
    scored_df["raw_probability_up"] = raw_probability_up
    scored_df["pred_direction"] = pred_direction
    scored_df["pred_magnitude_pct"] = pred_magnitude_pct
    logger.info(
        "Primary model scoring summary | scored_rows=%s | predicted_up_fraction=%.6f | mean_pred_magnitude_pct=%.6f",
        format_count(len(scored_df)),
        float(np.mean(pred_direction == 1)),
        float(np.mean(pred_magnitude_pct)),
    )
    return {
        "scored_df": scored_df,
        "direction_model": final_direction_model,
        "magnitude_model": final_magnitude_model,
        "magnitude_features": magnitude_features,
    }


def fit_final_confidence_models(shadow_outputs: dict, final_scored_df: pd.DataFrame, config: dict) -> dict:
    """Fit the frozen confidence models on validation out-of-sample labels and score all splits."""
    valid_df = shadow_outputs["valid_df"].copy().reset_index(drop=True)
    valid_df["direction_correct"] = (
        shadow_outputs["shadow_pred_direction"] == valid_df["actual_direction"].to_numpy()
    ).astype("int8")
    logger.info(
        "Direction confidence setup | correctness_rate=%.6f | rows=%s",
        float(valid_df["direction_correct"].mean()),
        format_count(len(valid_df)),
    )

    direction_train_features, direction_feature_columns = create_direction_confidence_features(
        valid_df,
        shadow_outputs["shadow_direction_probability_up"],
        shadow_outputs["shadow_pred_direction"],
    )
    direction_train_x, direction_score_x, _ = _fill_medians(
        direction_train_features[direction_feature_columns],
        create_direction_confidence_features(
            final_scored_df,
            final_scored_df["raw_probability_up"].to_numpy(),
            final_scored_df["pred_direction"].to_numpy(),
        )[0][direction_feature_columns],
    )
    logger.info(
        "Training direction confidence model | rows=%s | feature_count=%s",
        format_count(len(direction_train_x)),
        format_count(len(direction_feature_columns)),
    )
    direction_confidence_model = make_direction_confidence_model(config)
    direction_confidence_model.fit(direction_train_x, valid_df["direction_correct"])
    conf_direction = np.clip(
        direction_confidence_model.predict_proba(direction_score_x)[:, 1],
        config["direction_confidence"]["clip_min"],
        config["direction_confidence"]["clip_max"],
    )
    logger.info(
        "Direction confidence output summary | mean=%.6f | min=%.6f | max=%.6f",
        float(np.mean(conf_direction)),
        float(np.min(conf_direction)),
        float(np.max(conf_direction)),
    )

    magnitude_train_features, magnitude_feature_columns = create_magnitude_confidence_features(
        valid_df,
        shadow_outputs["shadow_pred_magnitude_pct"],
    )
    magnitude_train_features["absolute_magnitude_error"] = np.abs(
        shadow_outputs["shadow_pred_magnitude_pct"] - valid_df["actual_magnitude_pct"].to_numpy()
    )
    magnitude_train_features["log_absolute_magnitude_error"] = np.log1p(
        magnitude_train_features["absolute_magnitude_error"]
    )
    logger.info(
        "Magnitude confidence setup | abs_error_mean=%.6f | abs_error_median=%.6f | abs_error_max=%.6f",
        float(magnitude_train_features["absolute_magnitude_error"].mean()),
        float(magnitude_train_features["absolute_magnitude_error"].median()),
        float(magnitude_train_features["absolute_magnitude_error"].max()),
    )
    magnitude_train_x, magnitude_score_x, _ = _fill_medians(
        magnitude_train_features[magnitude_feature_columns],
        create_magnitude_confidence_features(
            final_scored_df,
            final_scored_df["pred_magnitude_pct"].to_numpy(),
        )[0][magnitude_feature_columns],
    )
    logger.info(
        "Training magnitude confidence model | rows=%s | feature_count=%s",
        format_count(len(magnitude_train_x)),
        format_count(len(magnitude_feature_columns)),
    )
    magnitude_confidence_model = make_magnitude_confidence_model(config)
    magnitude_confidence_model.fit(
        magnitude_train_x,
        magnitude_train_features["log_absolute_magnitude_error"],
    )
    training_expected_error = np.clip(np.expm1(magnitude_confidence_model.predict(magnitude_train_x)), 0, None)
    scored_expected_error = np.clip(np.expm1(magnitude_confidence_model.predict(magnitude_score_x)), 0, None)
    conf_magnitude = expected_error_to_confidence(
        scored_expected_error,
        np.sort(training_expected_error),
    )
    logger.info(
        "Magnitude confidence output summary | expected_error_mean=%.6f | expected_error_min=%.6f | expected_error_max=%.6f | conf_mean=%.6f | conf_min=%.6f | conf_max=%.6f",
        float(np.mean(scored_expected_error)),
        float(np.min(scored_expected_error)),
        float(np.max(scored_expected_error)),
        float(np.mean(conf_magnitude)),
        float(np.min(conf_magnitude)),
        float(np.max(conf_magnitude)),
    )

    scored_with_conf = final_scored_df.copy()
    scored_with_conf["conf_direction"] = conf_direction
    scored_with_conf["conf_magnitude"] = conf_magnitude
    scored_with_conf["expected_magnitude_error"] = scored_expected_error
    return {
        "scored_df": scored_with_conf,
        "direction_confidence_model": direction_confidence_model,
        "magnitude_confidence_model": magnitude_confidence_model,
        "direction_confidence_features": direction_feature_columns,
        "magnitude_confidence_features": magnitude_feature_columns,
    }
