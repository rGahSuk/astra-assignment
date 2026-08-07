"""Entry point for the Astra assignment production pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMDeprecationWarning
from scipy.stats import spearmanr

from src.data import (
    apply_chronological_splits,
    construct_targets,
    load_all_daily_data,
    list_minute_files,
    resolve_path,
    resolve_repo_root,
    validate_daily_data,
    validate_minute_universe,
)
from src.evaluate import (
    build_actuals_df,
    build_predictions_df,
    build_statistics_df,
    validate_output_frames,
    write_outputs,
)
from src.features import (
    DIRECTION_FEATURE_COLUMNS,
    build_feature_panel,
    add_trailing_magnitude,
    assert_no_feature_leakage,
)
from src.models import (
    build_magnitude_feature_list,
    create_direction_confidence_features,
    fit_final_confidence_models,
    fit_final_primary_models,
    make_direction_model,
    prepare_direction_frames,
    run_shadow_workflow,
    score_direction_model,
    score_magnitude_model,
    align_symbol_categories,
)
from src.utils import assert_no_infinite_values, ensure_dir, load_yaml, set_global_seed, setup_logging
from src.utils import (
    format_count,
    format_duration,
    format_size,
    get_benchmark_config,
    get_logging_config,
    log_benchmark_result,
    log_stage,
    package_versions,
)


logger = logging.getLogger(__name__)
NOTEBOOK_MAGNITUDE_HELPER_COLUMNS = {
    "stock_market_product",
    "market_return_squared",
    "rolling_mean_stock_return_60d",
    "rolling_mean_market_return_60d",
    "rolling_mean_stock_market_product_60d",
    "rolling_mean_market_squared_60d",
    "rolling_market_variance_60d",
    "rolling_stock_market_covariance_60d",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_or_build_panel(config: dict, config_path: Path, logger) -> tuple[pd.DataFrame, dict]:
    """Load cached processed data when allowed, otherwise rebuild from raw inputs."""
    repo_root = resolve_repo_root(config_path)
    daily_dir = resolve_path(repo_root, config["paths"]["daily_dir"])
    minute_dir = resolve_path(repo_root, config["paths"]["minute_dir"])
    processed_dir = ensure_dir(resolve_path(repo_root, config["paths"]["processed_dir"]))

    cache_enabled = config["cache"]["enabled"]
    cache_rebuild = config["cache"]["rebuild"]
    logging_config = get_logging_config(config)
    cache_panel_path = processed_dir / "model_features_with_splits_v1.parquet"
    cache_targets_path = processed_dir / "daily_panel_with_targets_v1.parquet"

    if cache_rebuild:
        logger.info("Cache rebuild requested; ignoring existing cache")

    if cache_enabled and cache_panel_path.exists() and cache_targets_path.exists() and not cache_rebuild:
        logger.info("Cache hit: %s", cache_panel_path)
        panel_df = pd.read_parquet(cache_panel_path).sort_values(["pred_date", "symbol"]).reset_index(drop=True)
        targets_df = pd.read_parquet(cache_targets_path).sort_values(["pred_date", "symbol"]).reset_index(drop=True)
        if "trailing_magnitude_20d" not in panel_df.columns:
            panel_df = add_trailing_magnitude(panel_df)
            panel_df = panel_df.sort_values(["pred_date", "symbol"]).reset_index(drop=True)
        logger.info(
            "Cached rows: %s | cached target rows: %s",
            format_count(len(panel_df)),
            format_count(len(targets_df)),
        )
        logger.info("Cached panel validation passed")
        return panel_df, {"repo_root": repo_root, "processed_dir": processed_dir, "targets_df": targets_df}
    if not cache_enabled:
        logger.info("Cache disabled; rebuilding feature panel from raw data")

    with log_stage(logger, "Loading daily data"):
        daily_df = load_all_daily_data(daily_dir)
        validate_daily_data(daily_df, config["data"]["expected_symbol_count"])

    with log_stage(logger, "Validating minute universe"):
        minute_files = list_minute_files(minute_dir)
        validate_minute_universe(minute_files, set(daily_df["symbol"].unique()), config["data"]["expected_symbol_count"])

    with log_stage(logger, "Constructing targets"):
        targets_df = construct_targets(daily_df)

    with log_stage(logger, "Building feature panel"):
        feature_df = build_feature_panel(targets_df, minute_files, logging_config=logging_config)
        feature_df = add_trailing_magnitude(feature_df)
        panel_df, first_embargo_dates, second_embargo_dates = apply_chronological_splits(
            feature_df,
            config["splits"]["validation_boundary"],
            config["splits"]["test_boundary"],
            config["splits"]["embargo_sessions"],
        )

    expected_split_counts = {"train": 186555, "valid": 54009, "test": 58656, "embargo": 2055}
    observed_counts = panel_df["split"].value_counts().to_dict()
    if observed_counts != expected_split_counts:
        raise AssertionError(f"Unexpected split counts: {observed_counts}")
    if len(first_embargo_dates) != 5 or len(second_embargo_dates) != 5:
        raise AssertionError("Embargo date counts do not match expectation")

    if cache_enabled:
        targets_df.to_parquet(cache_targets_path, index=False)
        panel_df.to_parquet(cache_panel_path, index=False)
        logger.info(
            "Saved processed panel: %s | Rows: %s | File size: %s",
            cache_panel_path,
            format_count(len(panel_df)),
            format_size(cache_panel_path.stat().st_size),
        )

    return panel_df, {"repo_root": repo_root, "processed_dir": processed_dir, "targets_df": targets_df}


def run_validation_checks(model_df: pd.DataFrame, config: dict, logger) -> dict:
    """Run notebook-style validation checks prior to the final refit."""
    benchmark_config = get_benchmark_config(config)
    train_df, valid_df, _ = prepare_direction_frames(model_df)
    expected_direction_counts = {"train": 182495, "valid": 53909}
    observed_direction_counts = {
        "train": len(train_df),
        "valid": len(valid_df),
    }
    if observed_direction_counts != expected_direction_counts:
        raise AssertionError(f"Unexpected direction warm-up counts: {observed_direction_counts}")

    logger.info("Running validation benchmark reproduction")
    logger.info(
        "Validation direction warm-up rows | train=%s | valid=%s",
        format_count(len(train_df)),
        format_count(len(valid_df)),
    )
    direction_train_x, direction_valid_x = align_symbol_categories(
        train_df[DIRECTION_FEATURE_COLUMNS],
        valid_df[DIRECTION_FEATURE_COLUMNS],
    )
    logger.info(
        "Training validation direction model | rows=%s | features=%s | categorical_features=1 | positive_target_fraction=%.4f | threshold=%.2f",
        format_count(len(train_df)),
        format_count(len(DIRECTION_FEATURE_COLUMNS)),
        float(train_df["actual_direction"].eq(1).mean()),
        config["direction"]["threshold"],
    )
    if get_logging_config(config).get("log_model_parameters", True):
        logger.info("Validation direction parameters | %s", json.dumps(config["direction"]["hyperparameters"], sort_keys=True))
    direction_model = make_direction_model(config)
    direction_model.fit(
        direction_train_x,
        train_df["actual_direction"].eq(1).astype("int8"),
        categorical_feature=["symbol"],
    )
    valid_probability_up, valid_pred_direction = score_direction_model(
        direction_model,
        direction_valid_x,
        config["direction"]["threshold"],
    )
    direction_score = float(
        np.sum(valid_pred_direction * valid_df["actual_return_pct"]) / np.sum(np.abs(valid_df["actual_return_pct"]))
    )
    logger.info(
        "Validation direction prediction summary | probability_mean=%.6f | probability_std=%.6f | probability_min=%.6f | probability_max=%.6f | predicted_up_fraction=%.6f | hit_rate=%.6f",
        float(valid_probability_up.mean()),
        float(valid_probability_up.std(ddof=1)),
        float(valid_probability_up.min()),
        float(valid_probability_up.max()),
        float(np.mean(valid_pred_direction == 1)),
        float(np.mean(valid_pred_direction == valid_df["actual_direction"].to_numpy())),
    )

    notebook_magnitude_df = model_df.sort_values(["symbol", "pred_date"]).reset_index(drop=True).copy()
    magnitude_train_df = notebook_magnitude_df.loc[notebook_magnitude_df["split"] == "train"].copy()
    magnitude_valid_df = notebook_magnitude_df.loc[notebook_magnitude_df["split"] == "valid"].copy()
    magnitude_features = [
        column
        for column in build_magnitude_feature_list(notebook_magnitude_df)
        if column not in NOTEBOOK_MAGNITUDE_HELPER_COLUMNS
    ]
    logger.info(
        "Validation magnitude benchmark rows | train=%s | valid=%s | features=%s",
        format_count(len(magnitude_train_df)),
        format_count(len(magnitude_valid_df)),
        format_count(len(magnitude_features)),
    )
    magnitude_train_x = magnitude_train_df[magnitude_features].copy()
    magnitude_valid_x = magnitude_valid_df[magnitude_features].copy()
    for frame in (magnitude_train_x, magnitude_valid_x):
        frame["symbol"] = frame["symbol"].astype("category")
    magnitude_model = lgb.LGBMRegressor(**config["magnitude"]["hyperparameters"])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=LGBMDeprecationWarning)
        magnitude_model.fit(
            magnitude_train_x,
            np.log1p(magnitude_train_df["actual_magnitude_pct"]),
            categorical_feature=["symbol"],
            eval_set=[(magnitude_valid_x, np.log1p(magnitude_valid_df["actual_magnitude_pct"]))],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )
    best_iteration = int(magnitude_model.best_iteration_)
    valid_pred_magnitude_pct = np.clip(
        np.expm1(magnitude_model.predict(magnitude_valid_x, num_iteration=best_iteration)),
        0,
        None,
    )
    valid_mask = magnitude_valid_df["trailing_magnitude_20d"].notna().to_numpy()
    actual_mag = magnitude_valid_df.loc[valid_mask, "actual_magnitude_pct"].to_numpy(dtype=float)
    pred_mag = valid_pred_magnitude_pct[valid_mask]
    magnitude_score = float(1 - np.sum(np.abs(pred_mag - actual_mag)) / np.sum(actual_mag))
    rank_frame = magnitude_valid_df.loc[valid_mask, ["pred_date"]].copy()
    rank_frame["actual"] = actual_mag
    rank_frame["prediction"] = pred_mag
    rank_ics = []
    for _, day in rank_frame.groupby("pred_date"):
        if len(day) >= 3 and day["actual"].nunique() > 1 and day["prediction"].nunique() > 1:
            corr = spearmanr(day["prediction"], day["actual"]).statistic
            if np.isfinite(corr):
                rank_ics.append(corr)
    rank_ic = float(np.mean(rank_ics))

    confidence_panel = model_df.sort_values(["pred_date", "symbol"]).reset_index(drop=True).copy()
    confidence_train_df = confidence_panel.loc[confidence_panel["split"] == "train"].copy()
    confidence_valid_df = confidence_panel.loc[confidence_panel["split"] == "valid"].copy()
    logger.info(
        "Validation confidence benchmark rows | train=%s | valid=%s",
        format_count(len(confidence_train_df)),
        format_count(len(confidence_valid_df)),
    )
    confidence_train_x = confidence_train_df[DIRECTION_FEATURE_COLUMNS].copy()
    confidence_valid_x = confidence_valid_df[DIRECTION_FEATURE_COLUMNS].copy()
    for frame in (confidence_train_x, confidence_valid_x):
        frame["symbol"] = frame["symbol"].astype("category")
    confidence_direction_model = make_direction_model(config)
    confidence_direction_model.fit(
        confidence_train_x,
        confidence_train_df["actual_direction"].eq(1).astype("int8"),
        categorical_feature=["symbol"],
    )
    confidence_probability_up, confidence_pred_direction = score_direction_model(
        confidence_direction_model,
        confidence_valid_x,
        config["direction"]["threshold"],
    )
    confidence_valid_df["direction_correct"] = (
        confidence_pred_direction == confidence_valid_df["actual_direction"].to_numpy()
    ).astype("int8")
    direction_conf_train, direction_conf_features = create_direction_confidence_features(
        confidence_valid_df,
        confidence_probability_up,
        confidence_pred_direction,
    )
    valid_dates = np.array(sorted(confidence_valid_df["pred_date"].drop_duplicates()))
    cutoff = int(len(valid_dates) * config["direction_confidence"]["experimental_validation_fraction"])
    cal_dates = set(valid_dates[:cutoff])
    sel_dates = set(valid_dates[cutoff:])
    confidence_cal_mask = confidence_valid_df["pred_date"].isin(cal_dates).to_numpy()
    confidence_sel_mask = confidence_valid_df["pred_date"].isin(sel_dates).to_numpy()
    x_cal = direction_conf_train.loc[confidence_cal_mask, direction_conf_features]
    x_sel = direction_conf_train.loc[confidence_sel_mask, direction_conf_features]
    y_cal = confidence_valid_df.loc[confidence_cal_mask, "direction_correct"]
    fill_values = x_cal.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    x_cal = x_cal.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    x_sel = x_sel.replace([np.inf, -np.inf], np.nan).fillna(fill_values)
    direction_conf_model = __import__("src.models", fromlist=["make_direction_confidence_model"]).make_direction_confidence_model(config)
    direction_conf_model.fit(x_cal, y_cal)
    conf_direction = np.clip(
        direction_conf_model.predict_proba(x_sel)[:, 1],
        config["direction_confidence"]["clip_min"],
        config["direction_confidence"]["clip_max"],
    )
    sel_df = confidence_valid_df.loc[confidence_sel_mask].copy()
    w = 2 * conf_direction - 1
    conf_direction_score = float(
        np.sum(w * confidence_pred_direction[confidence_sel_mask] * sel_df["actual_return_pct"])
        / np.sum(w * np.abs(sel_df["actual_return_pct"]))
    )

    shadow = run_shadow_workflow(model_df, config)
    valid_shadow_df = shadow["valid_df"].copy().reset_index(drop=True)
    valid_shadow_df["direction_correct"] = (
        shadow["shadow_pred_direction"] == valid_shadow_df["actual_direction"].to_numpy()
    ).astype("int8")
    magnitude_conf_train, magnitude_conf_features = __import__("src.models", fromlist=["create_magnitude_confidence_features"]).create_magnitude_confidence_features(
        valid_shadow_df,
        shadow["shadow_pred_magnitude_pct"],
    )
    magnitude_conf_train["absolute_magnitude_error"] = np.abs(
        shadow["shadow_pred_magnitude_pct"] - valid_shadow_df["actual_magnitude_pct"].to_numpy()
    )
    magnitude_conf_train["log_absolute_magnitude_error"] = np.log1p(magnitude_conf_train["absolute_magnitude_error"])
    shadow_dates = np.array(sorted(valid_shadow_df["pred_date"].drop_duplicates()))
    shadow_cutoff = int(len(shadow_dates) * config["direction_confidence"]["experimental_validation_fraction"])
    shadow_cal_dates = set(shadow_dates[:shadow_cutoff])
    shadow_sel_dates = set(shadow_dates[shadow_cutoff:])
    shadow_cal_mask = valid_shadow_df["pred_date"].isin(shadow_cal_dates).to_numpy()
    shadow_sel_mask = valid_shadow_df["pred_date"].isin(shadow_sel_dates).to_numpy()
    mag_x_cal = magnitude_conf_train.loc[shadow_cal_mask, magnitude_conf_features]
    mag_x_sel = magnitude_conf_train.loc[shadow_sel_mask, magnitude_conf_features]
    mag_y_cal = magnitude_conf_train.loc[shadow_cal_mask, "log_absolute_magnitude_error"]
    mag_fill = mag_x_cal.replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
    mag_x_cal = mag_x_cal.replace([np.inf, -np.inf], np.nan).fillna(mag_fill)
    mag_x_sel = mag_x_sel.replace([np.inf, -np.inf], np.nan).fillna(mag_fill)
    mag_conf_model = __import__("src.models", fromlist=["make_magnitude_confidence_model"]).make_magnitude_confidence_model(config)
    mag_conf_model.fit(mag_x_cal, mag_y_cal)
    train_expected_error = np.clip(np.expm1(mag_conf_model.predict(mag_x_cal)), 0, None)
    selection_expected_error = np.clip(np.expm1(mag_conf_model.predict(mag_x_sel)), 0, None)
    selection_conf_magnitude = __import__("src.models", fromlist=["expected_error_to_confidence"]).expected_error_to_confidence(
        selection_expected_error, np.sort(train_expected_error)
    )
    conf_magnitude_spearman = float(
        spearmanr(selection_conf_magnitude, -magnitude_conf_train.loc[shadow_sel_mask, "absolute_magnitude_error"]).statistic
    )

    references = {
        "direction_score": direction_score,
        "magnitude_score": magnitude_score,
        "rank_ic": rank_ic,
        "conf_direction_score": conf_direction_score,
        "conf_magnitude_spearman": conf_magnitude_spearman,
    }
    fail_on_drift = bool(get_logging_config(config).get("fail_on_benchmark_drift", False))
    for name, observed in references.items():
        benchmark = benchmark_config[name]
        log_benchmark_result(
            logger,
            name,
            observed,
            benchmark["reference"],
            benchmark["tolerance"],
            fail_on_drift,
        )
    logger.info(
        "Magnitude best iteration | observed=%s | reference=%s | delta=%s | status=%s",
        format_count(best_iteration),
        format_count(config["magnitude"]["final_best_iteration"]),
        format_count(best_iteration - config["magnitude"]["final_best_iteration"]),
        "PASS" if best_iteration == config["magnitude"]["final_best_iteration"] else "WARN",
    )

    return {
        "direction_score": direction_score,
        "magnitude_score": magnitude_score,
        "rank_ic": rank_ic,
        "conf_direction_score": conf_direction_score,
        "conf_magnitude_spearman": conf_magnitude_spearman,
        "magnitude_best_iteration": best_iteration,
    }


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    project_root = resolve_repo_root(config_path)
    logger = setup_logging(config, project_root)
    set_global_seed(config["seed"])
    versions = package_versions()

    logger.info("=" * 60)
    logger.info("ASTRA OVERNIGHT RETURN PIPELINE")
    logger.info("=" * 60)
    logger.info("Project root: %s", project_root)
    logger.info("Config: %s", config_path)
    logger.info("Python version: %s", versions["python"])
    logger.info("Seed: %s", config["seed"])
    logger.info("Cache enabled: %s", config["cache"]["enabled"])
    logger.info("Cache rebuild: %s", config["cache"]["rebuild"])
    logger.info("Output directory: %s", resolve_path(project_root, config["paths"]["final_output_dir"]))
    logger.info("Log file: %s", getattr(logger, "log_file_path", None))
    logger.info("=" * 60)
    logger.debug("Package versions: %s", json.dumps(versions, sort_keys=True))

    try:
        with log_stage(logger, "Pipeline"):
            model_df, context = load_or_build_panel(config, config_path, logger)
            assert_no_feature_leakage(DIRECTION_FEATURE_COLUMNS)
            if len(DIRECTION_FEATURE_COLUMNS) != 26:
                raise AssertionError("Final direction feature count must be 26 including symbol")
            if model_df["symbol"].nunique() != config["data"]["expected_symbol_count"]:
                raise AssertionError("Unexpected symbol count in model panel")
            assert_no_infinite_values(model_df, [column for column in model_df.columns if pd.api.types.is_numeric_dtype(model_df[column])], "model panel")

            with log_stage(logger, "Validation benchmark reproduction"):
                validation_summary = run_validation_checks(model_df, config, logger)

            with log_stage(logger, "Shadow-model workflow"):
                shadow_outputs = run_shadow_workflow(model_df, config)

            with log_stage(logger, "Final primary-model fitting and scoring"):
                primary_outputs = fit_final_primary_models(model_df, config, shadow_outputs)

            with log_stage(logger, "Final confidence-model fitting and scoring"):
                confidence_outputs = fit_final_confidence_models(
                    shadow_outputs,
                    primary_outputs["scored_df"],
                    config,
                )
            scored_df = confidence_outputs["scored_df"].copy()
            scored_df = scored_df.sort_values(["pred_date", "symbol"]).reset_index(drop=True)

            for split_name in ["train", "valid", "test"]:
                split_rows = int((scored_df["split"] == split_name).sum())
                logger.info("Scoring %s | rows=%s", split_name, format_count(split_rows))

            logger.info(
                "Prediction summary | rows=%s | predicted_up_fraction=%.6f | mean_pred_magnitude_pct=%.6f | mean_conf_direction=%.6f | mean_conf_magnitude=%.6f | conf_direction_min=%.6f | conf_direction_max=%.6f | conf_magnitude_min=%.6f | conf_magnitude_max=%.6f",
                format_count(len(scored_df)),
                float(np.mean(scored_df["pred_direction"] == 1)),
                float(scored_df["pred_magnitude_pct"].mean()),
                float(scored_df["conf_direction"].mean()),
                float(scored_df["conf_magnitude"].mean()),
                float(scored_df["conf_direction"].min()),
                float(scored_df["conf_direction"].max()),
                float(scored_df["conf_magnitude"].min()),
                float(scored_df["conf_magnitude"].max()),
            )

            with log_stage(logger, "Output dataframe construction"):
                predictions_df = build_predictions_df(scored_df, config["output"]["decimal_places_magnitude"])
                actuals_df = build_actuals_df(scored_df)
                statistics_df = build_statistics_df(scored_df)
                validate_output_frames(predictions_df, actuals_df, statistics_df)

            output_dir = ensure_dir(resolve_path(context["repo_root"], config["paths"]["final_output_dir"]))
            with log_stage(logger, "Writing output files"):
                write_outputs(predictions_df, actuals_df, statistics_df, output_dir, config["output"])

            logger.info(
                "Output rows | predictions=%s actuals=%s statistics=%s",
                format_count(len(predictions_df)),
                format_count(len(actuals_df)),
                format_count(len(statistics_df)),
            )
            logger.info("Split rows | %s", scored_df["split"].value_counts().sort_index().to_dict())
            logger.info("Validation summary | %s", json.dumps(validation_summary, indent=2))

            total_runtime = time.perf_counter() - start_time
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info("Total runtime: %s", format_duration(total_runtime))
            logger.info("Daily symbols: %s", format_count(config["data"]["expected_symbol_count"]))
            logger.info("Minute symbols: %s", format_count(config["data"]["expected_symbol_count"]))
            logger.info("Scored rows: %s", format_count(len(scored_df)))
            logger.info("Train rows: %s", format_count(int((scored_df['split'] == 'train').sum())))
            logger.info("Validation rows: %s", format_count(int((scored_df['split'] == 'valid').sum())))
            logger.info("Test rows: %s", format_count(int((scored_df['split'] == 'test').sum())))
            logger.info("Statistics rows: %s", format_count(len(statistics_df)))
            logger.info("Validation:")
            logger.info("Direction score: %.6f", validation_summary["direction_score"])
            logger.info("Magnitude score: %.6f", validation_summary["magnitude_score"])
            logger.info("Rank IC: %.6f", validation_summary["rank_ic"])
            logger.info("Direction-confidence score: %.6f", validation_summary["conf_direction_score"])
            logger.info("Magnitude-confidence Spearman: %.6f", validation_summary["conf_magnitude_spearman"])
            logger.info("Magnitude best iteration: %s", format_count(validation_summary["magnitude_best_iteration"]))
            logger.info("Outputs:")
            logger.info("%s", output_dir / config["output"]["predictions_filename"])
            logger.info("%s", output_dir / config["output"]["actuals_filename"])
            logger.info("%s", output_dir / config["output"]["statistics_filename"])
            logger.info("Log:")
            logger.info("%s", getattr(logger, "log_file_path", None))
            logger.info("=" * 60)
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        logger.error("=" * 60)
        logger.error("PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error("Exception type: %s", type(exc).__name__)
        logger.error("Exception message: %s", exc)
        logger.error("Elapsed time: %s", format_duration(elapsed))
        logger.error("Log-file path: %s", getattr(logger, "log_file_path", None))
        logger.error("=" * 60)
        logger.exception("Stack trace follows")
        raise


if __name__ == "__main__":
    main()
