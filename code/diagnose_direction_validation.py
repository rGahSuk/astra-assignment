"""Forensic comparison for the V1 validation-stage direction model."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

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
from src.features import DIRECTION_FEATURE_COLUMNS, DIRECTION_NUMERIC_FEATURE_COLUMNS, add_trailing_magnitude, build_feature_panel
from src.models import align_symbol_categories, make_direction_model, prepare_direction_frames, score_direction_model
from src.utils import load_yaml, set_global_seed


def _direction_score(actual_return_pct: np.ndarray, pred_direction: np.ndarray) -> float:
    return float(np.sum(pred_direction * actual_return_pct) / np.sum(np.abs(actual_return_pct)))


def _summary_stats(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        series = frame[column]
        rows.append(
            {
                "feature": column,
                "mean": float(series.mean(skipna=True)),
                "std": float(series.std(skipna=True)),
                "min": float(series.min(skipna=True)),
                "max": float(series.max(skipna=True)),
            }
        )
    return pd.DataFrame(rows)


def _missing_stats(frame: pd.DataFrame, columns: list[str], split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": columns,
            f"{split_name}_missing": [int(frame[column].isna().sum()) for column in columns],
        }
    )


def _print_prediction_stats(label: str, probability_up: np.ndarray, pred_direction: np.ndarray, valid_df: pd.DataFrame) -> None:
    print(f"\n=== {label} Prediction Diagnostics ===")
    print(
        json.dumps(
            {
                "probability_mean": float(np.mean(probability_up)),
                "probability_std": float(np.std(probability_up, ddof=1)),
                "probability_min": float(np.min(probability_up)),
                "probability_max": float(np.max(probability_up)),
                "predicted_up_fraction": float(np.mean(pred_direction == 1)),
                "hit_rate": float(np.mean(pred_direction == valid_df["actual_direction"].to_numpy())),
                "direction_score": _direction_score(valid_df["actual_return_pct"].to_numpy(dtype=float), pred_direction),
            },
            indent=2,
        )
    )
    print("confusion_matrix labels=[-1, 1]")
    print(confusion_matrix(valid_df["actual_direction"], pred_direction, labels=[-1, 1]))


def _fit_validation_direction_model(
    model_df: pd.DataFrame,
    config: dict,
    feature_columns: list[str],
) -> dict:
    train_df, valid_df, _ = prepare_direction_frames(model_df)
    x_train, x_valid = align_symbol_categories(train_df[feature_columns], valid_df[feature_columns])
    y_train = train_df["actual_direction"].astype("int8").reset_index(drop=True)
    y_valid = valid_df["actual_direction"].astype("int8").reset_index(drop=True)
    y_train_binary = (y_train == 1).astype("int8")
    y_valid_binary = (y_valid == 1).astype("int8")

    model = make_direction_model(config)
    fit_kwargs = {"categorical_feature": ["symbol"]}
    model.fit(x_train, y_train_binary, **fit_kwargs)
    probability_up, pred_direction = score_direction_model(model, x_valid, config["direction"]["threshold"])

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "x_train": x_train,
        "x_valid": x_valid,
        "y_train": y_train,
        "y_valid": y_valid,
        "y_train_binary": y_train_binary,
        "y_valid_binary": y_valid_binary,
        "model": model,
        "fit_kwargs": fit_kwargs,
        "probability_up": probability_up,
        "pred_direction": pred_direction,
    }


def _load_rebuilt_panel(config_path: Path, config: dict) -> pd.DataFrame:
    repo_root = resolve_repo_root(config_path)
    daily_dir = resolve_path(repo_root, config["paths"]["daily_dir"])
    minute_dir = resolve_path(repo_root, config["paths"]["minute_dir"])
    daily_df = load_all_daily_data(daily_dir)
    validate_daily_data(daily_df, config["data"]["expected_symbol_count"])
    minute_files = list_minute_files(minute_dir)
    validate_minute_universe(minute_files, set(daily_df["symbol"].unique()), config["data"]["expected_symbol_count"])
    targets_df = construct_targets(daily_df)
    panel_df = build_feature_panel(targets_df, minute_files)
    panel_df = add_trailing_magnitude(panel_df)
    panel_df, _, _ = apply_chronological_splits(
        panel_df,
        config["splits"]["validation_boundary"],
        config["splits"]["test_boundary"],
        config["splits"]["embargo_sessions"],
    )
    return panel_df.sort_values(["pred_date", "symbol"]).reset_index(drop=True)


def main() -> None:
    config_path = Path("code/config.yaml").resolve()
    config = load_yaml(config_path)
    set_global_seed(config["seed"])
    repo_root = resolve_repo_root(config_path)

    cached_panel_path = repo_root / "outputs" / "processed_data" / "model_features_with_splits_v1.parquet"
    feature_json_path = repo_root / "outputs" / "processed_data" / "final_direction_feature_columns.json"
    notebook_panel = pd.read_parquet(cached_panel_path).sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    with feature_json_path.open("r", encoding="utf-8") as handle:
        notebook_feature_list = json.load(handle)

    rebuilt_panel = _load_rebuilt_panel(config_path, config)

    print("=== Row Count Check ===")
    notebook_train, notebook_valid, _ = prepare_direction_frames(notebook_panel)
    rebuilt_train, rebuilt_valid, _ = prepare_direction_frames(rebuilt_panel)
    print(
        json.dumps(
            {
                "expected": {"train": 182495, "valid": 53909},
                "notebook_like": {"train": len(notebook_train), "valid": len(notebook_valid)},
                "production_rebuilt": {"train": len(rebuilt_train), "valid": len(rebuilt_valid)},
            },
            indent=2,
        )
    )

    print("\n=== Feature List And Order ===")
    print("notebook_feature_count:", len(notebook_feature_list))
    print("production_feature_count:", len(DIRECTION_FEATURE_COLUMNS))
    print("exact_same_order:", notebook_feature_list == DIRECTION_FEATURE_COLUMNS)
    print(json.dumps(DIRECTION_FEATURE_COLUMNS, indent=2))

    print("\n=== Extra / Suspicious Columns In Direction Matrix Source ===")
    excluded_direction_inputs = set(notebook_panel.columns) - set(DIRECTION_FEATURE_COLUMNS)
    suspicious_columns = sorted(
        [
            column
            for column in excluded_direction_inputs
            if any(term in column.lower() for term in ["actual_", "target_", "next_open", "trailing_magnitude", "v2", "v3", "future_", "forward_", "label"])
        ]
    )
    print("suspicious_non_feature_columns_present_in_panel:", suspicious_columns)
    print("direction_matrix_columns_exact:", DIRECTION_FEATURE_COLUMNS)

    print("\n=== Missing Value Counts ===")
    notebook_missing = _missing_stats(notebook_train, DIRECTION_NUMERIC_FEATURE_COLUMNS, "train").merge(
        _missing_stats(notebook_valid, DIRECTION_NUMERIC_FEATURE_COLUMNS, "valid"),
        on="feature",
    )
    rebuilt_missing = _missing_stats(rebuilt_train, DIRECTION_NUMERIC_FEATURE_COLUMNS, "train").merge(
        _missing_stats(rebuilt_valid, DIRECTION_NUMERIC_FEATURE_COLUMNS, "valid"),
        on="feature",
    )
    print("Notebook-like:")
    print(notebook_missing.to_string(index=False))
    print("\nProduction rebuilt:")
    print(rebuilt_missing.to_string(index=False))

    print("\n=== Validation Summary Stats ===")
    print("Notebook-like:")
    print(_summary_stats(notebook_valid, DIRECTION_NUMERIC_FEATURE_COLUMNS).to_string(index=False))
    print("\nProduction rebuilt:")
    print(_summary_stats(rebuilt_valid, DIRECTION_NUMERIC_FEATURE_COLUMNS).to_string(index=False))

    notebook_artifacts = _fit_validation_direction_model(notebook_panel, config, notebook_feature_list)
    rebuilt_artifacts = _fit_validation_direction_model(rebuilt_panel, config, DIRECTION_FEATURE_COLUMNS)

    print("\n=== Symbol Categories ===")
    print("notebook_train_categories_count:", len(notebook_artifacts["x_train"]["symbol"].cat.categories))
    print("production_train_categories_count:", len(rebuilt_artifacts["x_train"]["symbol"].cat.categories))
    print(
        "category_order_exact_match:",
        list(notebook_artifacts["x_train"]["symbol"].cat.categories)
        == list(rebuilt_artifacts["x_train"]["symbol"].cat.categories),
    )
    print("first_20_categories:", list(rebuilt_artifacts["x_train"]["symbol"].cat.categories[:20]))
    print(
        "validation_categories_same_as_train:",
        list(rebuilt_artifacts["x_train"]["symbol"].cat.categories)
        == list(rebuilt_artifacts["x_valid"]["symbol"].cat.categories),
    )

    print("\n=== LightGBM Parameters ===")
    print(json.dumps(rebuilt_artifacts["model"].get_params(), indent=2, default=str))
    print("categorical_feature_fit_arg:", rebuilt_artifacts["fit_kwargs"]["categorical_feature"])

    print("\n=== Target Mapping ===")
    print(
        json.dumps(
            {
                "train_actual_direction_values": sorted(rebuilt_artifacts["y_train"].unique().tolist()),
                "train_binary_values": sorted(rebuilt_artifacts["y_train_binary"].unique().tolist()),
                "mapping_check": {
                    "+1_to": int(rebuilt_artifacts["y_train_binary"][rebuilt_artifacts["y_train"] == 1].iloc[0]),
                    "-1_to": int(rebuilt_artifacts["y_train_binary"][rebuilt_artifacts["y_train"] == -1].iloc[0]),
                },
            },
            indent=2,
        )
    )

    print("\n=== Threshold And Formula ===")
    print("threshold_rule: probability_up >= 0.64 -> +1 else -1")
    print("direction_score_formula: sum(pred_direction * actual_return_pct) / sum(abs(actual_return_pct))")

    _print_prediction_stats(
        "Notebook-like Cached Panel",
        notebook_artifacts["probability_up"],
        notebook_artifacts["pred_direction"],
        notebook_artifacts["valid_df"],
    )
    _print_prediction_stats(
        "Production Rebuilt Panel",
        rebuilt_artifacts["probability_up"],
        rebuilt_artifacts["pred_direction"],
        rebuilt_artifacts["valid_df"],
    )

    print("\n=== Probability / Prediction Comparison Against Cached Notebook-Equivalent Run ===")
    common_keys = ["pred_date", "symbol"]
    notebook_preds = notebook_artifacts["valid_df"][common_keys].copy()
    notebook_preds["notebook_probability_up"] = notebook_artifacts["probability_up"]
    notebook_preds["notebook_pred_direction"] = notebook_artifacts["pred_direction"]
    rebuilt_preds = rebuilt_artifacts["valid_df"][common_keys].copy()
    rebuilt_preds["production_probability_up"] = rebuilt_artifacts["probability_up"]
    rebuilt_preds["production_pred_direction"] = rebuilt_artifacts["pred_direction"]
    pred_compare = notebook_preds.merge(rebuilt_preds, on=common_keys, how="inner", validate="one_to_one")
    pred_compare["probability_diff"] = pred_compare["production_probability_up"] - pred_compare["notebook_probability_up"]
    pred_compare["prediction_match"] = pred_compare["production_pred_direction"] == pred_compare["notebook_pred_direction"]
    print(
        json.dumps(
            {
                "rows_compared": len(pred_compare),
                "prediction_match_rate": float(pred_compare["prediction_match"].mean()),
                "probability_abs_diff_mean": float(pred_compare["probability_diff"].abs().mean()),
                "probability_abs_diff_max": float(pred_compare["probability_diff"].abs().max()),
            },
            indent=2,
        )
    )

    print("\n=== First Feature-Level Mismatch Search ===")
    compare_columns = ["pred_date", "symbol"] + DIRECTION_NUMERIC_FEATURE_COLUMNS + ["split", "actual_direction", "actual_return_pct"]
    merged = notebook_panel[compare_columns].merge(
        rebuilt_panel[compare_columns],
        on=["pred_date", "symbol"],
        how="inner",
        suffixes=("_notebook", "_production"),
        validate="one_to_one",
    ).sort_values(["pred_date", "symbol"]).reset_index(drop=True)
    first_mismatch = None
    for feature in DIRECTION_NUMERIC_FEATURE_COLUMNS:
        left = merged[f"{feature}_notebook"]
        right = merged[f"{feature}_production"]
        mismatch_mask = ~(
            (left.isna() & right.isna())
            | np.isclose(left.to_numpy(dtype=float), right.to_numpy(dtype=float), equal_nan=True, rtol=0.0, atol=1e-12)
        )
        if mismatch_mask.any():
            row = merged.loc[mismatch_mask].iloc[0]
            first_mismatch = {
                "feature": feature,
                "pred_date": str(pd.Timestamp(row["pred_date"]).date()),
                "symbol": row["symbol"],
                "notebook_value": row[f"{feature}_notebook"],
                "production_value": row[f"{feature}_production"],
            }
            break
    print(json.dumps(first_mismatch, indent=2, default=str))

    if first_mismatch:
        feature = first_mismatch["feature"]
        sample = merged.loc[
            (merged["symbol"] == first_mismatch["symbol"])
            & (merged["pred_date"] == pd.Timestamp(first_mismatch["pred_date"]))
        ][
            [
                "pred_date",
                "symbol",
                f"{feature}_notebook",
                f"{feature}_production",
                "split_notebook",
                "split_production",
                "actual_direction_notebook",
                "actual_direction_production",
                "actual_return_pct_notebook",
                "actual_return_pct_production",
            ]
        ]
        print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
