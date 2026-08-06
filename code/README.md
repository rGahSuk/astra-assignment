# Astra Assignment Production Pipeline

This folder contains a production implementation of the notebook research for the Astra Invest Quant Researcher Intern assignment. The pipeline rebuilds the modelling panel from raw Parquet inputs, preserves the frozen notebook decisions, and writes `final_outputs/predictions.csv`, `final_outputs/actuals.csv`, and `final_outputs/statistics.csv`.

Python 3.10+ is required.

Install:

```bash
pip install -r code/requirements.txt
```

Run:

```bash
python code/main.py --config code/config.yaml
```

Expected raw data placement:

- `data/daily/*.parquet`
- `data/minute/*.parquet`

Repository structure:

- `code/main.py`: orchestration only
- `code/src/data.py`: loading, validation, targets, splits, cache paths
- `code/src/features.py`: V1 daily, minute, and cross-sectional features
- `code/src/models.py`: primary models, shadow workflows, confidence models
- `code/src/evaluate.py`: output frames, metrics, validations
- `code/src/utils.py`: config, logging, seeding, timing helpers

Model summary:

- V1 direction features were retained.
- V2 and V3 direction features did not improve validation performance.
- The final direction model is a pooled LightGBM classifier with `symbol` categorical and a fixed `0.64` threshold.
- The final magnitude model is the original 31-leaf LightGBM L1 regressor with final `n_estimators=84`.
- Huber and larger magnitude models were rejected.
- The final direction-confidence model is logistic regression on shadow-validation correctness labels.
- Shallow LightGBM direction confidence was rejected.
- The final magnitude-confidence model is the 15-leaf LightGBM expected-error model.
- Larger magnitude-confidence LightGBM was rejected.
- Test was not used for model selection.

Validation benchmark references:

- The direction benchmark reference is the reconciled V1 notebook-equivalent score: `0.29481723269301396`.
- The magnitude benchmark reference remains the reproducible Notebook 04 path: `magnitude_score=0.254340`, `rank_ic=0.245931`, `best_iteration=84`.
- Notebook 04 used the 38-feature magnitude panel, while the final production panel exposes 8 additional rolling-beta helper columns; those extra numeric columns explain the production feature-count log of 46.
- The direction-confidence benchmark reference remains the reproducible Notebook 05 path: `conf_direction_score=0.268893`.
- A higher `conf_direction_score` around `0.302190` appears only when the benchmark is run on the direction warm-up filtered validation sample (`53,909` rows instead of `54,009`), not because of a changed confidence model.

Split policy:

- Train: `2020-06-01` through `2024-03-28`
- Embargo: 5 available sessions after each boundary
- Valid: `2024-04-08` through `2025-04-30`
- Test: `2025-05-09` through `2026-06-29`

Cache behavior:

- Cached processed panels live under `outputs/processed_data`.
- Cache is optional and controlled by `code/config.yaml`.
- Cached files are an optimization only; the pipeline can rebuild from raw data.

Expected runtime and memory:

- Runtime is dominated by minute-file aggregation and LightGBM fitting.
- The full run should complete in minutes on a normal laptop.
- Memory use should stay moderate because minute files are processed symbol by symbol.

Known limitations:

- The implementation intentionally follows the frozen notebook research rather than adding robustness features beyond the assignment scope.
- LightGBM may produce tiny floating-point differences across environments.

Time-budget prioritization:

- The work prioritized faithful reproduction of Notebook 02 feature logic, Notebook 07 sequencing, output schema validation, and regression checks before optional engineering polish.
