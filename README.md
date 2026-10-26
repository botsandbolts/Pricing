# pricing · Car Price Studio

Car Price Studio is a used-car asking-price GUI with custom make/model fields, vehicle dropdowns, mileage/year inputs, a dollar estimate, an indicative error band and measured test results.

**Current choices:** no make/model transformer; a fixed **10% raw database sample**; E5 description-price regression plus **Isolation Forest in auto mode** for the enhanced workflow; Huber LightGBM as the final regressor.

## What works now

The deployed studio provides **real estimates from a trained tabular-only LightGBM baseline**. It uses make/model as categorical fields and the other vehicle characteristics. The browser runs the actual exported trees; no hand-written pricing formula or fabricated example price is used. The model is included under `dist/baseline-model.json`, and Python weights/schema are under `models/studio-baseline/`.

Measured baseline test results: **MAE $3,749, RMSE $7,023, median APE 17.0%, R² 0.765**. The validation-calibrated 80% band covered 78.6% of test listing prices. Selection used validation USD MAE across dollar/log targets and Huber thresholds 0.5/0.9/1.5. The winner used log price and Huber 0.5.

**E5-enhanced training is not complete.** The current workspace has no CUDA GPU. The UI labels that mode as pending and disables it; seller descriptions do not affect baseline estimates. Because the baseline has no `predicted_text_price`, it does not apply the E5-price Isolation Forest filter. Baseline scores must not be presented as results for the enhanced pipeline.

## Only 10% of the original database

A deterministic uniform-priority sample selects **42,688 of 426,880 raw listings** before eligibility filtering. Price/text/date checks and description/VIN deduplication leave **31,001** rows: 21,700 chronological training, 4,650 validation and 4,651 test rows. Seed 42. The requested 10% is the raw sample size, not a fraction of an earlier pilot.

Source: Craigslist US asking prices from April–May **2021**, restricted to $1,000–$150,000 and descriptions of at least 80 characters. These are not verified sale prices, live valuations or inflation-adjusted prices. Newer cars are outside the training period.

```bash
python -m scripts.prepare_stack /path/to/vehicles.zip --fraction .1 --output training-cache/ten-percent.jsonl
```

## Requested enhanced pipeline

1. Fine-tune `embaas/sentence-transformers-multilingual-e5-base` on **seller descriptions only**, predicting standardized log1p(price). The pinned XLM-RoBERTa checkpoint is `ff629add2ff80256e42173ec84299eed1c1b6393`, with a 768-dimensional representation and 512-token windows. Up to 8,000 cleaned characters use consecutive windows and pooled representations.
2. Three-fold cross-fitting creates **`predicted_text_price`** in dollars. Every training row is predicted by a model that did not see its target or use it in target scaling.
3. `IsolationForest(n_estimators=300, contamination="auto", random_state=42)` fits **only those training OOF dollar prices**. Keep inliers and remove flagged rows from LightGBM training. No removal percentage is forced. Actual prices, tabular features, validation and test do not enter this detector.
4. Huber LightGBM combines the text-price scalar with year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. **Make/model have no transformer, embedding or separate price regressor.**

Isolation Forest can flag valid rare vehicles. Its actual removal count and held-out effects must be measured after E5 training. A collapsed text-price signal can leave too few inliers; the run stops rather than silently forcing retention.

Assessment fits OOF models within the chronological training partition and uses train-only regressors for validation/test. After test scoring, a fresh OOF pass covers the entire eligible **10% sample**, the detector filters that fitting pool, and the deployable model is refitted. All-row refit means all sampled candidates, never the other 90% of the database. Assessment scores describe the assessment model, not an independent evaluation of the refit.

## Run Car Price Studio

The hosted static app uses `dist/` and includes the trained baseline. Python GUI:

```bash
python -m pip install -r requirements-training.txt
python -m streamlit run app.py
```

The Python GUI defaults to `models/studio-baseline/`. Its mode selector can load an enhanced bundle from `artifacts/car-price-studio/` after training. Override with `PRICING_MODEL_DIR` if needed. No API key is required. Baseline browser inputs are processed locally.

For E5 training, open `train_gpu.ipynb` in a CUDA runtime or run:

```bash
python -m pricing.train --data training-cache/ten-percent.jsonl --output artifacts/car-price-studio --device cuda --folds 3 --epochs 2
```

This is substantial transformer fine-tuning. It uses activation checkpointing, batch size 1 and gradient accumulation. Completed OOF folds resume with matching row IDs, targets, configuration, source revision and code fingerprints. Explicit `--device cpu --allow-cpu` permits slow CPU execution; it is not substituted silently.

To reproduce the baseline:

```bash
python -m pricing.baseline --data training-cache/ten-percent.jsonl --output artifacts/studio-baseline --public dist
```

## Checks

```bash
python -m pytest tests -q
node tests/check-browser.mjs
python -m compileall -q pricing scripts app.py
```

Tests cover target exclusion, cache validity, transformer gradients, the complete E5 + scalar-only Isolation Forest + LightGBM flow on synthetic fixtures, make/model-transformer removal, and feature/inference schemas. Synthetic scores are not pricing-quality evidence. The browser check verifies exported-tree prices against native LightGBM on 30 held-out rows (maximum difference below $0.000001). Browser QA covered estimates, custom unseen names, examples, results and training-status labeling.

The GUI error band is indicative, based on validation log residuals. It is not a guaranteed per-car sale range. Back-transformed log predictions are not bias-corrected arithmetic means.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-base
