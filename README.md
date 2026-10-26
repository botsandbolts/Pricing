# pricing · Car Price Studio

Car Price Studio is a used-car asking-price GUI with custom make/model fields, vehicle dropdowns, mileage/year inputs, a dollar estimate, an indicative error band and measured test results.

**Current choices:** no make/model transformer; the **full 100% raw database**; MiniLM description-price regression plus **Isolation Forest in auto mode** for the enhanced workflow; Huber LightGBM as the final regressor.

## What works now

The deployed studio provides **real estimates from a trained tabular-only LightGBM baseline**. It uses make/model as categorical fields and the other vehicle characteristics. The browser runs the actual exported trees; no hand-written pricing formula or fabricated example price is used. The final deployment model, fitted on all 211,844 eligible rows, is included under `dist/baseline-model.json`, and Python weights/schema are under `models/studio-baseline/`; the native LightGBM model is gzip-compressed and loaded automatically.

Measured baseline test results: **MAE $3,329, RMSE $6,839, median APE 15.6%, R² 0.774**. The validation-calibrated 80% band covered 78.7% of test listing prices. Selection used validation USD MAE across dollar/log targets and Huber thresholds 0.5/0.9/1.5. The winner used log price and Huber 0.9.

**MiniLM-enhanced training is not complete.** The current workspace has no CUDA GPU. The UI labels that mode as pending and disables it; seller descriptions do not affect baseline estimates. Because the baseline has no `predicted_text_price`, it does not apply the MiniLM-price Isolation Forest filter. Baseline scores must not be presented as results for the enhanced pipeline.

## Full 100% database

All **426,880 source listings** were scanned. Eligibility checks and deterministic description/VIN deduplication leave **211,844 listings**. The assessment partitions contain **148,290 train / 31,777 validation / 31,777 test rows**. Validation selects the Huber configuration before test scoring. The deployment baseline is then **refitted on all 211,844 eligible rows**, with no sampling.

Reported test errors refer to the assessment model. The final all-row refit has no independent holdout; its indicative band reuses assessment validation residuals. The previous 10% scores used a different dataset and holdout, so they are not a paired performance comparison.

Full preparation uses a disk-backed SQLite pass with bounded memory. It assigns deterministic random priorities to all rows and applies text/VIN duplicate matching in that order. There is no database subsampling. The original data-quality rules remain: $1,000–$150,000 asking price, valid posting date and descriptions of at least 80 characters.

```bash
python -m scripts.prepare_full /path/to/vehicles.zip --output training-cache/full-cohort.jsonl
```

Source: Craigslist US asking prices from April–May **2021**. These are not verified sale prices, live valuations or inflation-adjusted prices. Newer cars are outside the training period.

## Requested enhanced pipeline

1. Fine-tune `sentence-transformers/all-MiniLM-L6-v2` on **seller descriptions only**, predicting standardized log1p(price). The pinned MiniLM checkpoint is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with a 384-dimensional representation and 256-token windows. Up to 8,000 cleaned characters use consecutive windows and pooled representations.
2. Three-fold cross-fitting creates **`predicted_text_price`** in dollars. Every training row is predicted by a model that did not see its target or use it in target scaling.
3. `IsolationForest(n_estimators=300, contamination="auto", random_state=42)` fits **only those training OOF dollar prices**. Keep inliers and remove flagged rows from LightGBM training. No removal percentage is forced. Actual prices, tabular features, validation and test do not enter this detector.
4. Huber LightGBM combines the text-price scalar with year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. **Make/model have no transformer, embedding or separate price regressor.**

Isolation Forest can flag valid rare vehicles. Its actual removal count and held-out effects must be measured after MiniLM training. A collapsed text-price signal can leave too few inliers; the run stops rather than silently forcing retention.

Assessment fits OOF models within the chronological training partition and uses train-only regressors for validation/test. After test scoring, a fresh OOF pass covers the entire eligible database, the detector filters that fitting pool, and the deployable model is refitted. All-row refit means every eligible candidate in the complete database, subject to the configured MiniLM-price Isolation Forest filter in enhanced mode. Assessment scores describe the assessment model, not an independent evaluation of the refit.

## Run Car Price Studio

The hosted static app uses `dist/` and includes the trained baseline. Python GUI:

```bash
python -m pip install -r requirements-training.txt
python -m streamlit run app.py
```

The Python GUI defaults to `models/studio-baseline/`. Its mode selector can load an enhanced bundle from `artifacts/car-price-studio-full/` after training. Override with `PRICING_MODEL_DIR` if needed. No API key is required. Baseline browser inputs are processed locally.

For MiniLM training, open `train_gpu.ipynb` in a CUDA runtime or run:

```bash
python -m pricing.train --data training-cache/full-cohort.jsonl --output artifacts/car-price-studio-full --device cuda --folds 3 --epochs 2
```

This is substantial transformer fine-tuning. It uses activation checkpointing, batch size 1 and gradient accumulation. Completed OOF folds resume with matching row IDs, targets, configuration, source revision and code fingerprints. Within each text-model fit, atomic optimizer checkpoints save weights, optimizer state, random state and batch position every ten updates; an interrupted fit resumes without repeating completed updates. Keep the same output directory and arguments to resume. Explicit `--device cpu --allow-cpu` permits slow CPU execution; it is not substituted silently.

To reproduce the baseline:

```bash
python -m pricing.baseline --data training-cache/full-cohort.jsonl --output artifacts/studio-baseline-full --public dist
```

## Checks

```bash
python -m pytest tests -q
node tests/check-browser.mjs
python -m compileall -q pricing scripts app.py
```

Tests cover target exclusion, cache validity, transformer gradients, the complete MiniLM + scalar-only Isolation Forest + LightGBM flow on synthetic fixtures, make/model-transformer removal, and feature/inference schemas. Synthetic scores are not pricing-quality evidence. The browser check verifies exported-tree prices against native LightGBM on 30 exported-model rows (maximum difference below $0.000001). Browser QA covered estimates, custom unseen names, examples, results and training-status labeling.

The GUI error band is indicative, based on validation log residuals. It is not a guaranteed per-car sale range. Back-transformed log predictions are not bias-corrected arithmetic means.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Explainable pricing GUI

After an estimate, the Python GUI offers a plain-English view, native LightGBM TreeSHAP contributions, and a technical view with target scaling, additivity checks, encoded inputs and global split gains. Log-target contributions add in log1p(price) space; they are not additive dollar adjustments. A separate mileage scenario recalculates the fitted model with other submitted fields fixed. These explain model behaviour, not causal effects or MiniLM word-level decisions.

The hosted browser GUI adds an **Explain the price** page, lay explanations, expandable technical notes and actual-tree scenarios for mileage, year and condition. Browser scenarios are sensitivity calculations, explicitly distinct from Python TreeSHAP.

## Integrated structural Pricing + CarBids GUI

The default **Pricing + CarBids** tab transfers auction depreciation/relative-value structure using the vehicle's **age at valuation**, actual mileage and a lagged official BLS used-car price index. It shows the 2021 asking estimate separately, an age/market-adjusted auction equivalent, an experimental asking LightGBM with auction features, decompositions, support warnings and per-car batch export. Both engines and trained weights are bundled; no second service or GPU is required.

The transfer experiment slightly worsened test MAE ($3,329 → $3,338) while improving RMSE ($6,839 → $6,807); the original asking model remains the default. The full 211,844-row Pricing cohort was processed, with 717 unavailable auction equivalents explicitly flagged. Retail-channel premiums remain assumptions, not fitted sale calibration. See [INTEGRATION.md](INTEGRATION.md) for equations/specification, actual assessments, index provenance, leakage controls and reproduction commands.

## Best trained LightGBM ensemble

The promoted regularized ensemble uses **9 members and 18,324 trees**, with CarBids structural features, bagging, categorical smoothing and three fixed seed repetitions. Historical test MAE improved **$3,328.66 → $3,288.05**, and RMSE **$6,839.35 → $6,664.99**. Pricing's normal model selector defaults to it; both shared GUIs use it. The original baseline is preserved.

See [ROBUST_MODEL.md](ROBUST_MODEL.md) for the complete design, capacity/latency tradeoff and reused-benchmark limits. [train_robust_kaggle.ipynb](train_robust_kaggle.ipynb) reproduces the CPU-only robust search without transformer epochs. The earlier transfer-only experiment and its reports are retained as historical comparisons.
