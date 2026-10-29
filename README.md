> **Deal Detective + social sharing:** import an advert/screenshot, compare a three-car shortlist, explore historical sales, play the price challenge, and download landscape/portrait PNG cards with QR links. Share through WhatsApp, LinkedIn, Facebook, Reddit, email or your device. Vehicle links are opt-in and omit seller text/visitor IDs. [Features, privacy, provenance and limitations](DEAL_DETECTIVE.md).

> **New studio design and experimental Value over time panel.** Current-value Ask/Bid scenarios separate categorical depreciation from a release-aware market clock, with an interactive same-car timeline and optional annual mileage. The historical MiniLM+AFT Ask and quantile-0.5 Bid weights are unchanged. [Method, research projects, backtests and limitations](TIME_TRANSPORT.md).

> **Default: MiniLM + CatBoost normal AFT for asking prices.**
> MiniLM reads the description and produces `predicted_text_price`; newly fitted CatBoost combines it with vehicle facts. Make/model remain categorical fields, without transformers.
> **Bid default: CatBoost quantile 0.5 on log prices**, newly fitted after a separate log-quantile CatBoost cleaner removes the highest-residual 25% within each original train/validation/test phase. Ask weights are unchanged.
> Sources: **Craigslist Used Cars** (asking listings) and **Used Car Auction Prices · Kaggle/tunguz** (auction selling prices). The auction dataset has no descriptions, so its CatBoost model remains tabular.
> Optional Probit/IMR is cleaner-only and currently disabled: no same-market sold/unsold total listing population is available. See [bid quantile workflow and limitations](BID_QUANTILE.md).
> Ask retains its original residual/OutlierTree exclusions. Bid now uses its separately recomputed quantile-residual 25% cuts only. Filtered validation/test scores are optimistic; original-test diagnostics are reported separately.
> See [MiniLM + CatBoost architecture and cross-fitting](MINILM_CATBOOST.md). Earlier architectures and scores below are historical.

# pricing · Car Price Studio

Car Price Studio is a used-car asking-price GUI with custom make/model fields, vehicle dropdowns, mileage/year inputs, a dollar estimate, an indicative error band and measured test results.

**Installed legacy architecture:** no make/model transformer; the **full 100% raw database**; MiniLM description-price regression plus **Isolation Forest in auto mode** for the enhanced workflow; Huber LightGBM as the final regressor.

## Cleanlab training migration

New asking-price and auction-sale training uses **Cleanlab regression auditing** instead of Isolation Forest. It examines vehicle facts and observed training log prices using internal CV and uncertainty estimation; validation/test cars remain unfiltered. The structural auction trainer now also supports Cleanlab. **Installed weights retain their actual legacy filter until retrained**; the GUI makes that status explicit. See [training commands, audit records and deployment safeguards](CLEANLAB.md).

## What works now

The Python studio defaults to the **installed MiniLM + Isolation Forest + LightGBM model**. MiniLM reads seller descriptions, a training-only Isolation Forest filters unusual OOF text-price signals, and LightGBM combines the resulting scalar with the form fields. The completed assessment weights run on CPU; their recorded test MAE is **$3,559**. The full-data refit is a separate checkpoint-resumed Kaggle job.

The trained tabular baseline and regularized ensembles remain selectable for comparison. The static browser app runs the actual exported baseline trees. Its deployment baseline, fitted on all 211,844 eligible rows, is included under `dist/baseline-model.json`, and Python weights/schema are under `models/studio-baseline/`; no hand-written pricing formula is substituted.

Measured baseline test results: **MAE $3,329, RMSE $6,839, median APE 15.6%, R² 0.774**. The validation-calibrated 80% band covered 78.7% of test listing prices. Selection used validation USD MAE across dollar/log targets and Huber thresholds 0.5/0.9/1.5. The winner used log price and Huber 0.9.

**MiniLM assessment training is complete and the model is installed.** Its transformer was fitted on 148,290 descriptions; automatic Isolation Forest filtering retained 109,609 OOF rows for LightGBM. The separate 211,844-candidate full-data refit is not yet installed. The GUI labels the scope explicitly and serves point estimates because the saved assessment omitted its interval calibration radius. Baseline scores above describe the baseline, not MiniLM.

## Description examples and recomputed audit

The Python calculator now offers **400 searchable real seller descriptions** with matching editable vehicle fields. Select an example, edit it if desired, then estimate. Prices and common contact/VIN formats are redacted; dataset attribution appears in the GUI.

Evaluation compares every bundled asking-price model and the text branch. A reproducible new audit uses **1,000 validation and 1,000 test listings**, sampled from chronological halves of the original assessment test pool. MiniLM + LightGBM MAE is **$3,192 / $3,732** (validation/test); text-only MAE is **$3,446 / $3,767**. These are reused historical holdout data, not a fresh untouched benchmark. Deployed tabular/ensemble weights were refitted on all rows, so their recomputed errors are explicitly labelled **in-sample**. Original pre-refit test results remain separate. See [audit methodology and full-pool command](AUDIT.md).

## Full 100% database

All **426,880 source listings** were scanned. Eligibility checks and deterministic description/VIN deduplication leave **211,844 listings**. The assessment partitions contain **148,290 train / 31,777 validation / 31,777 test rows**. Validation selects the Huber configuration before test scoring. The deployment baseline is then **refitted on all 211,844 eligible rows**, with no sampling.

Reported test errors refer to the assessment model. The final all-row refit has no independent holdout; its indicative band reuses assessment validation residuals. The previous 10% scores used a different dataset and holdout, so they are not a paired performance comparison.

Full preparation uses a disk-backed SQLite pass with bounded memory. It assigns deterministic random priorities to all rows and applies text/VIN duplicate matching in that order. There is no database subsampling. The original data-quality rules remain: $1,000–$150,000 asking price, valid posting date and descriptions of at least 80 characters.

```bash
python -m scripts.prepare_full /path/to/vehicles.zip --output training-cache/full-cohort.jsonl
```

Source: Craigslist US asking prices from April–May **2021**. These are not verified sale prices, live valuations or inflation-adjusted prices. Newer cars are outside the training period.

## Historical Isolation Forest pipeline · installed assessment

1. Fine-tune `sentence-transformers/all-MiniLM-L6-v2` on **seller descriptions only**, predicting standardized log1p(price). The pinned MiniLM checkpoint is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, with a 384-dimensional representation and 256-token windows. Up to 8,000 cleaned characters use consecutive windows and pooled representations.
2. Three-fold cross-fitting creates **`predicted_text_price`** in dollars. Every training row is predicted by a model that did not see its target or use it in target scaling.
3. `IsolationForest(n_estimators=300, contamination="auto", random_state=42)` fits **only those training OOF dollar prices**. Keep inliers and remove flagged rows from LightGBM training. No removal percentage is forced. Actual prices, tabular features, validation and test do not enter this detector.
4. Huber LightGBM combines the text-price scalar with year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. **Make/model have no transformer, embedding or separate price regressor.**

Isolation Forest can flag valid rare vehicles. In the installed assessment it removed 38,681 of 148,290 training rows; every validation and test row remained in evaluation. A collapsed text-price signal can leave too few inliers; the training run stops rather than silently forcing retention.

Assessment fits OOF models within the chronological training partition and uses train-only regressors for validation/test. After test scoring, a fresh OOF pass covers the entire eligible database, the detector filters that fitting pool, and the deployable model is refitted. All-row refit means every eligible candidate in the complete database, subject to the configured MiniLM-price Isolation Forest filter in enhanced mode. Assessment scores describe the assessment model, not an independent evaluation of the refit.

## Run Car Price Studio

The hosted static app uses `dist/` and includes the trained baseline. Python GUI:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The Python GUI defaults to MiniLM + Isolation Forest + LightGBM and opens the free-text valuation form. Its mode selector includes the installed MiniLM assessment bundle and can also load a completed full-refit bundle from `artifacts/car-price-studio-full/`. Override with `PRICING_MODEL_DIR` if needed. No API key is required. Baseline browser inputs are processed locally.

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

Tests cover target exclusion, cache validity, transformer gradients, the complete MiniLM + Cleanlab + LightGBM flow on synthetic asking and auction fixtures, make/model-transformer removal, and feature/inference schemas. Synthetic scores are not pricing-quality evidence. The browser check verifies exported-tree prices against native LightGBM on 30 exported-model rows (maximum difference below $0.000001). Browser QA covered estimates, custom unseen names, examples, results and training-status labeling.

The GUI error band is indicative, based on validation log residuals. It is not a guaranteed per-car sale range. Back-transformed log predictions are not bias-corrected arithmetic means.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## Explainable pricing GUI

After an estimate, the Python GUI offers a plain-English view, native LightGBM TreeSHAP contributions, and a technical view with target scaling, additivity checks, encoded inputs and global split gains. Log-target contributions add in log1p(price) space; they are not additive dollar adjustments. A separate mileage scenario recalculates the fitted model with other submitted fields fixed. These explain model behaviour, not causal effects or MiniLM word-level decisions.

The hosted browser GUI adds an **Explain the price** page, lay explanations, expandable technical notes and actual-tree scenarios for mileage, year and condition. Browser scenarios are sensitivity calculations, explicitly distinct from Python TreeSHAP.

## Integrated structural Pricing + CarBids GUI

The **Pricing + CarBids** comparison tab transfers auction depreciation/relative-value structure using the vehicle's **age at valuation**, actual mileage and a lagged official BLS used-car price index. It shows the 2021 asking estimate separately, an age/market-adjusted auction equivalent, an experimental asking LightGBM with auction features, decompositions, support warnings and per-car batch export. Both engines and trained weights are bundled; no second service or GPU is required.

The transfer experiment slightly worsened test MAE ($3,329 → $3,338) while improving RMSE ($6,839 → $6,807); the original asking model is retained as a comparison baseline. The full 211,844-row Pricing cohort was processed, with 717 unavailable auction equivalents explicitly flagged. Retail-channel premiums remain assumptions, not fitted sale calibration. See [INTEGRATION.md](INTEGRATION.md) for equations/specification, actual assessments, index provenance, leakage controls and reproduction commands.

## Best trained LightGBM ensemble

The promoted regularized ensemble uses **9 members and 18,324 trees**, with CarBids structural features, bagging, categorical smoothing and three fixed seed repetitions. Historical test MAE improved **$3,328.66 → $3,288.05**, and RMSE **$6,839.35 → $6,664.99**. The comparison panels use this structured ensemble; the main valuation form now defaults to MiniLM + Isolation Forest + LightGBM. The original baseline is preserved.

See [ROBUST_MODEL.md](ROBUST_MODEL.md) for the complete design, capacity/latency tradeoff and reused-benchmark limits. [train_robust_kaggle.ipynb](train_robust_kaggle.ipynb) reproduces the CPU-only robust search without transformer epochs. The earlier transfer-only experiment and its reports are retained as historical comparisons.

### Trained Isolation Forest asking experiment

The separate filtered ensemble is trained on the full original cohort with automatic predictor-only filtering. Test MAE: **$3,368**, compared with **$3,288** for the unfiltered ensemble. The **Best trained ensemble** option continues to select the unfiltered ensemble. Select **Isolation Forest trained ensemble** in the GUI to try the completed filtered model. Full counts, metrics and reproducibility details are in [ROBUST_MODEL.md](ROBUST_MODEL.md); The updated [train_filtered_kaggle.ipynb](train_filtered_kaggle.ipynb) now trains the Cleanlab successor. Reproduce the historical Isolation Forest experiment from commit `4e712f3`.

### Interactive price explanations

After estimating a car, click **Build all explanation graphs**. The studio adds a
whole-model dollar waterfall, native member SHAP beeswarm, dependence scatter,
comparison decision plot, mileage sensitivity curve and year × mileage heatmap.
Each includes a lay explanation, humour, hover details and explicit technical
units. The technical receipt reports reconstruction checks and Monte Carlo
sampling errors. Reference CSV uploads are optional; the default is a clearly
labelled 30-car historical gallery. See [EXPLANATIONS.md](EXPLANATIONS.md) for the
methods, target-scale handling, background limits and validation.


### Installed description transformer

Select **MiniLM + Isolation Forest + LightGBM** in the Python GUI to price a car
using both its free-text description and structured fields. The real fine-tuned MiniLM
weights are included; inference runs on CPU with no API key or model download.
`pip install -r requirements.txt` installs the CPU runtime.

This is the completed 148,290-description assessment model, with LightGBM fitted
on 109,609 Isolation Forest inliers. The separate 211,844-car refit is still pending;
the installed bundle explicitly records this distinction. Recorded test MAE: $3,559.
MiniLM + Isolation Forest + LightGBM is now the default studio choice; the regularized ensemble remains selectable for comparison. Point estimates are available;
the assessment did not persist its interval calibration radius. All six explanation
graphs support this model, conditional on the submitted description-price signal.
See [model provenance and installation](models/minilm-assessment/README.md).

### Modest ask and auction options

The **New model laboratory** compares default-sized Huber LightGBM, median quantile LightGBM and normal XGBoost AFT, with a Ridge feature baseline. It reports dollar errors, same-car Spearman matrices, and validation-fitted Ridge slopes, intercepts and test bias. See [MODEL_SUITE.md](MODEL_SUITE.md) for training, text OOF boundaries and the censoring assumptions.
