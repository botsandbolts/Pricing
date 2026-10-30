# Five-stage production model card

Release: 30 October 2026. The five-stage recipe is the user-requested default across the main calculator, Two-market garage, Value over time, Deal Detective, challenge/share outputs derived from those estimates, and Model lab's default pair. Explicitly selected historical comparison models remain available. This is **not an accuracy-based promotion**: all four cleaned models performed worse than their equivalent unfiltered-training baselines on the original test cohorts.

## Models and economic targets

- **Asking default:** fine-tuned MiniLM description-price feature + CatBoost normal SurvivalAft. Learns advertised Craigslist prices from April–May 2021, not completed retail sales.
- **Auction default:** tabular CatBoost Quantile:alpha=0.5 on log reference-dollar prices. Learns platform-reported wholesale selling prices from the 2014–2015 Used Car Auction Prices dataset (Kaggle / tunguz). Settlement is not independently verified. This source is not the Cars & Bids enthusiast website.
- Asking quantile and auction AFT are separately fitted comparison alternatives using the same retained cohort in their respective market.
- Neither target is a guaranteed current offer. The difference between asking and auction prices is not dealer profit or a negotiated discount.

## Five-stage cleaning

A separate tabular CatBoost median regressor scores absolute log-reference-price residuals. At each of five rounds, remove ceil(0.05 × surviving rows) independently in the original training, validation and test partitions. Refit the cleaner each round. Removed rows never return. Roughly 22.6% are removed overall, not 25%; about 77.4% survive.

Training residuals use three-fold KFold cross-fitting, shuffled with seed 42. A fourth model fits surviving training rows and predicts surviving validation/test rows. Ties are deterministically resolved by row ID. The cleaner excludes MiniLM's price signal, avoiding indirect reuse of its fold targets. No Isolation Forest, ExtraTrees or OutlierTree mask is unioned into this run. No IMR is used: the source lacks a genuine total sold/unsold listing population.

Each round comprises four fitted cleaners per market, or 40 cleaner fits across both markets. Cleaner settings: 400 iterations, depth 6, learning rate .05, L2 10, random strength 1, Bayesian bootstrap, bagging temperature 1, seed 42; GPU device 0. Validation/test predictions are training-only, but their observed prices are then used to select their retained rows.

## Eligibility and survivor accounting

Full canonical cohorts were processed, not a 10% sample. Canonical asking cohort: 211,844 rows; final model-eligible cohort: 211,127. Canonical auction cohort: 550,291 rows; eligibility/index availability and asking-VIN overlap exclusions leave 550,181. Source deduplication and chronological splits precede cleaning. Canonical 70/15/15 splits yield the exact eligible phase counts below after eligibility checks.

| Market | Phase | Before cleaning | After round five |
|---|---|---:|---:|
| ask | train | 147,777 | 114,345 |
| ask | validation | 31,668 | 24,502 |
| ask | test | 31,682 | 24,512 |
| bid | train | 384,198 | 297,283 |
| bid | validation | 82,380 | 63,742 |
| bid | test | 83,603 | 64,688 |

Deployment fits use all retained phases: **163,359 asking cars** and **425,713 auction cars**. This full-data deployment refit follows a separate assessment; it is not itself a new held-out experiment.

## Features and preprocessing

Numeric inputs: vehicle age, log mileage and auction grade (missing when unknown). Native categorical inputs: manufacturer, model/trim, body type, transmission, state, fuel, drivetrain, title status, cylinders and paint colour. Category text is lowercased/trimmed with an unknown label. Model levels require at least 20 occurrences in original training and are capped at 1,000 retained levels; other trims use the saved rare-model label. No one-hot encoding or make/model embeddings are used. Seller condition adjectives are not converted into auction inspection grades.

The asking final model additionally uses one dynamic **predicted_text_price** feature. Its existing fine-tuned sentence-transformers/all-MiniLM-L6-v2 head is preserved, not retrained in this CatBoost release. The encoder reads descriptions only, in 256-token windows (including special tokens), up to 8,000 characters. Masked mean pooling and normalization aggregate windows before the scalar regression head. MiniLM learns standardized log1p prices; inverse transformation produces a historical dollar-price feature. This differs from the final CatBoost log target.

Assessment text features use training-only OOF predictions and a training-only head for held-out rows. The deployment fit uses separate full-cohort OOF exports paired with the matching full-data head. Canonical hashes, row ordering, fold assignments and provenance were checked before training. MiniLM serving fingerprint: b65bc3bf487ca9dd4abea44de0c5a80ef358d7919c3644365ff0037a7a75b018. Auction training has no descriptions; no text information is invented.

## Final training, losses and inference

All final models allow 1,500 iterations, depth 8, learning rate .05, L2 10, random strength 1, Bayesian bootstrap and bagging temperature 1. Seed 42, four CPU threads; CPU fits also use rsm=.8. Validation stopping uses patience 100. Actual selected tree counts follow below. The same selected count is used for a separate all-retained-phase deployment fit. GPU floating-point reductions are not bitwise reproducible.

Normal AFT uses SurvivalAft:dist=Normal;scale=1.0 on exact positive bounds [P,P], with an internal log transformation. No interval censoring is established by these datasets. Quantile uses Quantile:alpha=0.5 on log(P). Serving exponentiates the raw score with a numerical upper cap of $10 million. These are median-type outputs, not arithmetic means; Duan smearing is not applied. AFT is a log-normal modelling assumption, not evidence that asking prices are censored sale prices.

| Bundle | Role | Trees | Retained refit rows | Hardware |
|---|---|---:|---:|---|
| ask | Default | 1,490 | 163,359 | CPU |
| bid | Default | 825 | 425,713 | GPU |
| ask-quantile | Comparison | 1,372 | 163,359 | GPU |
| bid-aft | Comparison | 1,500 | 425,713 | CPU |

## Historical assessment results

USD MAE below is calculated after converting auction reference-dollar predictions back to each observation's date. Baselines are freshly fitted equivalents without the five-stage row filtering, using the same model family/settings and stopping policy. They are not necessarily the previous production weights.

| Model | Original test cars | Baseline full-test MAE | Cleaned full-test MAE | Filtered-test MAE | Excluded-test MAE | Full-test RMSE | Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask | 31,682 | $3,097.05 | $3,402.38 | $2,002.38 | $8,188.54 | $7,209.20 | 0.9101 |
| bid | 83,603 | $1,863.23 | $2,235.22 | $1,443.26 | $4,943.67 | $4,695.88 | 0.9537 |
| ask-quantile | 31,682 | $2,969.83 | $3,342.53 | $1,969.81 | $8,035.44 | $7,139.16 | 0.9131 |
| bid-aft | 83,603 | $2,058.75 | $2,255.12 | $1,411.35 | $5,140.77 | $4,944.74 | 0.9534 |

**Consequences:** cleaning improved selected-test AFT/asking scores in some comparisons, but increased full-test error for every final model; auction quantile even worsened on the selected test against its equivalent baseline. High residuals can be genuine rare or expensive cars. Filtering can reduce representation and does not prove label errors. This is not a live outlier detector: cars entered into the calculator are not discarded by these training masks.

Validation and test prices determine which observations survive. Filtered scores are optimistic selected-cohort results. The full original test population is preserved in diagnostic reports, but historical test results have informed project development. Sequential cross-fitting is not fully nested cross-validation. An independent new external cohort is required for an unbiased improvement claim. Per-bundle evaluation.json also records validation errors, signed bias, log MAE/RMSE, excluded-test errors and baseline/cleaned prediction Spearman agreement. Original-test evaluation must not be confused with a new score for the all-retained deployment refit.

## Today's values and model explanations

The pricing anchors remain historical. The existing date-transport layer separately adjusts vehicle ageing and a public monthly BLS/FRED used-car CPI index, with bounded forecast horizons. Auction targets are normalized to January-2015 reference dollars; asking anchors use May 2021. Depreciation components and market-clock backtests are separate evidence, not validated present-day individual-car accuracy. Retail CPI is explicitly a wholesale proxy. Future scenarios and generation analogues remain assumptions. Model years through 2021 are supported in asking; auction generations after 2015 require the existing explicit analogue option where offered, and Deal Detective rejects them. Mileage and condition are interpreted at the chosen date.

Native CatBoost TreeSHAP explains the fitted log-price margin. Contributions add on that scale, not as independent dollar discounts. The dynamic MiniLM feature remains one tree feature; phrase perturbation analysis separately explains the whole asking response. Neither is causal. No validated individual-car prediction interval is supplied. BART-base-CNN narrates the supplied valuation report and does not calculate prices.

## Artifacts, provenance and reproduction

Saved run: https://www.kaggle.com/code/botsandbolts/car-price-studio-five-stage-catboost?scriptVersionId=363794926 (version 2). Runtime 4,181.1 seconds; saved batch completed and no active Kaggle event remained. GPU T4 was used for cleaners/quantile, CPU for CatBoost AFT. CatBoost version 1.2.10. Frozen source bundle SHA256: add9b321f5e1536b5d1c271d70398d863d5a6aa1a9a822fdc78ad93c700d1d14.

Production bundles live in models/iterative-catboost/{ask,bid,ask-quantile,bid-aft}. Manifests include parameters, features, category mapping, source SHA and text provenance. Each bundle includes schema, evaluation and filter reports. Root summary.json contains all four assessments. Cleaning subdirectories contain stage counts/thresholds and recipe metadata. Compressed deployment weights are losslessly split into checksummed parts; the loader validates every part and the assembled SHA256. training-run.json records the downloaded package checksum. serving-verification.json records real inference, schema/provenance, survivor accounting, full-test sizes, dynamic MiniLM and native SHAP checks.

The original saved Kaggle package retains assessment weights, per-row original-test predictions and every stage mask. Serving bundles deliberately do not package original seller descriptions or training rows. Training code: scripts/train_iterative_catboost.py; portable notebook: train_iterative_catboost_kaggle.ipynb. Full methodology: ITERATIVE_CLEANING.md. Production promotion changes the serving defaults; it does not retrain the time-transport curves or summarizer.
