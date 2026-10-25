# pricing

## Current request: 10% sample, Isolation Forest auto, E5-base → Huber LightGBM

Fixed 10% of the raw database: **42,688 sampled listings**, yielding **31,001** eligible/deduplicated listings: **21,700 train / 4,650 validation / 4,651 test**.

Isolation Forest fits training predictors with **`contamination="auto"`**. We retain rows with nonnegative decision function, equivalent to `predict == 1`; the detector determines the outlier count. No removal percentage is forced. Validation/test are never filtered. The shared text stage uses all training rows; this detector filters the LightGBM training stage. Its actual removal count and offset are recorded in the report/audit.

The active Huber sweep has **six** configurations: dollar/log-price targets × thresholds 0.5/0.9/1.5. E5 is trained on description text only, then generates cross-fitted scalar `predicted_text_price` for stacking with structured features. No Extra Trees is used.

**The full E5 run is pending GPU compute.** The exact checkpoint passed a regression gradient/save-load smoke test; any measured tabular-only results are labeled separately. No older or baseline model supplies substitute E5 estimates.

```bash
python -m scripts.prepare_stack /path/to/vehicles.zip --fraction .1 --output training-cache/ten-percent.jsonl
python -m pricing.train --data training-cache/ten-percent.jsonl --output artifacts/e5-base-lightgbm-10 --device cuda --folds 3 --epochs 2 --text-only
python -m pricing.experiments --data training-cache/ten-percent.jsonl --output artifacts/e5-base-lightgbm-10 --losses huber --huber-deltas .5 .9 1.5 --isolation-mode auto --trees 1000
python -m streamlit run app.py
```

`train_gpu.ipynb` uses these settings. The GUI defaults to `artifacts/e5-base-lightgbm-10`. The completed [auto-mode baseline report](reports/AUTO_10_RESULTS.md) records **0 of 21,700 rows flagged**, test **MAE $3,749 / RMSE $7,023** for the validation-MAE winner (log target, Huber 0.5). These are tabular-only results, not E5-stack scores; prior 10% fixed-removal and 25% results below remain archived, not current settings. To explicitly reproduce a fixed-percentage sweep, pass `--isolation-mode rates`.

## Previous 10% experiment: implementation and measured baseline

A used-car asking-price experiment: **fine-tune E5-base on descriptions and pricing targets**, calculate a dynamic dollar-valued `predicted_text_price`, and pass it with structured vehicle characteristics to **LightGBM**.

## Status

The complete training, experiment and inference implementation is included. The exact embaas checkpoint passed an actual regression gradient and save/load smoke test. The full E5 fine-tuning/stack experiment has **not** run: this workspace has no CUDA GPU. No pricing-quality claim is made for the smoke test. A separate measured **tabular-only** 60-configuration baseline is provided under `reports/` when available; it must not be called an E5 result.

The previous pricing model and its weights are removed. The pricing GUI only enables estimates when a completed E5 + LightGBM bundle exists. The hosted static Site shows experiment status, baseline results and a form preview; Python inference runs in `app.py`.

## Completed baseline result

60 tabular-only fits completed. The validation-MAE winner used **log price + Huber + 0% removal**, with held-out test **MAE $3,791, RMSE $6,980**. The validation-RMSE winner used dollar targets + squared error + 0% removal (test RMSE $6,894). Removing 50% raised the best validation-selected model’s test MAE to $4,056. These results do **not** include E5. See [the result table](reports/RESULTS.md) and the complete machine-readable report.

## Fixed 10% of the original database

We sample **42,688 of 426,880 raw listings** uniformly by seeded random priority, before eligibility filtering. This is 10% of the original database, not 10% of an earlier small pilot. Filtering to $1,000–$150,000, descriptions of at least 80 characters, valid dates, and deduplicating effective cleaned descriptions / repeated valid VINs leaves **31,001** listings:

| Partition | Rows |
|---|---:|
| Training | 21,700 |
| Validation | 4,650 |
| Test | 4,651 |

Eligible listings are sorted by posting time and split 70/15/15 chronologically. Seed: 42. Description deduplication uses the effective 8,000-character model input, lowercased for matching. Case is preserved for inference. Exact text/VIN deduplication does not remove every near duplicate.

```bash
python -m pip install -r requirements-training.txt
python -m scripts.prepare_stack /path/to/vehicles.zip \
  --fraction .1 --output training-cache/ten-percent.jsonl
```

## Architecture

| Stage | Inputs | Output |
|---|---|---|
| Fine-tuned `embaas/sentence-transformers-multilingual-e5-base` | Clean seller description only | Standardized log-price regression value |
| Inverse target transform | Text regression value | `predicted_text_price` in USD |
| LightGBM | Text-price scalar plus structured car characteristics | Final regression prediction |
| Inverse experiment-specific transform | Dollar or log-price prediction | Dollar asking-price estimate |

E5-base uses XLM-RoBERTa with **768 hidden dimensions and 512 tokens per window**. Every transformer layer and the regression head receive supervised gradients; this is not a frozen embedding regressor. The source revision is pinned to `ff629add2ff80256e42173ec84299eed1c1b6393` and recorded with the bundle.

Only seller descriptions enter E5. Explicit price phrases, URLs and email addresses are cleaned consistently. Unmarked prices can remain; redaction is imperfect. Each window includes `query: ` and special tokens within the 512-token budget. Up to 8,000 cleaned characters are processed in consecutive windows. Mean-pooled chunk vectors are normalized, averaged, normalized and passed to one regression head. This is chunk pooling, not attention across an 8,000-character context.

Structured inputs are year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and US state. LightGBM uses categorical values for make/model; no separate make/model embedding model is used. Unknown categories become missing. No Extra Trees, Ridge or TF-IDF stage is used.

## Leakage-safe text predictions

Three-fold cross-fitting within training produces every LightGBM training row’s text-price scalar from an E5 regressor that never used that row’s price, including its target-normalization statistics. A final E5 regressor then trains on **all training rows** with the same fixed epoch count for validation, test and inference. Fold models are discarded. No validation/test prices receive transformer gradients.

E5 trains against standardized `log1p(price)` with MSE, two epochs by default. Its scaling is reversed and `expm1` creates `predicted_text_price` in actual dollars. The downstream grid is shared across this one text stage to make loss/filter comparisons practical. Deployment uses more training data for its text regressor than each fold did; this standard stacking setup can introduce feature-distribution differences.

## 60 LightGBM configurations

| Axis | Values |
|---|---|
| Isolation Forest removal | 0%, 5%, 10%, 25%, 50% |
| LightGBM target | Dollars, `log1p(price)` |
| Loss | Squared error / RMSE, absolute error / MAE, Huber, quantiles 0.10 / 0.50 / 0.90 |

5 × 2 × 6 = **60** fits. Squared-error boosting minimizes MSE and is stopped using RMSE; MAE uses absolute error. Huber is stopped using MAE; quantiles use their own pinball loss. Both targets are standardized using retained training targets only. That linear scaling preserves the RMSE/MAE/quantile optimization target and makes Huber’s `.9` threshold meaningful in standardized units rather than fractions of a dollar.

Isolation Forest fits **training features only**, without the actual price column. Inputs are imputed numeric features, one-hot categorical features and the out-of-fold text-price scalar. It uses 300 trees and at most 256 samples per tree. The baseline omits the text-price scalar. A stable score ranking removes exactly `floor(rate × training_rows)` rows, making the subsets nested and ties deterministic. The 0% arm retains all rows. **The shared E5 stage is not retrained after filtering**, so these arms measure filtering at the LightGBM stage. Validation and test are never filtered.

For 21,700 training rows, the retained counts are 21,700 / 20,615 / 19,530 / 16,275 / 10,850. Removing unusual listings can discard valid rare vehicles and damage coverage; the experiment measures the tradeoff.

## Selection and evaluation

Every model and its validation metrics are fixed before test scoring. Two point-model champions are selected using **validation USD MAE** and **validation USD RMSE**. q10 and q90 models are interval endpoints and are excluded from point-model selection; q50 is eligible. The GUI uses the validation-MAE champion.

Each configuration reports USD MAE, RMSE, median absolute percentage error, R², RMSLE and log MAE. Quantile models also report pinball loss. q10/q90 pairs report actual test coverage, signed mean width and crossing frequency; quantiles can cross and nominal 80% coverage is not guaranteed. All 60 test results are exploratory diagnostics for a predeclared sweep and must not be used to reselect the winner.

A separate indicative GUI band uses the 80th percentile of validation absolute log residuals. Its actual test coverage is reported. Back-transforming a log prediction is not a bias-corrected arithmetic-mean price. No result is a guaranteed sale price or appraisal.

## Run the full E5 + LightGBM experiment

[Open the GPU training notebook](https://colab.research.google.com/github/botsandbolts/Pricing/blob/main/train_gpu.ipynb). Select a GPU runtime. This is a substantial full-transformer run, not a quick CPU task. A CUDA PyTorch installation and enough GPU memory are required; batch size 1, activation checkpointing and gradient accumulation control memory.

```bash
python -m pricing.train \
  --data training-cache/ten-percent.jsonl \
  --output artifacts/e5-base-lightgbm \
  --device cuda --folds 3 --epochs 2 --batch-size 1 --accumulation 8 --text-only

python -m pricing.experiments \
  --data training-cache/ten-percent.jsonl \
  --output artifacts/e5-base-lightgbm --trees 1000
```

An explicit `--device cpu --allow-cpu` permits slow CPU fine-tuning; it is never substituted silently. OOF fold caches are tied to data, revision, configuration and training-code fingerprints. Completed folds resume; interrupted final-model training restarts that stage. Keep the artifact directory. The grid refuses to use missing, mismatched, frozen-embedding or in-sample scalar caches. It exports model, schema, target transform and evaluation together.

For a clearly labeled tabular-only CPU baseline:

```bash
python -m pricing.experiments \
  --data training-cache/ten-percent.jsonl \
  --output artifacts/ten-percent-tabular-baseline --trees 1000 --tabular-only
```

This baseline does not export an E5 inference bundle.

## GUI and outputs

Download/unpack the completed notebook bundle so `artifacts/e5-base-lightgbm/manifest.json` exists. Then:

```bash
python -m streamlit run app.py
```

The GUI offers numeric fields, dropdowns, custom make/model fields and long free text. For every request it calculates `predicted_text_price` from the fine-tuned E5 model, combines that scalar with form fields, applies the selected LightGBM model and displays dollars. The user never has to enter the generated scalar. Set `PRICING_MODEL_DIR` for another bundle location. Inference runs on the Python host; the static hosted preview does not perform PyTorch inference.

Outputs include `experiment-plan.json`, `validation-results.json`, `selection.json`, `experiments.json`, `experiment-results.csv`, `isolation-audit.csv`, all 60 LightGBM models, the fine-tuned text model, `oof-audit.csv`, and the GUI-ready champion bundle. Raw seller descriptions and model binaries are not committed to Git.

## Verification

```bash
python -m pytest tests -q
python -m compileall -q pricing scripts app.py
python -m scripts.smoke_pretrained  # optional exact-source CPU gradient/save-load check
```

Tests exercise 512-token chunking, regression gradients through transformer attention, OOF exclusion, category handling, LightGBM serialization, the complete 60-configuration orchestration on tiny synthetic fixtures, and live bundle inference. Synthetic fixture scores are not pricing results. The exact embaas smoke test uses two synthetic examples and one optimizer step only; it verifies weight updates and save/load parity, not accuracy.

## Data limits

Source: [Craigslist Cars & Trucks](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data); [Zenodo snapshot](https://zenodo.org/records/10457828). Historical US **asking prices from April–May 2021**, not verified transaction prices or current valuations. No inflation adjustment, multilingual-market validation or independent validation of GUI-written descriptions is included. Newer vehicles are outside the training period.

Requested model: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-base
LightGBM: https://lightgbm.readthedocs.io/
