# pricing

Current pipeline: **free-text E5 description-price regression + compact make/model price regression + structured vehicle fields → Huber LightGBM**. Both price scalars are generated **out of fold** for LightGBM training. There is **no outlier filtering** and no 10% sampling in the current workflow.

## Current execution status

OOF generation and the complete stacking/refit/inference flow passed an end-to-end check with a tiny randomly initialized transformer. This is a software check, not a pricing evaluation. **Real full-database E5 fine-tuning and OOF prediction generation have not completed: the current assistant runtime has no CUDA GPU.** The prior transient training process/checkpoints are no longer available, so its completion cannot be verified. The saved make/model head and earlier reports remain historical component artifacts; they do not supply substitute predictions for this pipeline.

## Free-field E5 branch

`embaas/sentence-transformers-multilingual-e5-base` reads only seller-description text. Its source revision is pinned to `ff629add2ff80256e42173ec84299eed1c1b6393`. The XLM-RoBERTa backbone has 768 hidden dimensions and a 512-token window. All transformer layers and a scalar regression head learn standardized `log1p(price)`; inverse scaling and `expm1` produce dollar-valued **`predicted_text_price`**.

Price-like phrases, email addresses and URLs are cleaned consistently. Redaction is imperfect. The `query: ` prefix and special tokens count within the 512-token limit. Up to 8,000 characters are split into consecutive windows, pooled, averaged and normalized. This is chunk pooling, not one 8,000-character attention window. Blank descriptions are retained and receive a prediction from the prefix-only input.

## What out-of-fold means here

`pricing/oof.py` splits the fitting rows into three folds. For each fold:

1. Fit a fresh regression model on the other two folds, including their target-normalization statistics.
2. Predict only the excluded fold without passing its targets to the predictor.
3. Place those dollar predictions back in the original row positions.

Every row is predicted exactly once. The audit CSV contains `row_id`, `fold`, `predicted_text_price` and `predicted_make_model_price`. Each branch has resumable per-fold NPZ files tied to row IDs, targets, split configuration, source revisions and training-code fingerprints. Mismatched checkpoints are rejected. An interruption inside a fold restarts that fold. Final all-fitting-row text checkpoints are saved separately.

Two distinct passes prevent evaluation leakage:

| Stage | LightGBM fitting prices | Validation/test prices |
|---|---|---|
| Chronological assessment | OOF predictions within training rows only | Predictions from a fresh model fitted on training rows only |
| Deployment refit, after assessment | Fresh OOF predictions across all eligible rows | No independent holdout remains |

The deployment description and make/model regressors are then fitted on all eligible rows for incoming requests. The inference regressors see more fitting rows than each OOF model; this standard stacking arrangement can introduce a feature-distribution shift. Shuffled folds are used inside each fitting partition; the outer assessment split remains chronological.

## Make/model branch and LightGBM

The compact frozen `sentence-transformers/all-MiniLM-L6-v2` encoder reads make/model names only. Its supervised neural regression head also learns standardized log price and generates the separately cross-fitted **`predicted_make_model_price`** scalar. No year, mileage or description enters that component.

LightGBM receives both scalar prices plus year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. Raw embedding dimensions are not fed to LightGBM. Categorical vocabularies use fitting rows only; unknown categories become missing.

Six configurations compare dollar/log targets and Huber thresholds 0.5/0.9/1.5. Targets are standardized on fitting rows. Early stopping uses validation MAE in the fitted target scale. Validation USD MAE selects the final configuration before test scoring. Test metrics include dollar MAE/RMSE, median APE, RMSLE and R². The final all-row refit uses the selected configuration and boosting-round count. Test scores describe the assessment model, not the later refitted model. An indicative band reuses assessment validation log residuals; deployment coverage is not guaranteed.

## Full database run

Use [train_gpu.ipynb](train_gpu.ipynb) on a CUDA runtime, or:

```bash
python -m pip install -r requirements-training.txt
python -m scripts.prepare_all /path/to/vehicles.zip
python -m pricing.train --data training-cache/all-cohort.jsonl --output artifacts/e5-oof-all --device cuda --folds 3 --epochs 2 --identity-epochs 20
python -m streamlit run app.py
```

This is a substantial full-transformer fine-tuning run. Batch size 1, activation checkpointing and gradient accumulation control memory. Explicit `--device cpu --allow-cpu` permits slow CPU execution; it is never silently substituted.

Preparation scans the entire original database. It retains every listing with a $1,000–$150,000 target and valid posting date, after duplicate matching by effective cleaned description and valid VIN. There is no description-length filter; empty descriptions stay. Chronological splits are 70% training / 15% validation / 15% test. These data-quality exclusions are reported in the preparation metadata; no learned anomaly detector removes rows.

Key outputs:

```text
assessment/oof-predictions.csv   # Training-only OOF prices for honest assessment
assessment/description-oof/     # Per-fold E5 checkpoints and audit
assessment/identity-oof/        # Per-fold make/model predictions and audit
evaluation.json                # Chronological held-out metrics, before deployment refit
oof-predictions.csv             # Fresh full-cohort deployment OOF prices
text-model/                    # All-row fine-tuned E5 model + scalar head
identity-model/                # All-row make/model regressor
lightgbm.txt
schema.json
manifest.json                  # Written only after the full run finishes
```

The GUI has a free-text seller-description field alongside structured fields. Every request computes both price signals dynamically before LightGBM predicts dollars. `PRICING_MODEL_DIR` can point to another completed bundle. Estimates stay disabled without a completed bundle. The static hosted preview does not run PyTorch inference.

## Checks

```bash
python -m pytest tests -q
python -m compileall -q pricing scripts app.py
```

Tests verify per-row target exclusion, invariance to changing a held-out target, checkpoint resumption and mismatch rejection, real transformer regression gradients, both OOF stages, no-filter LightGBM fitting, full-cohort refitting and bundle inference. Synthetic fixture scores are not model-quality evidence.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-base

The source contains historical April–May 2021 US asking prices, not verified completed-sale prices or current valuations. No inflation adjustment or multilingual-market validation is included.
