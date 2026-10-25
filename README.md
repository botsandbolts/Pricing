# pricing

Current pipeline: **fine-tuned E5-base text regression → scalar `predicted_text_price` → Local Outlier Factor training filter → Huber LightGBM with tabular features**. Isolation Forest and Extra Trees are not used.

The fixed 10% sample contains 42,688 of 426,880 raw database listings. Eligibility checks and description/VIN deduplication leave 31,001 rows: 21,700 training, 4,650 validation and 4,651 test, split chronologically. Source prices are historical US asking prices from April–May 2021, not verified sales or current valuations.

## Status

The implementation and synthetic end-to-end tests pass. **The full E5 fine-tuning run and real LOF evaluation have not run: no CUDA GPU is available here.** The exact requested checkpoint passed a supervised-gradient and save/load smoke test on two synthetic examples, which is not a pricing evaluation. No actual LOF removal count, held-out score or price estimate is claimed yet. Previous reports under `reports/` are explicitly historical configurations and are not results for this LOF pipeline.

## Text regression and scalar feature

Use `embaas/sentence-transformers-multilingual-e5-base`, pinned to revision `ff629add2ff80256e42173ec84299eed1c1b6393`. It uses XLM-RoBERTa, a 512-token window and 768-dimensional hidden vectors. All transformer layers and a scalar regression head are fine-tuned on standardized `log1p(price)` with MSE. **Only cleaned seller descriptions enter the transformer.** Make/model use categorical LightGBM features, not separate embeddings.

The `query: ` prefix and special tokens fit within 512 tokens. Up to 8,000 characters are processed through consecutive windows with mean pooling and normalization. This is chunk pooling, not attention over the entire document. Price phrases, URLs and email addresses are redacted consistently; unmarked prices can remain.

Three-fold cross-fitting creates every training row’s text-price feature from a model that did not see its target, including its target-scaling statistics. Inverse scaling and `expm1` produce one dollar-valued `predicted_text_price`. A final text regressor learns from all training rows for validation, test and inference. Deployment can have a feature-distribution shift because it uses more training data than each fold model.

## LOF: only predicted_text_price

`LocalOutlierFactor(n_neighbors=20, contamination="auto", novelty=False)` fits **one column**, the out-of-fold `predicted_text_price` in dollars. The metric is Euclidean distance in that one-dimensional space. Use `fit_predict`; retain label `1`, remove label `-1` from **LightGBM training only**. No forced percentage or rank-based removal is applied.

The detector does not read true prices, residuals, descriptions, embeddings or tabular characteristics. Validation and test are never filtered, and new GUI submissions are not rejected by LOF. The shared E5 model is not retrained after filtering. Its purpose is to filter unusual predicted-price density before fitting LightGBM.

`lof-audit.csv` records row ID, out-of-fold text price, negative outlier factor and keep/remove flag. The report records the actual neighbor count, threshold and removal count. If fewer than 21 training rows are available, neighbors are capped at n−1. A configurable `--lof-neighbors` option changes the neighborhood size.

Rare or expensive valid cars can have sparse local predicted-price density. LOF does not establish that their listings are wrong. Equal or nearly equal predictions can affect local-density estimates. The filter must be judged by held-out performance.

## Huber LightGBM and evaluation

Six configurations compare **dollar / log1p targets × Huber thresholds 0.5 / 0.9 / 1.5**. Targets are standardized on retained training rows so thresholds are meaningful. Early stopping uses validation MAE in the fitted target scale. Model selection uses validation USD MAE or validation USD RMSE; test results do not choose the winner. The GUI uses the validation-MAE champion.

Predictors are the dynamic text-price scalar, year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. Categorical vocabularies use training rows; unseen values become missing. Test metrics include USD MAE/RMSE, median APE, RMSLE, log MAE and R². The indicative GUI band uses validation absolute log residuals and reports actual test coverage. Back-transforming log predictions does not guarantee an arithmetic-mean dollar price.

## Run on a GPU

Use [train_gpu.ipynb](train_gpu.ipynb), selecting a CUDA runtime, or:

```bash
python -m pip install -r requirements-training.txt
python -m scripts.prepare_stack /path/to/vehicles.zip --fraction .1 --output training-cache/ten-percent.jsonl
python -m pricing.train --data training-cache/ten-percent.jsonl --output artifacts/e5-base-lof-10 --device cuda --folds 3 --epochs 2 --text-only
python -m pricing.experiments --data training-cache/ten-percent.jsonl --output artifacts/e5-base-lof-10 --losses huber --huber-deltas .5 .9 1.5 --lof-neighbors 20 --trees 1000
python -m streamlit run app.py
```

This substantial full-transformer run uses activation checkpointing, batch size 1 and gradient accumulation. Completed OOF folds resume only with matching data/config/code fingerprints. An interrupted final text model restarts that stage. Slow CPU fine-tuning requires explicit `--device cpu --allow-cpu`.

The experiment runner requires the fine-tuned out-of-fold text-price cache. A tabular-only substitute is rejected because it cannot supply the requested LOF input. The fresh bundle directory prevents an older filter’s model from appearing as the new result.

## GUI and verification

The Python GUI defaults to `artifacts/e5-base-lof-10`, or `PRICING_MODEL_DIR`. It combines dropdowns/numeric fields/custom make-model fields with long description text. Each request generates its text price dynamically before LightGBM returns dollars. Estimates remain disabled without a completed bundle. The hosted static page explains the current architecture and status; it does not run PyTorch inference.

```bash
python -m pytest tests -q
python -m compileall -q pricing scripts app.py
```

Tests include transformer regression gradients, token windows, OOF boundaries, LOF on a single scalar with an isolated-tail example, full train/filter/LightGBM orchestration on synthetic fixtures, resume and bundle inference. Synthetic scores are not pricing-quality results.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-base
LOF: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
