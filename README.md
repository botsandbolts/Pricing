# pricing

Current pipeline: **fine-tuned E5-base text regression → scalar `predicted_text_price` → Isolation Forest training filter → Huber LightGBM with tabular features and a second MiniLM make/model-price scalar**. LOF and Extra Trees are not used.

The fixed 10% sample contains 42,688 of 426,880 raw database listings. Eligibility checks and description/VIN deduplication leave 31,001 rows: 21,700 training, 4,650 validation and 4,651 test, split chronologically. Source prices are historical US asking prices from April–May 2021, not verified sales or current valuations.

## Status

The implementation and synthetic end-to-end tests pass. **The full E5 fine-tuning run and real Isolation Forest evaluation have not run: no CUDA GPU is available here.** The exact requested checkpoint passed a supervised-gradient and save/load smoke test on two synthetic examples, which is not a pricing evaluation. No actual Isolation Forest removal count, held-out score or price estimate is claimed yet. Previous reports under `reports/` are explicitly historical configurations and are not results for this scalar-only Isolation Forest pipeline.

## Text regression and scalar feature

Use `embaas/sentence-transformers-multilingual-e5-base`, pinned to revision `ff629add2ff80256e42173ec84299eed1c1b6393`. It uses XLM-RoBERTa, a 512-token window and 768-dimensional hidden vectors. All transformer layers and a scalar regression head are fine-tuned on standardized `log1p(price)` with MSE. **Only cleaned seller descriptions enter the transformer.** Make/model also enter a separate compact transformer price model, described below; their categorical values remain LightGBM inputs.

The `query: ` prefix and special tokens fit within 512 tokens. Up to 8,000 characters are processed through consecutive windows with mean pooling and normalization. This is chunk pooling, not attention over the entire document. Price phrases, URLs and email addresses are redacted consistently; unmarked prices can remain.

Three-fold cross-fitting creates every training row’s text-price feature from a model that did not see its target, including its target-scaling statistics. Inverse scaling and `expm1` produce one dollar-valued `predicted_text_price`. A final text regressor learns from all training rows for validation, test and inference. Deployment can have a feature-distribution shift because it uses more training data than each fold model.

## Compact make/model price model

A frozen `sentence-transformers/all-MiniLM-L6-v2` encoder (revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) reads **make and model names only**. It uses at most 64 tokens and mean-pooled, normalized 384-dimensional vectors. A neural 384→64→1 regression head learns standardized log1p(price), with 60 fixed training epochs, AdamW and dropout. The encoder is not fine-tuned on prices; its regression head is. The main E5 description model still receives full supervised fine-tuning.

Three-fold cross-fitting on the same training partition generates **`predicted_make_model_price`**, a dollar-valued scalar. The final head learns all training rows for inference. LightGBM receives both price scalars and the structured fields, not the MiniLM vector. Isolation Forest remains restricted to `predicted_text_price`.

This component has been **trained and evaluated** on the real 10% cohort: 21,700 train / 4,650 validation / 4,651 test; 7,760 unique make/model strings across the cohort. Test **MAE $7,803 / RMSE $12,103**, R² 0.302. These are make/model-only component scores, not full-stack results. The feature cannot distinguish years or mileage within a make/model. See `reports/make-model-evaluation.json`.

The checkpoint saves the frozen encoder/tokenizer, learned head and scaling under `identity-model/`. `identity-features.npz` and `identity-oof-audit.csv` preserve fold-generated scalars and row IDs. Fingerprints and data hashes reject mismatched caches. Run identity training before the E5/LightGBM stages.

## Isolation Forest: only predicted_text_price

`IsolationForest(n_estimators=300, max_samples=256, contamination="auto", random_state=42)` fits **one column**, the out-of-fold `predicted_text_price` in dollars. The sample size is capped at the number of training rows. `fit_predict` labels inliers `1` and outliers `-1`; only inliers train LightGBM. The learned score threshold determines the removal count; no fixed percentage or rank-based removal is applied.

The detector does not read actual prices, residuals, descriptions, embeddings or tabular characteristics. Validation/test stay untouched, and GUI submissions are not rejected by the detector. The shared E5 model trains on all training rows before this filtering stage.

`isolation-text-price-audit.csv` records row ID, out-of-fold predicted text price, isolation score and keep/remove flag. The report records the offset, actual removal count and detector configuration. The detector now uses predicted text price alone; historical multivariate Isolation Forest results do not evaluate this configuration.

Unusual predicted prices can belong to valid rare or expensive cars. An outlier label does not establish that a listing is wrong. Judge the filter by unfiltered held-out performance. The real removal count remains unknown until full E5 predictions have been generated.

## Huber LightGBM and evaluation

Six configurations compare **dollar / log1p targets × Huber thresholds 0.5 / 0.9 / 1.5**. Targets are standardized on retained training rows so thresholds are meaningful. Early stopping uses validation MAE in the fitted target scale. Model selection uses validation USD MAE or validation USD RMSE; test results do not choose the winner. The GUI uses the validation-MAE champion.

Predictors are both dynamic price scalars, year, odometer, make, model, condition, cylinders, fuel, title status, transmission, drive, body type, paint color and state. Categorical vocabularies use training rows; unseen values become missing. Test metrics include USD MAE/RMSE, median APE, RMSLE, log MAE and R². The indicative GUI band uses validation absolute log residuals and reports actual test coverage. Back-transforming log predictions does not guarantee an arithmetic-mean dollar price.

## Run on a GPU

Use [train_gpu.ipynb](train_gpu.ipynb), selecting a CUDA runtime, or:

```bash
python -m pip install -r requirements-training.txt
python -m scripts.prepare_stack /path/to/vehicles.zip --fraction .1 --output training-cache/ten-percent.jsonl
python -m pricing.identity --data training-cache/ten-percent.jsonl --output artifacts/e5-base-two-prices-10 --device cpu
python -m pricing.train --data training-cache/ten-percent.jsonl --output artifacts/e5-base-two-prices-10 --device cuda --folds 3 --epochs 2 --text-only
python -m pricing.experiments --data training-cache/ten-percent.jsonl --output artifacts/e5-base-two-prices-10 --losses huber --huber-deltas .5 .9 1.5 --trees 1000
python -m streamlit run app.py
```

This substantial full-transformer run uses activation checkpointing, batch size 1 and gradient accumulation. Completed OOF folds resume only with matching data/config/code fingerprints. An interrupted final text model restarts that stage. Slow CPU fine-tuning requires explicit `--device cpu --allow-cpu`.

The experiment runner requires the fine-tuned out-of-fold text-price cache. A tabular-only substitute is rejected because it cannot supply the requested Isolation Forest input. The fresh bundle directory prevents an older filter’s model from appearing as the new result.

## GUI and verification

The Python GUI defaults to `artifacts/e5-base-two-prices-10`, or `PRICING_MODEL_DIR`. It combines dropdowns/numeric fields/custom make-model fields with long description text. Each request generates its text price dynamically before LightGBM returns dollars. Estimates remain disabled without a completed bundle. The hosted static page explains the current architecture and status; it does not run PyTorch inference.

```bash
python -m pytest tests -q
python -m compileall -q pricing scripts app.py
```

Tests include transformer regression gradients, token windows, OOF boundaries, Isolation Forest on a single scalar with an isolated-tail example, full train/filter/LightGBM orchestration on synthetic fixtures, resume and bundle inference. Synthetic scores are not pricing-quality results.

Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Model: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-base
Isolation Forest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

## Use the trained make/model component without retraining

The learned price head is committed as `models/make-model-price/head.json`. Its public frozen MiniLM encoder is downloaded at the pinned revision on first use; no API key is required.

```bash
python -m scripts.predict_make_model Toyota Corolla
```

This returns the trained `predicted_make_model_price` signal (about $7,506 for Toyota Corolla), with no year or mileage adjustment. This is a 2021 make/model-only signal, not a current individual-car valuation. Full-stack inference still requires a completed E5 bundle. The GPU notebook reproduces all component training and the final LightGBM experiment.
