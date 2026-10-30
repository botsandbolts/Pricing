# Hierarchical residual laboratory

The new **Hierarchical model** panel is a separate research experiment. Landing-page CatBoost defaults are unchanged. There are no installed trained hierarchical weights: deployed aggregate reports do not contain the signed, group-aware OOF calibration residuals required to fit this layer honestly.

## Model

`log(corrected USD) = log(base USD) + global + make + model-within-make + residual-age effect`.

Sequential empirical-Bayes partial pooling uses calibration residuals `log(observed/base)`. Between-group variance is estimated by group-mean variance minus average sampling variance, truncated at zero. Group means shrink by `n*tau²/(sigma²+n*tau²)`. Unseen groups have zero group offset. All cars share a ridge-regularized residual-age slope; model-specific slopes need at least 40 calibration rows and three years of age variation. This is a lightweight sequential approximation, not joint hierarchical MCMC. Standard errors omit estimated-hyperparameter uncertainty and dependencies between sequential effects. They are not calibrated sale-price intervals.

## Honest evaluation input

CSV columns: `row_id,vehicle_group,split,prediction_kind,base_fold,manufacturer,model,age,observed_price,base_price,base_model_id,price_basis`.

- Separate ask and bid files; one base-model identity and common dollar price basis per file.
- `split`: calibration, validation or test (minimum 30/10/10 rows).
- Calibration predictions must be genuinely group-aware OOF (`prediction_kind=oof`, at least two `base_fold` values).
- Validation/test base predictions must exclude those vehicles and their labels from all base-model/text training (`prediction_kind=heldout`). Do not use the installed all-phase deployment refit to generate an unbiased historical test.
- Vehicle groups cannot cross folds or partitions. The app enforces these declarations but cannot verify upstream training provenance from CSV alone.
- No residual removal from any partition. Validation and test each report dollar MAE/RMSE, log MAE and dollar bias for exactly the same supplied cars. Paired vehicle-group bootstrap (300 resamples) reports uncertainty of the MAE difference. Repeated model selection on this test invalidates a fresh-test interpretation.
- The text branch remains inside the supplied CatBoost base prediction. No retraining, synthetic accuracy results or automatic promotion occurs in the web app.

Offline fitting: `python -m scripts.train_hierarchical --residuals ask-oof-and-heldout.csv --side ask --output models/hierarchical/ask` (repeat with bid). The panel can also fit an uploaded file and export model/evaluation JSON. Uploads and prediction rows remain session-local; the export may contain normalized make/model group labels from the upload. Do not upload personal data.

## Database

Turso table `hierarchical_estimations` is created idempotently by the normal analytics initializer. It records only explicit, consented historical-heldout comparisons: visitor/session foreign keys, UTC timestamp, city/country, side, base/corrected dollar estimates, log correction and approximate standard error, make/model support counts, model fingerprint and method. No uploaded rows, row IDs, descriptions, names, raw model labels or source filenames are saved. GUI does not expose database records. Writes are parameterized, best-effort and do not affect inference.

Rows expire after 90 days in existing visit/daily retention jobs. Visitor deletion accounts for the new foreign-key references. `historical_heldout_check` distinguishes research comparisons from landing-page price requests. Analytics consent and configured Turso credentials remain mandatory. Creation is deployed as an automatic migration; existing secret configuration is reused.

## GUI and analytics

An explanation, readiness status and template precede the experiment upload. Results show paired evaluation before the per-car comparison, support counts and additive log corrections. There is no claimed current-price accuracy. GA uses `panel_name=hierarchical_model`, bounded `hierarchical_action` and download events. Uploaded fields are never forwarded to GA.

## Automatic CatBoost residual generation

The workbench accepts raw observation columns `row_id,vehicle_group,split,manufacturer,model,age,odometer,observed_price,price_basis`. It generates three-fold group-held-out calibration predictions and validation/test predictions from a calibration-only CatBoost fit. Ask uses normal AFT with exact positive price bounds; bid uses quantile 0.5 on natural log prices. Fixed parameters: 250 rounds, depth 6, learning rate .05, L2 10, seed 42, two CPU threads. No early stopping or test-driven selection. All supplied rows are kept. Signed dollar and log residuals are downloadable.

This is an explicitly separate tabular assessment pilot, not a claimed evaluation of existing all-phase deployment weights. It excludes MiniLM: genuine end-to-end OOF for the text branch would require nested encoder/head refitting inside each base fold. The existing-residual route can accept properly nested text/base predictions when available. No source training cohorts are bundled in the web deployment. CPU pilot limit is 10,000 uploaded rows. The same helper is available offline: `python -m scripts.generate_hierarchical_residuals --observations cars.csv --side bid --output models/hierarchical/bid`.
