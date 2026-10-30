# Original-data hierarchical pricing laboratory

The **Hierarchical model** panel is a separate research comparison. It automatically loads fitted results from the original public datasets; visitors do not need to upload a CSV. The new experiment uses **SGD-Huber on log prices**, followed by an empirical-Bayes hierarchical residual correction. An additional XGBoost random-forest comparison uses the same cohorts, with its own hierarchical correction. CatBoost is excluded from this experiment. The main landing-page calculator, car-details fields, description dropdown and MiniLM text features remain unchanged.

## Data and reproducibility

Original sources:

- Craigslist Used Cars / Tom, [Zenodo record 10457828](https://zenodo.org/records/10457828), CC-BY 4.0. Advertised asking prices, not confirmed retail transaction prices.
- [Used Car Auction Prices / tunguz](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), CC0. Platform-reported wholesale sale prices; settlement is not independently verified.

All source records are scanned; **100% of eligible canonical rows** enter the assessment. There is no 10% sampling and no residual-based deletion. Existing canonical preparation retains dated, deduplicated asking listings with prices $1,000–150,000 and cleaned descriptions >=80 characters; auction preparation requires valid VIN and source-price provenance and keeps one canonical sale per vehicle. Predictor eligibility checks valid year/date/mileage; missing mileage is allowed. Known auction VINs found in asking descriptions are excluded to avoid cross-source vehicle overlap. Released vehicle identifiers are hashed; no descriptions or raw VINs enter the assessment artifacts.

Separate chronological calibration/validation/test cohorts are retained. Calibration predictions are 3-fold vehicle-group OOF; these folds are group-held-out, not rolling-origin folds. Validation/test predictions come from a calibration-only base fit. All preprocessing is learned inside the appropriate training fold. Neither correction fitting nor parameter selection uses held-out targets. Scores include every eligible validation/test car; none are removed for having large residuals. The example browser displays at most 1,000 reproducibly selected test cases for speed, explicitly labelled as a display subset.

```
python -m scripts.prepare_full training-cache/ask.zip --output training-cache/full-cohort.jsonl --workers 4
python -m carbids.data --input training-cache/bid.zip --output training-cache/bid-cohort.jsonl
python -m scripts.train_original_residual_comparison
python -m scripts.package_original_residuals
```

Download the original archives to `training-cache/ask.zip` and `training-cache/bid.zip`. Each market's provenance includes original archive and canonical SHA-256, full counts, exclusions, date ranges and source attribution. Raw archives are not committed.

## SGD-Huber baseline

Target: natural logarithm of price. Asking dollars are the historical 2021 advertised dollars. Auction targets are indexed to the project's January 2015 CPI reference using the existing used-car index and a two-month release lag. Index snapshot revisions remain a limitation. Do not compare the markets' MAEs as if their price bases and targets were identical. These are historical assessments, not forecasts of current transactions.

Features: make, model, body/type, transmission, state, fuel, drive, title status, cylinders and color where available; age, log mileage, missing-mileage indicator, auction grade and missing-grade indicator. No description encoder, sale outcome leakage, seller identity, MMR or raw sale price is used as a predictor. Model-year age is a proxy; exact manufacture dates are unavailable.

`OneHotEncoder(min_frequency=20, max_categories=50, handle_unknown='infrequent_if_exist')` learns common levels from training rows only. Rare levels share an infrequent bucket; unseen values use that bucket when fitted, otherwise all-zero coding. Numerical features use training-only StandardScaler. The shared residual-regression pipeline has a log-base feature, held constant at zero for this direct price model; it carries no prediction information.

`SGDRegressor(loss='huber', epsilon=0.1, penalty='elasticnet', alpha=0.001, l1_ratio=0.05, average=True, max_iter=3000, tol=1e-5, learning_rate='invscaling', eta0=0.01, power_t=0.25, random_state=42, early_stopping=False)`.

Training log targets are centered using the training-only mean for numerical conditioning; the mean is restored to the fitted intercept. Huber is squared near zero and linear for large residuals; 0.1 log units is roughly a 10% price ratio. It limits influence rather than deleting rows. Predictions are exponentiated and clipped to $1–10 million. No automatic Duan smearing is applied: this is a robust log-price point estimate, not a guaranteed arithmetic expected sale price.

## Hierarchical correction

`log(corrected USD) = log(SGD USD) + global + make + model-within-make + residual-age effect`.

Fit only calibration OOF residuals `log(observed/base)`. Between-group variance is estimated from group-mean variance minus average sampling variance, truncated at zero. Group means shrink by `n*tau²/(sigma²+n*tau²)`. Sparse makes/models borrow strength; unseen groups have zero group offset. A regularized common residual-age slope is shared; model-age slopes require at least 40 calibration rows and three years of age variation.

This is sequential empirical-Bayes partial pooling with Gaussian residual assumptions, **not joint Bayesian MCMC**. Approximate correction standard errors omit hyperparameter uncertainty and dependencies among sequential effects. They are not individual-car sale-price intervals. The optional internal SGD residual-layer helper remains available for research, but the installed SGD comparison uses just its price baseline and hierarchy.

Validation/test report dollar MAE, RMSE, log MAE and signed dollar bias, plus the paired MAE change. A vehicle-group bootstrap with 300 resamples provides a 95% interval for the MAE difference, not for a car price. No test-driven tuning or automatic production promotion is performed. Repeatedly selecting models on this test would weaken its fresh-test interpretation.

## Artifacts, GUI and storage

`models/hierarchical-original/{ask,bid}` contains the fitted base SGD pipeline, hierarchy weights, OOF/held-out signed residuals, full test predictions, display examples, evaluation and provenance. Large compressed artifacts are losslessly split with checksum manifests. Reassemble using `pricing.suite_storage.read_model_file(path)`; CSV artifacts are gzip-compressed. Only the small example/model/evaluation artifacts are loaded for serving; the website never retrains the original dataset during a visit.

The panel presents source/counts, a four-method scorecard when forest results are installed, uncertainty of the MAE change, residual inspection, held-out car examples and an additive correction chart. Technical recipes are expandable. The developer workbench remains optional and capped at 10,000 uploaded observations, whereas installed results cover the full eligible original cohorts. Uploaded data remain session-local. The existing landing-page models and all their description examples are retained.

The existing Turso `hierarchical_estimations` table records only explicit, consented historical held-out comparisons: visitor/session keys, UTC time, city/country, market, base/corrected prices, log correction, approximate standard error, support counts, fingerprint and method. No descriptions, raw row identities, model labels or uploaded files are saved. Parameterized writes are best-effort, with 90-day retention and visitor deletion handling. GUI does not expose private records. GA receives bounded panel/action names, never uploaded rows or descriptions.

## XGBoost random forest comparison

Reproduce with `python -m scripts.train_original_residual_comparison --engine forest --output training-cache/forest-assessment-fit`, then `python -m scripts.stage_forest_results`. Identical source eligibility, chronological splits, training-only preprocessing and OOF fold grouping are used. RF is trained directly on natural-log price for both markets with squared-error objective, not on CatBoost outputs.

Configuration: 200 parallel trees, one boosting round, learning rate 1, maximum depth 8, minimum child weight 20, lambda 10, row subsampling .8, per-node column subsampling .8, hist method, 128 bins, four threads, seed 42. Native categorical vocabularies are learned inside each training fold; unseen levels become missing and follow XGBoost missing routing. No one-hot expansion is necessary for this forest; SGD retains the pooled one-hot recipe. Predictions are exponentiated and clipped to $1–10 million.

The recorded release was trained through `XGBRFRegressor` (200 estimators), which internally creates one boosting round with 200 parallel trees. XGBoost 3.2 deprecates this convenience wrapper. The reproduction code uses its equivalent supported interface, `XGBRegressor(n_estimators=1, num_parallel_tree=200)`. A unit test verifies their predictions agree with identical parameters. This is XGBoost's parallel-tree RF-style algorithm, not an assertion of equivalence to sklearn's conventional bagged random forest. There is no early stopping or validation/test tuning.

Forest results include both RF alone and RF plus the same empirical-Bayes correction. The public panel shows both and uses the same historical example identities for comparison. It does not replace landing-page production defaults or remove description features.

The forest release stages evaluation, provenance, hierarchy weights and display examples. Large RF base models and full row-level RF residuals remain offline outputs, with SHA-256 and exact reproduction commands in `release-scope.json`; the web panel does not serve fresh forest inference. Complete SGD artifacts are versioned in the repository.
