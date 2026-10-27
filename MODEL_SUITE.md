# Modest asking and auction model laboratory

The **New model laboratory** tab keeps the existing studio models and adds three
separate selectable models on each market side, plus a Ridge reference:

| Option | Target / settings | Meaning |
|---|---|---|
| LightGBM Huber | log1p(price); default 100 estimators, learning rate 0.1, alpha 0.9 | Robust price centre |
| LightGBM median | log1p(price); objective quantile, alpha 0.5; other defaults | Conditional median price |
| XGBoost normal AFT | positive dollar bounds; 100 boosting rounds; normal scale 1; native categories | Log-normal location / median |
| Ridge reference | log1p(price); one-hot categories; standardized numeric fields; alpha 1 | Linear baseline |

Seed 42 and four CPU threads control reproducibility and runtime. These are modest
single models, not tuned ensembles. Dollar MAE, RMSE, RMSLE, R² and median absolute
percentage error are reported on the original chronological test partition.
Cleanlab audits training prices from structured vehicle facts only; it never sees
held-out targets or target-trained stacking features. A second audit and fit use
all eligible rows for deployment **after** assessment. Cleanlab flags are suspected
label problems, not proof that a seller or auction made an error.

Auction models use age at valuation, not raw model year, and a two-month-lagged
used-car CPI adjustment to January 2015 reference dollars. Bid estimates are
historical wholesale auction sale-price proxies, not executable offers. All known
VIN overlaps with the asking cohort are excluded from auction fitting.

## AFT and Tobit: what the data identify

XGBoost `survival:aft` with normal errors fits `ln(price) = tree(x) + sigma Z`.
Use **dollars**, not already logged prices, in `label_lower_bound` and
`label_upper_bound`; the AFT likelihood handles the logarithm.

The implementation accepts optional `aft_lower_price` and `aft_upper_price`
columns. Exact labels use `[price, price]`; left censoring `[0, upper]`; right
censoring `[lower, +infinity]`; measured intervals `[lower, upper]`. Both columns
are required together; invalid bounds fail before fitting. The Python `fit`
function also accepts `lower` and `upper` arrays.

The available datasets provide exact asking or reported auction prices, **not
measured censoring bounds**. Their default run therefore uses exact bounds.
This is an uncensored log-normal price model. Calling an asking price an upper
bound on a latent sale price would impose an unverified assumption. Treating every
listing as left censored would not identify a useful latent sale-price level.
A genuine Tobit analysis needs measured censoring and enough uncensored outcomes;
this run makes no claim to recover confirmed retail sale prices.

## Correlations and Ridge comparisons

`comparison.json` contains the complete Spearman matrix and all directed Ridge
pairs across the eight new models on the **same asking test cars**, with auction
predictions transported to those cars' dates and ages. No unrelated auction and
asking row numbers are paired. Positive rank agreement does not prove price
accuracy, causal influence, or paired transaction agreement.

Each pair fits `response = intercept + beta * predictor` on asking **validation**
cars, and reports test agreement. Ridge alpha is 1 after standardizing its single
predictor on validation data; beta and intercept are converted to original dollar
units. Beta is dollars per dollar; intercept is dollars; bias is the test mean of
`prediction − reference`. Every side also reports calibration to its own observed
prices and a separate Ridge feature-model baseline. No test labels fit calibration.

## Reproduce

Prepare the full cohorts as described in README and run:

```bash
pip install -r requirements-cleanlab.txt
python -m pricing.model_suite \
  --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --output models/price-model-suite-tabular
```

For the description-enabled comparison, add `--text-features DIRECTORY` and use
`--output models/price-model-suite`. The directory must contain original
`assessment-oof.csv`, `held-predictions.csv`, `full-oof.csv`, `provenance.json`,
and `full-text-provenance.json`. Cohort hashes, identities, row order, original
three-fold assignments and head fitting counts are checked. Training-only OOF and
train-only-head held-out inference are used in assessment; full-cohort OOF is used
only in deployment. Reusing verified completed MiniLM fits does not require another
transformer fine-tune. The auction source has no genuine descriptions, so it has
no invented text branch.

The compatible default studio MiniLM bundle can be refitted with:

```bash
python -m pricing.refit_cached \
  --data training-cache/full-cohort.jsonl \
  --text-features training-cache/cleanlab-text-features \
  --full-text-model training-cache/kaggle-completed/artifacts/car-price-studio-full/text-model \
  --output models/minilm-cleanlab
```

Large reconstructable audit-row exports stay local; their hashes and full audit
settings are recorded in compact reports. Compressed model files preserve all
weights losslessly. The three new options and Ridge reference are available in
**New model laboratory**; existing studio models remain accessible.
