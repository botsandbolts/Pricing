# Bid model evaluation: RMSE in dollars, no RMSE training

**Decision (6 November 2026):** Keep the original fitted model
objectives unchanged (log-price Huber, exact-observation AFT, or median
quantile). Do **not** retrain model weights to optimize RMSE. Recompute
and rank their **evaluation metrics** using original dollar RMSE,
original dollar MAE, and hard-to-miss tail statistics.

## Why dollar RMSE?

Given the original auction sale price P_i and an out-of-sample prediction
P_hat_i, define

RMSE_USD = sqrt(mean((P_hat_i - P_i)^2))
MAE_USD = mean(abs(P_hat_i - P_i))

RMSE makes one very expensive pricing mistake count more than many
small mistakes. This is relevant to automotive appraisal decisions when
outliers create disproportionate financial exposure.

RMSE is NOT "average absolute error", a confidence interval, or an
individual vehicle price bound. Rare exotic vehicles, erroneous seller
records and heteroskedasticity can dominate squared errors. For this
reason the report also includes absolute-dollar error p90/p95/p99/p99.9,
average error of the worst 1%, bias and fractions with errors above
USD 5k, 10k and 20k. Separately reported MAE remains important.

## Research sources — cited in the Streamlit app

1. Chai, T. & Draxler, R. R. (2014). *Root mean square error (RMSE) or
   mean absolute error (MAE)? – Arguments against avoiding RMSE in the
   literature*. **Geoscientific Model Development**, 7, 1247–1250.
   DOI: https://doi.org/10.5194/gmd-7-1247-2014
   Defends the information contributed by RMSE; not a claim it is always best.
2. Hodson, T. O. (2022). *Root-mean-square error (RMSE) or mean absolute
   error (MAE): when to use them or not*. **Geoscientific Model
   Development**, 15, 5481–5487.
   DOI: https://doi.org/10.5194/gmd-15-5481-2022
   Neither is universally superior; appropriateness depends on error
   distribution and application.
3. Willmott, C. J. & Matsuura, K. (2005). *Advantages of the mean
   absolute error (MAE) over the root mean square error (RMSE) in
   assessing average model performance*. **Climate Research**, 30,
   79–82. DOI: https://doi.org/10.3354/cr030079
   Counterpoint: MAE communicates average absolute error more directly.

These papers concern the measurement properties of accuracy metrics, not
a new validated auction cost model.

## What is evaluated, and what is not changed?

The RMSE-only review reads cached ORIGINAL validation predictions from:

- 29 trained FT-Transformer + Huber models, run 42256103702
- 10 trained FM/FFM/RFF kernel models, run 42275395253
- 26 FM-family models from the existing full vs Lemon Huber sweep,
  run 42279280866 (once finished)

Original cohort: 384,198 full training auctions, 288,148 Lemon training
auctions, 82,380 untouched VIN-disjoint validation auctions, and 83,603
untouched test auctions. Auction prices are historical 2015-reference
normalized US dollars. The unchanged frozen XGBoost Lemon per-vehicle
holdout predictions form the exact row-paired comparator; published full
XGBoost scores are aggregate/unpaired.

For each study, rank **existing** models by original validation RMSE.
Independently rank a small preset grid of XGBoost Lemon blend
coefficients by validation RMSE, with price weights
0, .1, .2, .35, .5, .75 and 1. Score only validation-selected model
checkpoint(s) on the original untouched test set; do not score
every candidate's test and then declare the best.

**No training step is included in the RMSE evaluation workflow.**
No changes are made to the price-fitting objective, model architecture,
training split, cleanup policy, feature engineering, live pricing
pipeline, or stored weights.

Streamlit panels:
- **🧠 The science** — readable RMSE vs MAE guidance and DOI citations.
- **📊 Evaluation** — scientific explanation and verified RMSE tables
  when published.
- **🧬 Experimental models** — compact literature reference.

New reports are published under reports/bid-rmse/<study>/;
they are deliberately hidden from the app until the status and
provenance fields pass strict checks. The existing production model
is not automatically replaced by any winner.
