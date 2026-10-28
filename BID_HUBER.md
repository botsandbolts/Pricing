# Bid-only CatBoost log-Huber default

Ask stays exactly as installed: MiniLM description-price stacking plus CatBoost normal AFT. Its weights, text head, feature schema and masks are unchanged. The new bid default is a fresh **CatBoost `Huber:delta=0.2`** model. The preceding log-quantile-0.5 and normal-AFT bid bundles remain available for comparison.

## Cleaner, quarter removal, final model

Use the full original eligible auction cohort and original chronological phase assignments. The canonical source hash and eligible row identities must match the prior audit. Its retained flags are NOT reused.

1. Fit a separate CatBoost Huber cleaner to natural-log reference-dollar prices. Three-fold OOF predictions supply original training residuals. A training-only fit supplies original validation/test predictions. Fixed cleaner: 800 trees, depth 7, learning rate 0.05, delta 0.2.
2. Within EACH original phase, rank `abs(log(observed)-log(predicted))` and remove exactly `ceil(0.25*n_phase)`, resolving ties by row ID. Recalculate all masks using Huber residuals. No union with old quantile masks, OutlierTree, Isolation Forest or other flags. Source data remain unchanged. These are model-dependent exclusions, not confirmed bad records.
3. Fit a NEW Huber estimator to retained training, with depth 8 and a 1,500-tree cap. Retained validation selects stopping with patience 125. Report original and filtered test diagnostics separately. Refit deployment on all retained train/validation/test cars at that selected tree count.

Both cleaner and final model use native categorical features and gradient leaf estimation. A preliminary Newton-leaf Huber fit produced unstable extreme extrapolations; it was rejected before deployment. The replacement uses first-order leaf updates to avoid division by near-zero Huber curvature in tail-dominated leaves. This is a numerical safeguard, not evidence of better external-test accuracy. Bid has no descriptions in this source, so no transformer branch is fabricated. Existing lagged CPI/date normalization is preserved: fit `log(price/index_factor)` and apply the valuation-date factor to outputs. Data source: **Used Car Auction Prices (Kaggle/tunguz), historical reported selling prices**, not Cars & Bids and not independently verified settlement records. Ask source: **Craigslist Used Cars**, asking listings.

## Why Huber and what delta means

For log residual `u`, Huber is `0.5*u*u` when `abs(u)<=delta`, otherwise `delta*(abs(u)-0.5*delta)`. Small errors have smooth squared-error gradients; large residual gradients are capped in magnitude. This is a reasonable compromise for auction noise and extreme prices, not proof that unusual cars are errors or that it is the best-performing loss.

**Delta 0.2 was fixed before this experiment**, not tuned on the test set. It is in natural-log units: the switch corresponds to observed/predicted ratios `exp(0.2)=1.2214` or `exp(-0.2)=0.8187`, approximately +22%/−18%. It is not a $0.20 tolerance. Other deltas require a separately recorded validation-only experiment.

Each cleaner fold subtracts the median log target of its OWN fitting rows. Assessment uses the retained training log median, and deployment uses the all-retained log median. This translation leaves the Huber residual objective unchanged and avoids saturated gradients from an arbitrary zero-log-price initialization. Prediction restores the saved offset exactly once; TreeSHAP adds it to the base value. The exponentiation yields a positive robust log-location estimate, **not generally a conditional median or unbiased arithmetic dollar mean**. No `log1p`/`expm1` or double logging is used.

The former quantile alpha=0.5 loss minimizes half absolute log error and targets a median. Huber targets a different robust location, characterized by zero expected clipped residual. Huber reduces response-outlier influence but does not solve high leverage, selection bias, or dataset shift. The fixed 25% removal is an additional requested operation, separate from robust loss.

Ask AFT retains its normal log-price likelihood and exact observed bounds `[P,P]`. AFT can model genuine censoring intervals when observed; asking-price data alone do not establish interval censoring. No interval-censoring claim is made for either source. Under normal log errors `exp(f(X))` is a median, not a dollar mean. Neither AFT nor Huber is universally optimal; the GUI displays measured trade-offs.

## Probit and inverse Mills ratio: cleaner only

The optional selection branch is repeated for each Huber cleaner fit when `--selection-population PATH` is supplied. It requires the genuine same-market total listing population, observed sold and unsold outcomes with a defined observation window, unique row IDs, original phase identities and pre-sale fields. Sold rows must cover all eligible bid rows. Fit the selection equation on training population only; exclude held sold IDs and matching VIN episodes during OOF cleaning. Prices never enter its predictors.

A simple regularized Probit estimates `Pr(sold=1|Z)=Phi(Z*gamma)`. For sold observations `lambda=phi(z)/Phi(z)` is computed stably and added ONLY to the cleaning CatBoost. The final Huber model and calculator do not require or accept this feature. A logistic index cannot simply be put into the normal Probit IMR formula and called an equivalent correction.

**Actual run: IMR disabled.** The supplied auction dataset contains sales only; no valid same-market non-sale population is available. A constant sold outcome cannot identify Probit/Logit, missing prices are not inferred non-sales, and Craigslist listings are not fabricated failed auctions. The code rejects one-class selection input. A lambda-augmented nonlinear Huber cleaner is not classical Heckman two-step conditional-mean regression and does not identify population prices without further assumptions, coverage and credible exclusion restrictions.

## Evaluation honesty

Validation/test outcomes rank the residuals and choose which cars remain. Filtered scores are optimistic selected-cohort diagnostics. They cannot establish generalization to all cars. Different cleaners retain different test cars, so filtered MAEs are not paired comparisons. Original unfiltered test scores allow a same-car comparison but are from an already examined historical benchmark. A fresh external unfiltered test is still needed. Deployment refits all retained phases only AFTER separate assessment; its training fit is not a held-out score.

## Reproduce

```bash
python -m scripts.train_bid_huber \
  --bid training-cache/auction-cohort.jsonl \
  --prior training-cache/prior-bid \
  --output models/bid-huber-cleaned --delta 0.2
```

The prior audit supplies only eligible row/phase identity and source checksum. Final weights, manifest, metrics and schema are deployed together. The UI activates Huber only when its complete ready manifest is installed; otherwise it retains the previous completed bid bundle. Current run uses CPU; no Kaggle GPU is started or left running.

References:
- https://catboost.ai/docs/en/concepts/loss-functions-regression — Huber requires explicit delta; GPU support exists.
- https://www.stata.com/support/faqs/statistics/inverse-mills-ratio/ — Probit IMR and selection assumptions.

## Recorded assessment

| Model | Original test MAE USD | Original test RMSE USD | Spearman |
|---|---:|---:|---:|
| New log Huber | 2,302.81 | 5,040.16 | 0.95569 |
| Previous log quantile 0.5 | 2,148.14 | 4,454.94 | 0.95906 |
| Previous normal AFT | 2,090.20 | 4,251.58 | 0.96060 |

Selected deployment trees: 1500; retained rows: 412,635. Filtered test MAE $1,349.83 / RMSE $1,909.34, on 62,702 outcome-selected cars, is optimistic. Original test has all 83,603 cars. The default follows the requested Huber loss and does not imply the lowest measured error.

Complete masks, OOF predictions, cleaning and assessment weights, original-test predictions, code and passing GUI proof are saved in `bid-huber-cleaning-assessment-audit.zip` (41,389,393 bytes), Library ID `libfile_0040679a8da48191b9fb154e1ada3e84`. Final deployment weights are checksum-verified Git parts.
