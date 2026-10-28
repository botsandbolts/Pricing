> Historical quantile comparison bundle. The current bid default is [CatBoost log-Huber](BID_HUBER.md). The workflow and recorded results below describe the previous quantile release.

# Bid-only CatBoost log-median default

The asking default, its MiniLM text head, CatBoost-AFT weights, model schema,
filter membership and training remain unchanged. Only the auction/bid branch
is refitted. The old auction normal-AFT model remains available for comparison.

## Two different models with the same loss

1. The **cleaner** is CatBoost `Quantile:alpha=0.5` fitted to log dollars.
   Original training residuals come from three-fold out-of-fold predictions.
   Original validation/test residuals come from a model fitted to original
   training only, with fixed cleaning hyperparameters. Their prices do not
   train that cleaner, but subsequently DO select which evaluation cars remain.
2. Within each original phase remove exactly `ceil(0.25 * n_phase)` cars ranked
   by `abs(log(observed_price) - log(predicted_price))`. Break ties by `row_id`.
   This replaces the previous bid masks; there is no union with OutlierTree,
   Isolation Forest or another flagger. These removals are a requested trimmed
   population, not verified erroneous cars.
3. Fit a **new final CatBoost quantile-0.5 model** to the retained training
   cars. Retained validation selects the number of trees; separately score
   retained and original unfiltered test cars. Then refit a deployment model
   on all retained train + validation + test rows at that tree count.

Both stages use log prices and native categorical predictors. The existing
auction date/CPI transport is preserved: divide historical nominal dollars by
the lagged index factor to express the target in January-2015 reference dollars,
then apply the valuation-date factor at inference. This changes neither the
asking reference date nor its target. Residuals are proportional/log errors;
there is no clipping of labels or use of sale prices as predictors.

Original eligible auction membership is recovered from the previous signed-off
row-identity audit, only to preserve the 110 known asking-VIN exclusions and
source eligibility. Its previous retained/removed decisions are NOT reused.
The canonical source SHA256 must match the earlier audit. The source is
**Used Car Auction Prices (Kaggle / tunguz)**, not the Cars & Bids platform.

## AFT versus quantile 0.5

Normal AFT uses a positive-outcome log-normal likelihood. It can use real
censoring intervals when they exist; our existing asking fit uses exact bounds
`[price, price]`. An asking listing is not a verified transaction and is not
by itself evidence of interval censoring. `exp(f(X))` is a median under the
normal-log-price assumption, not an unbiased arithmetic mean of dollars.

At alpha=0.5 the quantile/pinball loss is half the absolute log error.
Exponentiating a conditional log-price median gives a conditional price median.
It does not require normal log-price errors, target mean dollars, eliminate
selection bias, or make residual trimming unnecessary. The losses are sensible
choices for the two requested models; neither is a theorem of optimal car
pricing. Prior original-test results remain a reused benchmark.

- CatBoost objectives: https://catboost.ai/docs/en/concepts/loss-functions-regression
- Probit IMR and assumptions: https://www.stata.com/support/faqs/statistics/inverse-mills-ratio/

## Optional Probit/IMR — cleaner only

The current auction data contain sold cars only. There is **no observed same-
market sold/unsold listing population**, so the actual training run explicitly
sets IMR to **disabled**. Missing/invalid sale prices are not inferred non-sales;
Craigslist is not used as a different-market control population. A constant
`sold=1` column cannot identify a selection model. No constant/fake lambda is
added to pretend selection has been corrected.

The implemented optional input is a canonical JSONL listing population with:

- unique `row_id`, original `phase` in train/validation/test, binary observed
  `sold`, and pre-sale vehicle/date fields;
- sold IDs and phases covering every eligible auction sale;
- genuinely observed non-sales from the same listing population, with a defined
  sale-observation window and defensible population coverage;
- `vehicle_group` for all listing episodes of a VIN when applicable.

`--selection-population PATH` fits a simple L2-regularized sparse **Probit**
selection equation on the training part of that full listing population.
For OOF cleaning it also excludes held-out sold IDs and every matching VIN.
Only pre-sale age, log mileage, auction grade, make, body, transmission, state
and fuel enter the equation. Prices and later sale outcomes are not predictors.

For selection index `z`, the sold-observation IMR is
`lambda = phi(z) / Phi(z)`, computed stably using log-CDFs. This is the normal
Probit formula; a logistic index cannot simply be substituted and called the
same Heckman estimator. Lambda is supplied only to the separate cleaning
CatBoost, never to the final quantile price model or prediction form.

A flexible median-regression cleaner augmented by lambda is not the classical
Heckman two-step conditional-mean regression. Credible exclusion restrictions,
selection/distribution assumptions and coverage remain necessary for a causal
or identified selection-correction interpretation. No such claim is made.

## Evaluation limitation

The requested removal of large-residual validation and test cars uses their
outcomes. Filtered scores are therefore optimistic selected-cohort scores,
not estimates of generalization to all listings. Original unfiltered-test errors
are also reported, but that historical benchmark has already informed model and
filter choices. A new external, unfiltered test is required for an unbiased
comparison. Deployment refit metrics are not held-out assessment scores.

## Reproduction

```bash
python -m scripts.train_bid_quantile \
  --bid training-cache/auction-cohort.jsonl \
  --prior training-cache/prior-bid \
  --output models/bid-quantile-cleaned
```

Optional listing-population input is deliberately absent from the current run.
No transformer retraining or GPU session is needed. Model parts have SHA256 and
size checks; the GUI routes bid requests to this completed quantile bundle,
while asking requests retain `models/minilm-catboost-aft/ask`.

The user-provided Gemini sharing link could not be retrieved. Implementation
follows the user's explicit requested workflow and the primary documentation
above, without claiming to have read the shared conversation.

The complete cleaning masks, OOF checkpoint predictions, cleaner and assessment
weights are archived in `bid-quantile-cleaning-assessment-audit.zip`
(43,122,149 bytes), Library ID `libfile_727c25dcf6c081918cd913b6c5909383`.
The final deployed model is stored in checksum-verified Git parts.
