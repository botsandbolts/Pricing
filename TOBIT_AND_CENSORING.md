# Why AFT is the preferred option, and what Tobit means here

## The recommendation and its metric

**Normal XGBoost AFT is the preferred asking and auction option, and the default
in both calculators.** Among the four new models, it has the lowest recorded
held-out RMSE on both datasets and the lowest auction MAE. This is an empirical
benchmark result, not proof of a universally optimal model.

| Market | AFT MAE | Huber MAE | AFT RMSE | Huber RMSE |
|---|---:|---:|---:|---:|
| Asking, MiniLM + structured fields | $3,169.37 | $3,097.29 | **$6,483.50** | $6,517.16 |
| Reported auction sale | **$1,854.72** | $1,868.38 | **$3,181.75** | $3,701.38 |

RMSE gives large dollar mistakes more weight; MAE counts each dollar of error
linearly. AFT wins this RMSE comparison, while Huber remains the better asking
option by MAE. The asking RMSE difference is modest: $33.66. No paired uncertainty
test has established statistical superiority. The leaderboard describes the
already assessed candidates; choosing a default from it does not create a fresh
unbiased estimate of future performance. The old, larger ensembles are separate
experiments. Our AFT model uses 100 boosting rounds, normal errors and scale 1.

The model has a useful second advantage: it can learn directly from measured
price bounds when future data provide them. That capability is why it is
**Tobit-ready**, even though the current fitted models use exact-label likelihoods.
A censoring-aware tool does not make every input dataset censored. The statistical
hammer is excellent; we still check whether the car is actually a nail.

## Interval censoring in both datasets: evidence versus assumption

Both market datasets can be extended to an interval-censored formulation, **if a
credible observation mechanism supplies lower and upper bounds for each affected
row**. The current project does not establish that mechanism for either dataset.
Its recorded asking or reported sale prices are treated as point labels. The
completed evaluations record zero interval-censored training rows; neither weights
nor scores have been relabelled as a censored-data experiment.

| Dataset | Observed target | What could justify a price interval | What is currently missing |
|---|---|---|---|
| Craigslist asking data | Advertised asking price | A documented rounding rule for the unrounded asking amount, or a verified same-car negotiation range for a separately defined sale target | Per-row rounding rules or paired negotiation/transaction bounds |
| Wholesale auction data | Provider-reported `sellingprice` | Documented rounding of the underlying transaction, or auction rules and bid records that bound a specifically defined latent reservation value | Verified rounding increments, bid histories and reservation-price bounds |

For example, **if** an amount $12,000 was rounded to the nearest $100, the underlying
unrounded amount lies in approximately [$11,950, $12,050). This example applies to
either dataset only when that rounding rule is verified. A round-looking number,
an integer column, or the absence of cents is not evidence of that rule. The
unrounded asking amount remains an asking amount, not a completed sale price.

A seller's ask is not automatically a valid upper bound on eventual sale price.
Likewise, a winning auction amount is an observed transaction target, not by itself
a two-sided interval for the buyer's willingness to pay. Such reinterpretations
require explicit behavioural assumptions, fees/auction rules, and a distinct
latent target. We do not match unrelated cars across the two datasets to invent
an ask–bid interval. CPI-transporting an auction prediction also does not turn that
prediction into an observed lower bound on a retail sale.

## Four different data problems

- **Censoring:** a row is present and its outcome is known only to lie above,
  below, or between specified limits.
- **Truncation/selection:** some rows never enter the analysed sample. Keeping
  only completed auctions or removing prices outside an eligibility range is
  selection; it does not automatically yield censored labels for excluded cars.
- **Measurement error:** a recorded amount differs from the target through an
  error mechanism. Without known limits, this is not interval censoring.
- **A missing different target:** observing an asking price without a sale price
  leaves the sale outcome unknown. That alone supplies neither a sale-price
  interval nor a censored-regression identification strategy.

Cleanlab audits possible label noise. It does not infer censoring bounds, recover
missing transactions, or remove auction-selection bias. These are different jobs;
there is no universal statistical duct tape.

## Classical Tobit

A simple Tobit model assumes a latent continuous response

\[
y_i^*=x_i^\top\beta+\varepsilon_i,
\qquad \varepsilon_i\sim N(0,\sigma^2).
\]

With fixed lower and upper censoring limits, the recorded response is the lower
limit when the latent response falls below it, the latent response itself inside
the observable range, and the upper limit above it. A boundary observation is not
an exact measurement at the boundary: it represents a tail probability.

For prices, a log-scale model is often useful because prices are positive and
relative differences can be more stable than absolute differences. Modelling
`log(price)` is a modelling choice; it does not solve censoring or selection on
its own. A normal error on log price implies a log-normal conditional price law.

Classical Tobit uses a linear location function. Normal AFT with boosted trees
replaces that linear function with a flexible tree ensemble:

\[
\log P_i^* = f_\theta(x_i) + \sigma Z_i,
\qquad Z_i\sim N(0,1).
\]

When real censoring bounds are supplied, this gives a nonlinear, Tobit-like
censored likelihood on log price. It is not the classical linear Tobit estimator,
and price here is the positive response; no time-to-sale outcome is being fitted.
Our implementation fixes `sigma=1`; it does not estimate per-car volatility.

## What the likelihood learns

Let `mu = f_theta(x)`, `z(b) = (log(b) - mu) / sigma`, with `Phi` the standard normal
CDF and `phi` its density. Bounds below are in **dollars**:

| Observation | Bounds supplied to XGBoost | Likelihood contribution |
|---|---|---|
| Exact price `p` | `[p, p]` | `phi(z(p)) / (sigma * p)` |
| Left censored | `[0, U]` | `Phi(z(U))` |
| Right censored | `[L, +infinity]` | `1 - Phi(z(L))` |
| Interval censored | `[L, U]`, `0 < L < U` | `Phi(z(U)) - Phi(z(L))` |

Exact values use a density; their likelihood is not the zero-width interval
probability. Training minimizes negative log likelihood. XGBoost applies the log
transformation internally, so do **not** pass already logged prices as bounds.

With normal log errors, `exp(mu)` is the conditional median; the untruncated
log-normal mean is `exp(mu + sigma^2 / 2)`. The GUI serves XGBoost's median/location
prediction, clipped to its price guardrails. It does not silently multiply prices
by a log-normal mean correction or describe distribution quantiles as calibrated
prediction intervals. AFT is fitted by likelihood, not directly by dollar RMSE;
its observed RMSE advantage is a benchmark finding.

## Identifiability and a credible censored-data experiment

If every observation supplies only an upper limit, pushing the predicted location
far enough down can increase all corresponding probabilities. A set of upper
limits alone generally cannot pin down a useful latent price level without
additional identifying information. Adequate exact outcomes, informative finite
intervals, a justified observation mechanism and sufficient covariate support are
needed. Selection into completed transactions is a further issue; ordinary Tobit
is not automatically a model of that selection process.

The AFT tree accepts interval bounds, but the existing MiniLM target and Cleanlab
audit are point-supervised. A genuinely latent censored-target experiment must
also adapt or justify those upstream steps; adding two columns alone does not
validate the whole pipeline for censored outcomes.

To perform a new interval-censored experiment on **either or both** datasets:

1. Define the target precisely: advertised amount, transaction price, or a latent
   reservation value. Do not switch meanings between fitting and evaluation.
2. Obtain verified row-level bounds and their evidence. Keep exact outcomes exact.
   Record rounding increments, auction rules, fees and any negotiation assumptions.
3. Add `aft_lower_price` and `aft_upper_price` to the prepared cohort. The current
   trainer accepts both columns, validates their bounds, and applies the same CPI
   scaling to auction bounds and targets. Bounds are supervision, not predictors.
4. Keep the original chronological and vehicle-identity boundaries. Never infer
   bounds from test outcomes or use model predictions as observed labels.
5. Use a new output directory. Report exact/left/right/interval row counts, bound
   widths and sources, likelihood on held-out bounded outcomes, and dollar errors
   on genuinely observed point outcomes. Midpoints are not verified sale labels.
6. Compare against the existing exact-label AFT and Huber fits on matching rows.
   Report sensitivity to plausible bounds, scale assumptions and selection.

A supported input format is not proof that these experiments have already run.
The installed scores still belong to the documented exact-label experiment.

## Sources

- [XGBoost: AFT survival analysis and bound encoding](https://xgboost.readthedocs.io/en/stable/tutorials/aft_survival_analysis.html).
- [Stata: Tobit regression, censoring and truncation](https://www.stata.com/manuals/rtobit.pdf).
- [Craigslist source dataset description](https://zenodo.org/records/10457828).
- [Wholesale auction source](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), imported from `sellingprice`; see `carbids/data.py`.
- The project's committed `models/price-model-suite/{ask,bid}/evaluation.json`
  files are the source for the benchmark numbers above.
