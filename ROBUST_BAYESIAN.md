# The second opinion: Bayesian redesign

The old layer added Gaussian make/model offsets and one age slope with a single noise variance. This replacement changes that statistical structure while keeping regular MiniLM/XGBoost prices separately available.

## Business English

An inexpensive high-mileage car and a premium low-mileage car need different corrections and error bars. The redesign learns curved age, mileage and baseline-price patterns, shared vehicle-detail corrections, make-specific age slopes, and different residual noise levels by price/age. It removes no additional adverts.

Model names with fewer than 12 calibration examples share a rare-model pool within their make. Unseen groups receive additional uncertainty. This does not invent evidence for an unseen trim. The GUI compares regular prices, the previous hierarchy and the replacement on identical historical test cars, and shows difficult-segment results rather than hiding them behind an overall average.

## Technical English

The response is log(observed/base), from genuine retained-training five-fold OOF residuals. The fixed 16-column basis contains intercept, age, log mileage, log base price, numeric auction grade and its missingness indicator, three fixed hinge functions each for age/log mileage/log price, and an age-by-log-mileage interaction. Predictors never use observed prices.

Cross-classified Gaussian random intercepts cover make, model-within-make/rare-model pool, state, body, transmission, fuel, drive, title, retail condition, cylinders and paint where available and nonconstant. Auction grade is numeric, not a retail-condition category. Makes have random age slopes. Twelve predicted-price-by-age cells have separate observation variances.

Gaussian, Student-t(4) and a labelled narrow/wide Gaussian contamination likelihood share this structure. Two conditional Gibbs chains sample coefficients, effects, variances and latent precisions/allocations. Proper Normal/inverse-gamma priors regularize coefficients and effects. Contamination scale ratio is fixed at five and its probability has a Beta(1,9) prior. Collapsed intercept/effect and common-age/make-slope updates reduce confounding. Stored posterior draws serve requests; MCMC never runs in the app. Dollar estimates inverse-transform the posterior median location. Split R-hat <1.1 is required for monitored coefficients and variance parameters; it does not prove every latent effect has converged.

## Evaluation and deployment

The original training-only 25% residual trim remains fixed. These later OOF errors are selection-conditional, not nested CV of the entire selection procedure. No validation or test car is removed by that rule.

Group-stable hashing splits validation into roughly two-thirds model selection and one-third interval calibration. Selection minimizes dollar MAE among candidates passing monitored convergence. Held-out log-price predictive ELPD is reported too; it is not PSIS-LOO. Non-Gaussian density integration uses a variance approximation for unseen effects.

The separate calibration cars set Mondrian split-conformal 90% multipliers. Predefined strata cross rarity (<12 model examples), baseline price <$5,000, and age >12 years or mileage >150,000. A stratum needs 100 calibration cars; otherwise it falls back to the rarity class and then the global pool. Actual support and fallback are shown in the GUI. Coverage needs exchangeability within a group; fallback does not establish sparse-group or current-market coverage.

After selection is locked, the unchanged original test cohort measures point error and range coverage. Promotion requires validation improvement, positive lower 95% paired vehicle-group-bootstrap MAE savings versus both old Bayes and regular prices, monitored convergence, and overall historical coverage 88–95%. Reports retain all candidates and overlapping subgroup results. Overall coverage is not a guarantee for every subgroup.

The same historical test cohort has been reported in earlier releases; it is not newly collected external evidence. Current-date age/index transport and generation analogues remain unvalidated. Asking and auction datasets are different populations, without matched dealer spreads or verified censoring intervals. No joint spread, Tobit bounds or current-sale accuracy is invented.

Workflow **Robust Bayesian hierarchy deployment** pins baseline weights at d6534b97830cde2ab1a01bfdab327bae9884226c and consumes the matched evidence artifact from run 41230037897 (subject to retention). It compares candidates, archives results, rechecks promotion/base identities, smoke-tests inference and memory, then publishes only passing posteriors. Active scores are in models/xgboost-hierarchy/{ask,bid}/bayesian/bayesian-evaluation.json and the GUI Accuracy comparison panel.
