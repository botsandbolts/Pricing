# Merchant evidence and Bayesian calibration

## Business English

The merchant report asks whether comparable historical adverts have different asking-price patterns. It reuses the saved Lemon Squeezer predictions instead of training a different vehicle-price model. Each retained training car was predicted by one of five models that did not train on that vehicle group.

Merchant names are candidates extracted from text, not verified business identities. Missing names do not establish that a seller is private. The analysis retains original extracted candidates without an additional confidence cutoff.

Comparisons require shared inventory across vehicle model, age, mileage, location and listing period. Five missing-data imputations show whether conclusions depend on completion of missing fields. Unsupported comparisons stay blank. A merchant's service package, vehicle condition or inventory can explain differences; these are asking-price associations, not claims about fairness, conduct, profit or causal effects.

The Bayesian pricing model already accounts for many vehicle groups and nonlinear patterns. A separate calibration experiment tests whether the size of its correction should vary by evidence and risk group. An experiment is not automatically the live model. The Accuracy comparison panel shows its actual error and interval results.

## Technical English

### Frozen residual provenance

Source: Actions run 41605651393, artifact `lemon-hierarchy-ask`. There are 110,832 retained calibration rows and five vehicle-group-separated folds, each using approximately 80% of those rows. The asking objective is XGBoost `survival:aft`. Its description feature averages two cross-fitted MiniLM dollar-price heads inside the outer training folds.

Residuals are `log(observed_price / predicted_price)`. The loader requires exact row-ID correspondence, unique IDs, all five folds, positive finite predictions, matching recorded dollar prices, matching residuals and matching installed evaluation metadata. It records the export SHA-256. No new merchant vehicle-price model is fitted.

These OOF scores are conditional on the earlier adaptive 25% training selection; they are not nested validation of cleaning. Original validation and test cohorts remain separate.

### Merchant support and uncertainty

Shared cells use make/model/state/listing month, five-year age bands and 25,000-mile bands. At least three adverts per merchant and two merchants per cell are needed. Estimates require minimum total support and vehicle-group support. Vehicle-group resampling respects repeated observations of a vehicle. Five stochastic imputations contribute within- and between-imputation variation.

A later-date slice of the saved OOF errors is a chronological sensitivity analysis. It is not a strict temporal holdout: OOF training folds can contain later observations. Trim and source-date limitations remain explicit. Imputation cannot create shared support or repair uncertain merchant identities.

### Bayesian calibration challenger

The saved robust posterior is held fixed. Vehicle-group hashing divides validation into correction-selection and interval-calibration cohorts. Selection considers a correction strength from 0 to 2 in increments of 0.25 and a multiplicative price-level calibration bounded to exp(-0.1) through exp(0.1). For each strength, the weighted median of observed/predicted ratios, weighted by predicted dollars, minimizes absolute dollar loss over the bounded multiplier.

Risk groups distinguish rare models, starting prices under USD 5,000, and older/high-mileage cars. A group needs 500 selection cars; otherwise the global setting applies. Separate validation cars calibrate 90% log-price outcome ranges, with 100 cars required for a group-specific scale. Other groups use the global scale.

The historical test compares dollar MAE with the unchanged Bayesian model, paired vehicle-group bootstrap savings and interval coverage. A positive lower 95% savings bound and coverage between 88% and 95% are experiment gates. Passing them is not automatic deployment. The report discloses reuse of this historical test in previous development.

This is a validation calibration layer, not a new posterior. Ranges condition on the selected calibration settings and do not include their selection uncertainty. Neither historical coverage nor this merchant analysis validates current-date price extrapolation.
