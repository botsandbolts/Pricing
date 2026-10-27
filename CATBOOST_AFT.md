# CatBoost AFT price model and residual diagnostics

This experiment interprets the requested AFC loss as AFT, using CatBoost's `SurvivalAft:dist=Normal;scale=1.0`. It is an assessed tabular alternative, not a replacement of deployed pricing defaults. Both markets retain every eligible car; neither OutlierTree nor Isolation Forest filters are applied.

## Target and interpretation

For normal errors, the model represents log(price) = f(vehicle facts) + scale * error. Supply positive dollar bounds `[price, price]` to CatBoost; it logs the bounds internally. Logging the labels first would model log(log(price)), an incorrect target here. RawFormulaVal is the fitted log-price location; exponentiating it yields the conditional median under normal log errors. It is not automatically the mean dollar price. Auction prices are normalized to the existing CPI reference month for fitting and restored to each observation's month for evaluation.

We have exact observed asking or wholesale sale labels, not verified interval censoring. Asking price is a different economic target from a completed retail sale. The model uses exact equal bounds; it does not invent sale intervals or claim to identify retail transactions. With exact labels and normal errors, AFT fitting has the log-squared-error structure, so AFT is not intrinsically robust to label errors.

## Fitting and evaluation

Use original chronological train/validation/test partitions and existing AFT eligibility/VIN overlap rules; input SHA256 and eligible counts are asserted against the committed audit. CatBoost handles make/model and other categorical predictors natively. No one-hot encoding, descriptions, transformer text-price feature, observed price input, MMR or identifiers are passed as predictors. This isolates the tabular comparison; it is not an evaluation of the transformer stack.

Parameters: up to 1,000 iterations, depth 6, learning rate 0.05, L2 leaf regularization 3, seed 42, four CPU threads. Validation selects the best stopping iteration, with patience 100. Test remains untouched by fitting and model selection. Saved `assessment.cbm` models use training labels with validation stopping; they are not full-cohort deployment refits.

## Reading the residual analysis

Residual = observed - predicted: positive dollar residual means underpricing, negative means overpricing. Log residual = log(observed / predicted), expressing proportional error. Dollar MAE/RMSE, log MAE/RMSE, signed bias, median absolute error and Spearman correlation are reported separately for validation/test.

Plots show dollar/log residual histograms, log normal Q-Q, residuals versus fitted values, and mean observed/predicted price by prediction decile. Test residual skewness, excess kurtosis, quantiles, make-level metrics (at least 30 test cars), and Spearman correlation of absolute error with fitted price help assess heavy tails, calibration and heteroscedasticity. Make-level comparisons reflect different car mixes and prices, not causal effects or standalone fairness claims.

A review rule learns the median and 1.4826 * median absolute deviation of log residuals on validation, then flags test absolute robust-z > 3.5. These are model-disagreement candidates, not confirmed label errors, and are never removed. Empirical 2.5%/97.5% residual ratios summarize held-out error spread; they are not validated conditional prediction intervals.

A separate smearing correction estimates mean(exp(log residual)) on validation only and evaluates corrected dollar predictions on test. The primary reported model remains exp(raw); smearing is an explicitly reported alternative, not a test-selected winner. No unvalidated exp(scale^2/2) correction is imposed.

```bash
pip install -r requirements-catboost-aft.txt
python -m scripts.train_catboost_aft \
  --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --output reports/catboost-aft-residuals
python -m pytest -q tests/test_catboost_aft.py
```

Official target/loss specification: https://catboost.ai/docs/en/concepts/loss-functions-regression#SurvivalAft
