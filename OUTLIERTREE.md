# OutlierTree vehicle audit

This independent diagnostic audits both complete eligible Pricing datasets. It does not delete cars, alter Cleanlab, refit models, or change AFT defaults. The observed log price is an audit input, not a pricing-model predictor.

## Reproduce

Use a separate environment with `requirements-outliertree.txt`; on Linux without clang, set `CC=gcc CXX=g++ DONT_SET_MARCH=1` during installation. Prepare the public asking and auction cohorts using the existing preparation scripts, then run:

```bash
python -m scripts.audit_outliertree \
  --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --pct-outliers 0.099 --threads 4 \
  --output reports/outliertree-pct0099
python -m pytest -q tests/test_outliertree_audit.py
```

The audit verifies prepared input SHA256 and eligible row counts against the committed Cleanlab audit. It excludes known asking VIN overlaps from auctions, retains the original chronological phases, and uses the same AFT eligibility rules. Auction prices are converted to the bundled CPI reference dollars; asking prices remain historical asking dollars. Prices are natural-log transformed; mileage is log1p transformed by the shared vehicle preprocessing.

## Interpretation and limitations

OutlierTree searches for conditional numeric and categorical exceptions. A flag can concern a vehicle characteristic instead of price. Flags are review candidates, not verified errors; removing them is not evidence of improved MAE. The primary explanation is not an exhaustive list of all suspicious fields. Smaller reported outlier scores indicate stronger evidence among flagged rows; they are not calibrated error probabilities or a continuous score for every car.

`pct_outliers` affects the internal tail-detection criteria. It does not force that fraction of records to be flagged or deleted. OutlierTree 1.10.0.post4 requires this parameter below 0.10; 0.25 is unsupported. The requested 0.099 rerun keeps the other settings: max_depth=4, min_gain=0.01, z_norm=2.67, z_outlier=8, numeric/categorical minimum group sizes 25/50, binarized categorical splits and tail categorical detection.

Vehicle facts include age, log mileage, auction grade where available, native make/model, body type, transmission, location, fuel, drive, title, cylinders and paint; seller condition is also included for asking cars. No descriptions, transformer predictions, MMR, VIN, identifiers or held-out model predictions enter the detector.

Categories are transported as object strings, which the library factorizes natively. This avoids an observed library issue silently omitting unordered pandas Categoricals. An assertion checks that every categorical input is present in the fitted detector. There is no one-hot or ordinal-number encoding.

The original unpooled asking fit exceeded memory. To keep every car, only infrequent model-name levels are pooled: retain levels occurring at least 20 times, capped at the 1,000 most frequent names, separately within each fitting partition. Make and other categorical columns remain unchanged. Reports record pooling counts; flagged-car exports preserve original names. This limits resolution for rare trims.

`assessment` fits training rows only. `full_cohort` is a separate descriptive fit including validation/test, so its flags must not filter held-out evaluation or support claims of improved model quality. Independent fits can return fewer flags even on larger pools because the conditional reference groups change.

Outputs include aggregate summaries, conditional examples, and compressed flagged-row exports. These audit results do not contain full descriptions, VINs or contact fields.

Official implementation: https://github.com/david-cortes/outliertree
