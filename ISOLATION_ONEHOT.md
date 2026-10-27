# One-hot categorical Isolation Forest audit

This is a full-cohort descriptive comparison with OutlierTree, not a replacement of deployed Cleanlab or AFT pricing models. No cars are deleted. There are no verified anomaly labels in these datasets, so accuracy, precision and recall cannot be measured.

The same SHA256-verified eligible asking and auction cohorts are used, including the auction VIN overlap exclusions. Vehicle facts and observed log price are inputs, matching the OutlierTree audit scope. Auction price is adjusted to the bundled CPI reference month. The detector never sees descriptions, text-price predictions, VIN, MMR or identifiers. Full-data results include validation/test and must not be used to filter held-out model evaluation.

Scikit-learn ColumnTransformer median-imputes numeric inputs and applies OneHotEncoder(handle_unknown='ignore', sparse_output=True) to categorical inputs. Every distinct category is retained, including rare models; unlike the memory-limited native categorical OutlierTree fit, no model-level pooling is needed. The result is a sparse float32 matrix. Numeric scaling is unnecessary for this split-based detector and is not applied.

IsolationForest settings are 100 trees, max_samples=256, contamination='auto', max_features=1.0, bootstrap=False, random_state=42 and n_jobs=4. Max_samples=256 is the per-tree default subsample; every eligible car is a fit candidate and is scored, not a 256-car dataset experiment. Auto uses the library's offset -0.5; it does not select a fixed percentage of outliers. Negative decision_function is flagged; lower score_samples is more anomalous, with decision_function = score_samples - offset. Scores are not calibrated error probabilities.

One-hot encoding can dilute anomaly signals across many sparse columns and overemphasize rarity or feature representation. Counts and agreement alone do not establish that Isolation Forest is better or worse at identifying actual errors. In particular, a low auto flag count does not certify clean data; forcing a contamination fraction would only impose a count quota.

```bash
python -m scripts.audit_isolation_onehot \
  --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --output reports/isolation-onehot-auto
python -m pytest -q tests/test_isolation_onehot.py tests/test_outliertree_audit.py
```

Use the separate requirements-outliertree.txt environment, since the report utility reuses the OutlierTree audit's serialization helper. Run OutlierTree first to create the comparison flags. Outputs contain per-market reports, full row-identified compressed score tables, and an aggregate summary. Reported statistics include flag counts and fractions, score quantiles, phase counts, feature dimensionality, most anomalous examples, and overlap/Jaccard with OutlierTree at pct_outliers=0.099. Large full score tables remain local and reproducible; committed reports include aggregate results and score-table checksums.
