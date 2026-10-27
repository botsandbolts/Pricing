# Requested larger XGBoost AFT defaults

The user requested removal of the highest-residual 25% in every original train, validation and test partition, plus all earlier outliers. This is deliberately a selected cohort, not an unbiased anomaly-cleaning benchmark. It is not valid to claim an improvement in generalization from the filtered test MAE.

## Filter rule

Use absolute log-price residual abs(log(observed/predicted)) so the ranking measures proportional disagreement instead of automatically discarding expensive cars. Within each original phase, remove ceil(25% * rows), with row_id as deterministic tie breaker, then union all OutlierTree full-cohort flags at pct_outliers=0.099. The previous one-hot Isolation Forest auto run flagged no cars. Thus removals can slightly exceed 25%. No original cohort files are overwritten; retained membership is an explicit audited view.

Training residuals come from three-fold out-of-fold CatBoost normal AFT predictions (1,000 iterations, depth 6), never in-sample predictions. Each training car is absent from the model that scores it. Original validation/test residuals come from the saved CatBoost training-partition assessment, with original validation-based stopping. The test labels nevertheless choose which test cars survive; full-cohort OutlierTree also uses all labels. Filtered validation/test scores are therefore optimistic.

## XGBoost recipe

Normal survival:aft, scale 1, exact positive dollar bounds, native categorical features. Maximum 2,500 rounds, eta .03, max_depth 8, min_child_weight 20, lambda 20, alpha .1, row/column subsampling .8, max_bin 256, seed 42, four CPU threads. Validation AFT negative log-likelihood controls early stopping with patience 150. Regularization and subsampling are intended to reduce variance; actual robustness is not guaranteed by the recipe. There is no verified censoring interval and no invented censoring. No transformer text feature is included: validated OOF text-feature caches are not available in the repository for this retraining.

Fit an assessment model on retained training cars, select rounds on retained validation cars, and evaluate filtered test, excluded test, and the entire original test. Original-test errors are a useful diagnostic but not a fresh unbiased benchmark: this test has already informed filter-rule development. A new external test is necessary for clean selection claims. After assessment is saved, refit separate deployment weights on retained train+validation+test using the selected round count. Deployment weights are never used to produce the earlier assessment metrics.

Both ask and bid models become the GUI defaults only when both ready manifests exist. The original models remain available. The GUI labels filtered versus original-test results and does not present filtered MAE as untouched-test accuracy. Auction prices retain the existing age/CPI transport interpretation; asking labels remain asking prices, not confirmed retail transactions.

```bash
python -m scripts.train_residual_trimmed_xgb \
 --ask training-cache/full-cohort.jsonl \
 --bid training-cache/auction-cohort.jsonl
```

Run `scripts.train_catboost_aft` and `scripts.audit_outliertree --pct-outliers 0.099` first. OOF fold caches allow resumption only with matching source and preparation fingerprints. Output manifests, schema, filter counts, assessments and weight parts are versioned in the repository after fitting. Per-row audit/prediction tables remain reconstructable; source hashes and output checksums identify the artifacts.

To bound native categorical model size and rare-trim variance, final XGBoost model-name pooling is learned only on retained training cars: retain names with at least 20 examples, capped at 1,000 names. Other and unseen names map to a shared rare-model token. No car or residual rank changes. Make and other categories remain native. Each manifest persists the mapping, reused in assessment, final refit and GUI prediction.
