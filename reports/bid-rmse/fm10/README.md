# USD RMSE-first bid re-analysis: fm10

Original frozen VIN-disjoint cohort: 384,198 training sales,
288,148 Lemon training sales, 82,380 validation, 83,603 test.
Re-ranked existing weights using original validation USD RMSE.
Weights were previously trained using Huber on LOG USD,
**not newly retrained against dollar squared error**.

Standalone validation RMSE winner: fm32_full
Standalone original-test USD RMSE: 3569.25
Blend validation RMSE winner: fm32_full (weight 0.75)
Blend original-test USD RMSE: 3348.93
Paired Lemon XGBoost original-test USD RMSE: 3887.69
RMSE was NEVER computed on test to select a model.
Model MAE, p95, p99, p99.9, worst 1pct and catastrophic dollar
error rates are also reported. No automatic production promotion.
