# Bid-only log-median CatBoost default

Cleaner: CatBoost Quantile alpha=0.5 on log reference-dollar prices. Three-fold
OOF training residuals and training-only validation/test predictions remove
exactly ceil(25%) within each original phase. No old OutlierTree flags are added.

Retained: 288,148 train + 61,785 validation + 62,702 test = **412,635**.
The final model has **1,449 trees**, the same quantile-0.5/log-price objective,
native categorical features, and **no IMR feature**.

IMR is implemented only for the optional cleaner; it is currently disabled
because the source contains sold cars without a full sold/unsold listing
population. No non-sales or lambdas are fabricated.

Filtered test MAE: **$1,344.86** (selected easy-car cohort). Original unfiltered
test MAE: **$2,148.14**, RMSE **$4,454.94** on 83,603 cars. Previous AFT original-
test MAE was **$2,090.20**. This is the user-requested default, not an accuracy
improvement. Filtered validation/test scores are optimistic; the original test
has also been reused for model/filter development. A new external test is needed.

Asking MiniLM + AFT files remain unchanged. The fixed Corolla example still
predicts $11,120.23057881292. See ask-preservation-check.json and gui-smoke.json.

[Methods, selection assumptions and reproduction](../../BID_QUANTILE.md).
