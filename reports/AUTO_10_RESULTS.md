# 10% sample, Isolation Forest auto: measured tabular baseline

**Tabular-only baseline. The full fine-tuned E5 stack is still pending GPU training.**

42,688 raw rows sampled; 31,001 eligible/deduplicated: 21,700 train, 4,650 validation, 4,651 test. Isolation Forest uses contamination="auto", fits training predictors only and retains decision_function >= 0. No percentage is forced. Validation/test are untouched.

| Configuration | Removed training rows | Validation MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| isolation_auto_dollars_huber_d0.5 | 0 | $3,905 | $3,808 | $7,000 |
| isolation_auto_dollars_huber_d0.9 | 0 | $3,896 | $3,832 | $6,869 |
| isolation_auto_dollars_huber_d1.5 | 0 | $3,918 | $3,865 | $6,869 |
| isolation_auto_log1p_huber_d0.5 | 0 | $3,820 | $3,749 | $7,023 |
| isolation_auto_log1p_huber_d0.9 | 0 | $3,834 | $3,791 | $6,980 |
| isolation_auto_log1p_huber_d1.5 | 0 | $3,851 | $3,830 | $7,039 |

Auto mode removed **0 of 21,700 training rows (0.00%)**, using offset -0.5. Automatic mode may retain every row; this is the fitted detector's result, not forced retention.

Selected on validation USD MAE: `isolation_auto_log1p_huber_d0.5`. Selected on validation USD RMSE: `isolation_auto_dollars_huber_d1.5`. Test scores are diagnostic and do not change selection.

Source prices are historical 2021 asking prices. Huber thresholds are applied to targets standardized using retained training rows only. The detector uses imputed numerical and one-hot categorical features; this axis-aligned representation and auto threshold need not identify every problematic listing. The full E5-stack detector will also see cross-fitted predicted_text_price, so its auto removal count may differ.

The GPU notebook trains the requested embaas E5-base checkpoint directly on text/log-price regression, creates out-of-fold text-price scalars and repeats these six LightGBM experiments. No full-stack pricing scores are available yet.
