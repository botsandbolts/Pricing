# Fixed-10% experiment: completed baseline, E5 stack pending

The 10% sample contains 42,688 raw listings; 31,001 remain eligible and deduplicated. Training/validation/test: 21,700 / 4,650 / 4,651, chronological partitions. All prices are historical 2021 asking prices.

**Completed:** 60 tabular-only LightGBM configurations, including Isolation Forest removal, dollar/log targets, RMSE/MAE/Huber and quantile losses. **Not completed:** the full fine-tuned E5 text-price stack; no CUDA GPU was available. Do not describe the following scores as text-model scores.

The best point model within each removal rate was selected using validation USD MAE. All five used log-price targets and Huber loss:

| Removed | Training rows retained | Validation MAE | Test MAE | Test RMSE |
|---:|---:|---:|---:|---:|
| 0% | 21,700 | $3,834 | $3,791 | $6,980 |
| 5% | 20,615 | $3,853 | $3,800 | $7,024 |
| 10% | 19,530 | $3,859 | $3,803 | $7,008 |
| 25% | 16,275 | $3,933 | $3,831 | $7,088 |
| 50% | 10,850 | $4,216 | $4,056 | $7,420 |

The validation-MAE champion was `remove00_log1p_huber`; median test absolute percentage error 17.01%, R² 0.768. The validation-RMSE champion was `remove00_dollars_rmse`, with test MAE $3,937 and RMSE $6,894. The two metrics favor different targets/losses. No filtering performed best by either predeclared validation criterion in this baseline. This is one dataset/split/seed, not a universal result or a significance test.

At 0% removal, the uncalibrated log-target q10–q90 interval covered 70.18% of test prices, below the nominal 80%; mean signed width $10,050 and crossing fraction 0.043%. Do not present it as a calibrated 80% prediction interval. The separate validation-residual band for the MAE champion covered 78.33% of test prices.

Full results are in `tabular-baseline.json` and `tabular-baseline.csv`. Selection used validation only; the full test grid is diagnostic. Models were fitted with at most 1,000 boosting rounds and validation early stopping. All listed scores come from the native LightGBM models; no hand-coded price rules were used.

The exact requested E5-base checkpoint was independently checked on two synthetic examples and one regression optimizer step: its first attention-query weight matrix changed (max absolute delta 0.0000200868), and predictions matched after saving and reloading. This is a software smoke test, not a pricing-quality evaluation. See `e5-gradient-smoke.json`.

To run the full requested stack, use `train_gpu.ipynb` or the two commands in README: first train cross-fitted E5 text regressors and cache text prices, then execute `pricing.experiments`. The resulting full-stack report must be compared against this baseline before claiming an improvement.
