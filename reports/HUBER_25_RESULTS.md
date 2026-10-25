# 25% sample: Huber LightGBM baseline

**Measured tabular-only results. The fine-tuned E5 stack has not been trained/evaluated.** This workspace has no CUDA GPU; the exact checkpoint passed a gradient/save-load smoke test only.

106,720 raw rows sampled from 426,880; 68,846 eligible/deduplicated. Chronological train / validation / test: 48,192 / 10,327 / 10,327. Historical April–May 2021 asking prices, not transaction prices.

30 configurations = five Isolation Forest removal rates × dollar/log targets × Huber thresholds 0.5, 0.9 and 1.5. Target standardization uses retained training rows only. The detector fits training predictors without the true price; validation/test are never filtered. Early stopping uses validation MAE in the fitted target scale, with at most 1,000 rounds.

Best validation-MAE configuration at each removal rate:

| Removed | Target | Huber threshold | Retained | Validation MAE | Test MAE | Test RMSE |
|---:|---|---:|---:|---:|---:|---:|
| 0% | log1p | 0.9 | 48,192 | $3,433 | $3,614 | $7,147 |
| 5% | log1p | 0.9 | 45,783 | $3,450 | $3,613 | $7,163 |
| 10% | log1p | 0.5 | 43,373 | $3,462 | $3,622 | $7,272 |
| 25% | log1p | 0.5 | 36,144 | $3,543 | $3,691 | $7,426 |
| 50% | log1p | 0.5 | 24,096 | $3,765 | $3,909 | $7,623 |

The validation-MAE winner is `remove00_log1p_huber_d0.9`: test MAE $3,614, RMSE $7,147, median absolute percentage error 15.95%, R² 0.771. Its validation-residual 80% band covers 78.26% of test prices.

The validation-RMSE winner is `remove00_dollars_huber_d1.5`: test MAE $3,681 and RMSE $6,995. Test results are diagnostics; winners were selected before test scoring. The 5% removal arm has a slightly lower test MAE ($3,613), but that does not override the validation choice. No significance claim is made.

The new sample and holdout differ from the prior 10% experiment, so their test errors are not a paired comparison. The full requested E5 stack must still be measured on this same 25% cohort.

Run `train_gpu.ipynb` on a CUDA runtime to fine-tune three out-of-fold E5 models plus the all-training-row inference model, then repeat this Huber grid with `predicted_text_price` added to the structured inputs. Filtering applies to LightGBM training rows; the shared E5 stage is not refitted per removal arm.

Machine-readable results: `tabular-huber-25.json` and `tabular-huber-25.csv`.
