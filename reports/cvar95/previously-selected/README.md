# Original untouched historical CVaR95 audit

The new metric is the mean absolute dollar error of the worst 5% of
original test predictions. CVaR95 is evaluation ONLY: weights unchanged.
All candidates here were selected using historical validation RMSE/MAE,
not test-based CVaR optimization. Model replacement requires a NEW
validation CVaR selection protocol; this table is descriptive.

## Wholesale

| Previously selected model | CVaR95 USD | VaR95 USD | MAE USD | RMSE USD |
|---|---:|---:|---:|---:|
| fm26 / validated_blend | USD 8,693.64 | USD 5,043.83 | USD 1,787.81 | USD 3,089.95 | 
| fm26 / standalone | USD 9,018.79 | USD 5,298.15 | USD 1,877.61 | USD 3,178.15 | 
| fm10 / validated_blend | USD 9,402.82 | USD 5,501.60 | USD 1,960.56 | USD 3,348.93 | 
| fm10 / standalone | USD 10,044.57 | USD 5,924.07 | USD 2,123.84 | USD 3,569.25 | 
| ft29 / standalone | USD 10,169.95 | USD 5,531.19 | USD 2,026.89 | USD 3,607.61 | 
| ft29 / validated_blend | USD 10,215.12 | USD 5,497.18 | USD 1,928.58 | USD 3,576.04 | 
| Lemon XGBoost historical baseline | USD 11,140.19 | USD 5,753.70 | USD 1,920.38 | USD 3,887.69 | 

## Asking

| Previously selected model | CVaR95 USD | VaR95 USD | MAE USD | RMSE USD |
|---|---:|---:|---:|---:|
| BGE-256 + full DeepCross/AFT | USD 19,985.05 | USD 10,140.17 | USD 3,036.75 | USD 6,299.01 | 
| MiniLM scalar + DeepCross/AFT, 12% dropout | USD 21,453.14 | USD 10,990.78 | USD 3,261.54 | USD 6,740.98 | 
| MiniLM scalar + DeepCross/AFT, 4-expert MoE | USD 21,721.84 | USD 11,581.49 | USD 3,368.51 | USD 6,779.54 | 
