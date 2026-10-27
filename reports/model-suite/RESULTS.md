# Completed model assessments

| Market | Model | Test rows | MAE USD | RMSE USD |
|---|---|---:|---:|---:|
| ask | lgb_huber | 31,682 | 3,097.29 | 6,517.16 |
| ask | lgb_quantile_50 | 31,682 | 3,165.50 | 6,729.40 |
| ask | xgb_aft_normal | 31,682 | 3,169.37 | 6,483.50 |
| ask | ridge_baseline | 31,682 | 4,981.42 | 11,376.72 |
| bid | lgb_huber | 83,603 | 1,868.38 | 3,701.38 |
| bid | lgb_quantile_50 | 83,603 | 1,889.29 | 3,888.66 |
| bid | xgb_aft_normal | 83,603 | 1,854.72 | 3,181.75 |
| bid | ridge_baseline | 83,603 | 2,765.26 | 4,467.06 |

The compatible default MiniLM + Cleanlab Huber studio model: MAE $3,063.77 on 31,777 test rows. Its raw-year tabular schema keeps the complete original cohort; the age-based laboratory requires valid age and therefore uses 31,682 asking test rows. These two MAEs are not on identical cohorts.

MiniLM is fine-tuned on descriptions, with separate train-only assessment and full-data deployment text heads. No auction text is available. Full deployment refits occurred after assessment; their training predictions never replace held-out metrics.

Cleanlab automatically chose zero exclusions in both the assessment and deployment audits for these new runs. All eligible fitting rows were retained; no arbitrary fraction was removed. Its label-quality scores and audit settings are still recorded.

The three new models per side use ordinary default-sized boosters. The older tuned ensembles are separate historical choices, not entries in these new-model correlation matrices.

## Same-car Spearman correlations

| Pair | Spearman rho |
|---|---:|
| ask_lgb_huber / ask_lgb_quantile_50 | 0.996596 |
| ask_lgb_huber / ask_xgb_aft_normal | 0.988394 |
| ask_lgb_huber / bid_lgb_huber | 0.764568 |
| ask_xgb_aft_normal / bid_xgb_aft_normal | 0.750087 |

See the committed comparison JSON and GUI for the complete 8×8 matrix and all 56 directed Ridge fits. These are predictions on the same asking cars; they are not paired observed asking/transaction prices.

## Ridge example

Asking Huber = $5,939.46 + 1.344497 × transported auction Huber. This fit used asking validation cars. On test cars, raw auction-minus-asking prediction bias is $-8,538.06; the calibrated mapping has MAE $5,325.62 and bias $-353.40 relative to the asking model. This maps two model outputs, not observed retail sales.

AFT used exact dollar bounds because measured censoring bounds are unavailable. It is an uncensored normal AFT on log price, not an identified latent-sale Tobit.
