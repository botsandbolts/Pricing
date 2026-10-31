# Deployed MiniLM two-price release — 31 October 2026

Training/deployment run: https://github.com/botsandbolts/Pricing/actions/runs/41225149395

## What is installed

Asking: one `predicted_text_price` feature, the arithmetic mean of two MiniLM/Ridge dollar-price predictions, plus vehicle facts in XGBoost normal AFT. The frozen 128-token int8 MiniLM encoder is internal to the text-price computation. Neither TF-IDF nor its 384 raw embedding dimensions enter XGBoost.

Every training row receives two group-held-out text-price predictions from two repeated two-fold partitions. Text-head fitting is nested inside each outer XGBoost fold. Two complementary training-only heads serve new and held-out adverts. This is not newly supervised backbone fine-tuning.

Auction: structured-only XGBoost log-median quantile regression. The auction source has no descriptions; MiniLM is not used there.

Both markets use fresh five-fold log residuals, a single 25% training trim, new regular fits and matching separate Gaussian make/model/age Bayesian residual corrections. Validation/test are not trimmed by the outlier rule. No joint-market likelihood, t/mixture LOO comparison or verified censoring bounds are implemented.

## Untouched eligible historical test results

| Market | Test cars | Regular MAE USD | Bayesian MAE USD | Historical 90% range coverage |
|---|---:|---:|---:|---:|
| Asking | 31682 | 3706.42 | 3638.46 | 89.55% |
| Auction | 83603 | 1920.38 | 1892.45 | 92.30% |

Previous TF-IDF-serving historical MAE: ask regular $3,633.96 / Bayesian $3,527.01; auction regular $1,975.73 / Bayesian $2,035.16. This new asking release is worse on those test measures; auction errors are lower after the fresh pipeline rebuild. This is a pipeline comparison, not an isolated MiniLM ablation or proof that the trim improved accuracy.

Current-date prices use the separate age/mileage and monthly-index transport. These historical metrics and coverage do not validate today's car-level errors or intervals.

## Serving checks

Both-market real inference passed. Packaged serving files: 91,529,711 bytes (about 87.3 MiB). Smoke peak RSS: 398,608 KiB (about 389.3 MiB), below the 550 MiB gate. These are that process's measurements, not a total-memory guarantee for every concurrent workload. BART generation is on request; inference caches are released before generation and its weights afterward.

The GUI includes regular/Bayesian views, complete-function dollar SHAP, native log TreeSHAP visualizations, conditional sensitivity plots, direct MiniLM phrase analysis, the persistent value timeline and explicit sharing. Native SHAP excludes the separate date/Bayesian layers; dollar plots include the selected complete function. Sensitivity grids are not SHAP interaction values.
