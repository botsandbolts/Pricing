# Pricing release audit — 31 October 2026

This records the initial TF-IDF snapshot and subsequent repairs. For the current installed feature contract, consult `models/xgboost-hierarchy/release.json` and `ask/features.json`; for the two-price MiniLM replacement see `MINILM_RELEASE.md`. Earlier snapshot scores are not automatically the new release’s scores.

## Production contract at the initial audit snapshot

- Horizontal landing navigation; one main panel executes per rerun. Regular pricing and Bayesian pricing remain separate views. Model controls for old benchmarks are explicitly historical.
- Current bundle: `models/xgboost-hierarchy`. XGBoost ask normal AFT, exact dollar bounds; wholesale median quantile on log dollars. 300 rounds, depth 5. Sparse TF-IDF descriptions for ask (2,000 vocabulary entries); no UMAP, MiniLM or CatBoost in this current estimate path.
- Lemon Squeezer: fresh five-fold vehicle-group-held-out log residuals; one top-25% training trim. Validation/test are not trimmed by the outlier rule. Recorded scores refer to eligible model rows, not every raw scraped listing. Post-selection OOF diagnostics are not nested CV of selection.
- Separate make/model/age Gaussian Bayesian residual posteriors. Matched eligible historical test MAE: ask regular $3,633.96 vs Bayesian $3,527.01; bid regular $1,975.73 vs Bayesian $2,035.16. Auction worsens. This comparison does not isolate the benefit of cleaning.
- Default valuation date follows the calendar; custom dates remain possible. Dated monthly market observations plus bounded forecasting and age/mileage transport produce a scenario, not verified daily transaction data. Auction generations after 2015 require the displayed analogue assumption. Historical interval coverage is not current-date coverage.

## Repairs

1. Replaced competing sidebar/tab navigation with a single horizontal landing control. Import/share callbacks route through it. Inactive panels no longer construct calculators or run models.
2. Regular waterfall now uses regular anchors and totals, rather than Bayesian values under a regular label.
3. Current estimate publishes the correct selected estimator and date to sharing; PNG, QR and JSON tools use that receipt. Vehicle import populates the current car form.
4. New timeline uses the same installed model anchors and saved car as current pricing. It does not silently switch to an old estimator. Mileage assumptions are explicit.
5. Current-model native TreeSHAP drivers are available on request. Log contributions and their ordered dollar decomposition are labelled; dollar steps are order-dependent and not causal effects.
6. Separate on-demand 128-token int8 ONNX MiniLM/Ridge description-price attribution is restored with explicit historical/standalone labels. It is not an input to the current XGBoost model.
7. Source report and download cover both prices, dated market assumptions, current models and evaluation. BART overview is optional/on demand to protect CPU and memory. Exact model figures remain deterministic.
8. Replaced superseded five-stage default documentation with the current one-cut cross-validation method. Removed legacy default claims from the historical sidebar; blocked automatic CatBoost calculators in the model/transfer labs.
9. Added current estimator completion hooks and horizontal-control detection to opt-in GA instrumentation. Events report bounded action IDs, never entered descriptions or vehicle values. Actual GA receipt must be checked in the owner's GA property; local tests verify client behavior, not Google's delivery.

## Verification

- Local Streamlit AppTest: initial page, regular estimate, all nine main panels, Bayesian route, current TreeSHAP graph, time graph, share callback and PNG/JSON controls.
- Default regression example: 2015 Toyota Corolla, 95,000 miles, CA, blank description, 31 October 2026: regular ask $7,577 / auction $3,830; Bayesian ask $7,558 / auction $3,834 using the dated snapshot. Live index changes can change results.
- Regression tests added in `tests/test_release_navigation.py`.
- Existing tests: consent, Google library gating/withdrawal, native-share fallbacks; summary orchestration/source reports, market-date transport, OCR, advert parsing and completion deduplication.

## Earlier requests that are NOT completed

- MiniLM replacement throughout the current XGBoost cleaning/refit/hierarchy path. Verified Kaggle fold outputs do not by themselves constitute that retrained release. Current serving is TF-IDF, and says so.
- Joint ask/bid hierarchical likelihood; Student-t and mixture alternatives; LOO ELPD comparison; actual Tobit/interval-censoring bounds. Asking prices alone do not supply those bounds. Current hierarchy is Gaussian with separate markets.
- An independent current-market dataset validating today's transported prices or current-date interval coverage.
- Demonstrated improvement from the 25% trimming against an otherwise matched untrimmed model.
- The full six-view explanation suite is now wired to the current estimator: complete-function grouped dollar SHAP, native log-TreeSHAP beeswarm/dependence/decision paths, selected-date mileage sensitivity and a year/mileage price surface. The latter two are sensitivity plots, not SHAP interaction values. The gallery is at most 16 supported fixture cars; text and auction grade stay fixed. Dollar SHAP uses at most three background cars and four sampled orders. Old benchmark tools retain their identities.
- BART source/orchestration tests passed; actual hosted BART generation and separate ONNX MiniLM phrase highlights were also exercised successfully on 31 October. Runtime latency and factual accuracy are not guaranteed.

No zero-bug claim is made. This audit records reproducible checks and explicit gaps rather than treating old promises as proof of implementation.

## Timeline follow-up
The timeline now builds on entry, retains its graph across reruns and regenerates when assumptions change. Depreciation calculations are vectorized over months. Pre-update saved receipts recover vehicle fields from their saved signature. A regression checks exact agreement at the saved valuation date for both markets and both regular/Bayesian modes.
