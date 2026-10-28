# A release-aware clock for the same car

The original asking model learns **2021 Craigslist advertised prices**. The auction model learns **2014–2015 reported wholesale sales**. They are different markets and different calendar periods. The new **Value over time** panel adds experimental present-date scenarios to the existing MiniLM + CatBoost AFT Ask and CatBoost quantile-0.5 Bid. Their trained weights and their historical evaluation records are unchanged.

## What changed

The earlier companion fixed its valuation date at May 2021. Its separate structural model already used vehicle age and a lagged CPI anchor, but tree leaves can give flat age responses near/outside their learned support. A calendar offset such as `-11` does not correct this: **vehicle age, calendar market level and vehicle generation are distinct**. Model-year age is `valuation year + fractional year − model year`, always nonnegative.

The new layer separates:

1. **Vehicle identity and historical price level:** existing saved text/tree estimators, with the actual make/model and vehicle facts.
2. **Depreciation:** separately fitted monotone additive age and log-mileage shapes, with shrunk manufacturer/model/body age slopes and categorical price effects. Make/model remain categorical labels; no new transformer is trained for them.
3. **Market time:** a separately backtested monthly BLS/FRED used-car index and a short-horizon forecast for an unpublished month.
4. **Generation and support:** explicit guards and visible fallback flags, rather than extrapolating negative ages or inventing newer generations.

The timeline follows one car. It can hold mileage constant or reconstruct/project mileage from a user-selected annual driving assumption. All plotted car values are model-derived, including past dates. Solid historical portions are reconstructed/transported scenarios. Future portions are dashed. The graph marks today, supports hover inspection and exports its displayed rows.

## Price equations and anchors

For car facts `x`, age `a(t)`, mileage `m` and fitted depreciation log function `d`:

```
Ask(t) = saved_Ask_2021(x,m)
         * exp(d_ask(a(t),m,x) - d_ask(a(2021-05-01),m,x))
         * retail_index(t) / retail_index(2021-05)

Bid(t) = saved_Bid_reference_dollars(x,m,age=5)
         * exp(d_bid(a(t),m,x) - d_bid(5,m,x))
         * retail_index(t) / retail_index(2015-01)
```

The Ask anchor is the existing model's historical dataset market, represented by May 2021; the underlying listings span a short window, not one exact transaction date. This anchoring convention is an assumption. It does not provide paired modern retail-sale calibration.

The Bid estimator's saved target is January-2015 reference dollars. Its predictors contain age, not raw calendar year. We query an age-five comparable in those saved units, then apply a smooth age ratio and a market-level ratio. The same actual category labels are retained. A synthetic age-five date supplies the age feature only; it does not create an observed sale of that generation.

The Ask description signal is held at the submitted description's historical-model output. The timeline computes that signal once, then recomputes vehicle-price anchors when assumed mileage changes. The resulting contemporary scenarios are not established conditional medians or unbiased dollar means in the target market. No smearing correction is silently applied.

## Depreciation experiments

Three curve families are compared: log-linear, monotone piecewise linear and integrated monotone splines. Piecewise/spline families include pooled manufacturer/model/body age terms. Sparse one-hot categorical effects cover make/model, body, transmission, location, fuel and condition. Rare labels pool into a fallback; unseen labels remain usable and are flagged. Integrated B-spline or positive segment bases with nonpositive age/mileage coefficients enforce nonincreasing age/mileage effects within this ordinary-car scenario. Collector appreciation is outside its scope.

Fitting uses four Huber IRLS steps with penalized least squares and bounded optimization. Category and category-age effects shrink toward shared structure. This resembles penalized MAP/random-effects estimation; **it is not MCMC, a full hierarchical Bayesian posterior or causal identification**. It does not provide posterior uncertainty bands. Age and generation are confounded in cross-sectional snapshots, even with these constraints.

Only ages **0–30 years**, finite mileage up to 2,000,000 and positive eligible prices enter the depreciation experiment. The original raw eligibility/deduplication procedures are preserved. No residual-quartile filtering selects the new component's tests. Existing CatBoost residual masks are not recomputed. New component counts and metrics therefore are not directly comparable with the old base-model cohorts.

## Cross-validation and backtesting

- **Vehicle curves:** three forward-date folds inside original training only, with whole-date boundaries. Average CV absolute log error selects the curve family. The original validation and test phases are scored separately after selection; they do not choose the family. Deployment curves then refit all in-scope source rows. A separate age-tail stress fit trains on age ≤12 and scores age 12–18 within original training. This is cross-sectional extrapolation stress, not observed longitudinal depreciation.
- **Market clock:** last-value, seasonal-naive, damped Holt and AR(1,12) forecasts are evaluated with rolling origins and 1/3/6/12-month horizons. Only origins before 2023 select the method, using 1/3-month absolute log index error. Later origins are a locked diagnostic test. Every forecast fits a prefix of the series. The current-vintage history is not an ALFRED real-time-vintage replay. Empirical forecast sensitivity is for the index, not the entire car price.
- **What remains untested:** the complete 2015→2026 or 2021→2026 per-car transport. Neither short source dataset supplies current outcomes for those cars. The source component tests, historical base-model metrics and market-index tests must not be combined or relabelled as a measured 2026 price MAE.

Detailed measurements, original date ranges, optimizer status, source hashes, candidates and cohort counts are in `models/time-transport/*-evaluation.json` and `market-backtest*.{json,csv}`. No synthetic test scores are presented as market evidence.

## Market data, releases and refreshing

The live adapter retrieves public **CUUR0000SETA02**, not seasonally adjusted, from FRED/BLS, verifies monthly continuity and positive values, and reads the source's release timestamp. The UI caches refresh attempts for six hours and retains an audited dated snapshot on failure. It names the last observed month, release date and forecast horizon. Today follows the Europe/Berlin calendar on reruns and while the active panel refreshes hourly.

When replaying an earlier information cutoff before the saved latest release, the market-clock API uses a conservative two-month lag. Historical timeline reconstructions instead use information available now, with that hindsight stated. Reference historical index values are known today, not claimed to have been published at the beginning of the reference month. A forecast never consumes observations after its information cutoff. Forecasts beyond six months from the last available observation fail visibly, rather than carrying stale prices forward forever.

The BLS index measures a constant-quality retail used-vehicle market. Depreciation is separate from that market-level movement, avoiding a simple inflation-only adjustment. Its sample and methodology are not a bespoke price index for every vehicle segment. Monthly data do not become a daily observed price feed: daily differences arise from age and stated scenario assumptions.

**Wholesale improvement researched:** Manheim MUVVI is a better-matched wholesale signal than retail CPI. Its official September 2026 report and downloadable historical workbook were located. The workbook endpoint returned an unavailable HTML response in this environment, so a complete history was not obtained. We do not splice one September figure into a different index or claim a Manheim backtest. Current Bid transport therefore explicitly labels retail CPI as a wholesale proxy. A vetted, consistently based historical wholesale series is the next market-data upgrade. Mid-month checkpoints must not be treated as completed monthly observations.

## Domain guards

- Model years after **2021** are rejected by current-value scenarios. The historical form also caps entry at 2021.
- Bid training contains generations through **2015**. Model years 2016–2021 require an explicit generation-analogue checkbox; otherwise a current Bid is unavailable. Manufacturer/body fallback is labelled, not presented as generation-specific knowledge.
- Ages outside 0–30 and invalid mileage fail visibly. A future clock cannot turn a nonexistent car into a valid historical observation.
- An unsold population is still missing. This layer does not identify selection correction, infer retail transactions from asking prices or revive the disabled IMR.
- No validated whole-price prediction interval is drawn. Aggregate index forecasting error omits depreciation, generation, channel, condition and base-estimator uncertainty.

## Research: relevant projects and why they matter

| Project / source | Useful idea | Decision here |
|---|---|---|
| [used-car-temporal-ml](https://github.com/masoudmaleki/used-car-temporal-ml) | Repeated cross-sections, external deflation and later-period testing for Turkish listings | Relevant evaluation design. Turkish TRY data do not calibrate US prices; repository describes an under-review paper. |
| [pyGAM](https://github.com/dswah/pyGAM) | Smooth terms, categorical factors and monotonic constraints | Inspired the additive curve candidate. Small explicit SciPy bases keep the deployed curve auditable and lightweight. |
| [linear-tree](https://github.com/cerlymarco/linear-tree) and [LightGBM linear trees](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#linear_tree) | Linear leaves can extrapolate where constant leaves flatten | Researched alternative. Unconstrained leaf extrapolation and lack of monotonic guarantees for leaf fits make it unsuitable as an automatic long-horizon solution. |
| [StatsForecast](https://github.com/Nixtla/statsforecast) | Statistical forecasting models and rolling-origin comparison | Relevant toolkit. The small global index experiment uses a SciPy implementation of damped Holt and explicit baselines; a neural network/GPU is unnecessary for this series size. |
| [PyMC examples](https://github.com/pymc-devs/pymc-examples) | Partial pooling and hierarchical inference | Useful future posterior modelling. Bayesian priors cannot create missing modern transaction labels or identify unrestricted age/period/cohort effects. |
| [RWI-GEO-CARMKT](https://github.com/Forschungsdatenzentrum-Ruhr/RWI-GEO-AS24) | Multi-year European listing panels | Better longitudinal evidence, but access requires an agreement and research/noncommercial restrictions; it was not silently ingested into a public US app. |

An LSTM can model a well-populated sequence; these data do not contain long sequences of the same comparable cars. More network capacity cannot recover an unobserved US make/model-by-month panel. The measured simple baselines are preferable until a richer longitudinal source is available.

Primary methodology/data references:
- BLS used vehicles: https://www.bls.gov/cpi/factsheets/used-cars-and-trucks.htm
- FRED series and release metadata: https://fred.stlouisfed.org/series/CUUR0000SETA02
- BLS age-depreciation research: https://www.bls.gov/opub/mlr/2024/article/a-consumption-measure-for-automobiles.htm
- Manheim September 2026: https://www.coxautoinc.com/insights/manheim-used-vehicle-value-index-september-2026-trends/
- Manheim methodology: https://site.manheim.com/wp-content/uploads/sites/2/2024/02/Used-Vehicle-Summary-Methodology.pdf

## Reproduce

```
python -m scripts.prepare_market_clock
python -m scripts.train_time_transport --ask training-cache/ask-cohort.jsonl --bid training-cache/auction-cohort.jsonl
python -m pytest tests/test_time_transport.py tests/test_bid_quantile.py tests/test_catboost_serving.py -q
python -m streamlit run app.py
```

Preparation uses the project's existing `scripts.prepare_full` and `carbids.data` canonicalizers. Training ran on CPU. No Kaggle GPU was started. The compact JSON curves, index snapshot, measured evaluations and complete index forecast rows are version-controlled alongside the implementation.

## Measured run

| Component | Linear CV log-MAE | Piecewise CV log-MAE | Spline CV log-MAE | Selected |
|---|---:|---:|---:|---|
| Ask | 0.32723 | 0.29412 | 0.29421 | piecewise |
| Bid | 0.22379 | 0.18350 | 0.18324 | spline |

The Ask curve refit uses 198,995 in-scope listings; the Bid curve uses 549,995 in-scope sales. Their held source-test MAEs are $4,323.83 and $2,314.74 on 29,893 and 83,616 cars respectively. Those are structural-component level predictions, not the complete anchored time-transport model or a 2026 price test. The independently retained original base models serve the actual anchors.

Selected market model: damped Holt. Locked later-origin index MAPE is 0.80% / 3.05% / 6.47% / 7.51% at 1 / 3 / 6 / 12 months (11 forecasts per horizon). Selection used earlier origins only. The current GUI limits forecasts to six months even though 12-month diagnostics are reported. These percentages do not measure individual-car pricing accuracy.

Real-model Streamlit testing passed for unchanged historical Ask output ($10,608.983456680162 on the fixed control car), quantile Bid default, two current-value receipts, a 298-row two-market timeline, annual-mileage scenarios and preserved explanation controls. Nine focused tests cover monotonic curves, categorical pooling/serialization, publication cutoffs, horizon limits, missing inputs, model-year guards and existing serving.
