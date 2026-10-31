# Original-data random forest and hierarchical laboratory

The **Hierarchical model** panel loads real original-cohort results automatically. It compares **XGBoost random forest** with **the same forest plus an empirical-Bayes residual correction**. SGD has been removed. CatBoost is not used in this experiment. The normal landing-page calculator, car-details fields, long description dropdown and MiniLM text features keep their existing production models.

## Data and historical assessment

Sources: Craigslist Used Cars / Tom, [Zenodo 10457828](https://zenodo.org/records/10457828), CC-BY 4.0; [Used Car Auction Prices / tunguz](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), CC0. Asking advertisements are not confirmed retail sales. Auction prices are platform-reported, not independently settlement-verified.

All source records are scanned and 100% of eligible canonical cars are assessed. No 10% sampling or residual trimming. Existing canonical preparation retains dated, deduplicated asking listings priced $1,000–150,000 with cleaned descriptions >=80 characters; auctions require valid VIN/price provenance and one canonical sale per vehicle. Predictor checks require valid year/date/mileage; missing mileage remains eligible for forest training. Known auction VIN overlap with asking descriptions is excluded.

| Market | Eligible cars | Test cars | Forest test MAE | Forest + hierarchy test MAE |
|---|---:|---:|---:|---:|
| Craigslist asking | 211,127 | 31,682 | $4,722 | $4,065 |
| Wholesale auction | 550,181 | 83,603 | $2,240 | $2,127 |

Historical price bases differ: 2021 advertised asking dollars versus January 2015 CPI-reference auction dollars. Do not directly rank these two MAEs as identical tasks or current-car guarantees. Auction normalization uses the project's two-month-lag index recipe; current-vintage revisions remain a limitation.

Chronological calibration/validation/test cohorts are preserved. Calibration predictions use three vehicle-group OOF folds, not rolling-origin folds. Category vocabularies are learned inside each training fold. Validation/test predictions come from a calibration-only fit; hierarchy parameters come only from OOF calibration errors. No held-out labels fit a model or select parameters. Every eligible test/validation row remains in the scores. The browser displays at most 1,000 reproducibly selected test examples, clearly distinguished from the complete scoring cohort.

## Forest and hierarchy recipe

The forest fits natural-log price with squared error: 200 parallel trees in **one boosting round**, learning rate 1, depth 8, minimum child weight 20, lambda 10, row and per-node column subsampling .8, histogram method, 128 bins, four threads, seed 42. Native categorical vocabularies handle vehicle make/model/body/transmission/state/fuel/drive/title/cylinders/color where present; numerical facts include age, log mileage and auction grade. Unseen category levels become missing. There is no seller identity, MMR, target-price leakage or text encoder. Predictions are exponentiated and clipped to $1–10 million; no claim to arithmetic expected sale price or automatic Duan smearing.

The release used `XGBRFRegressor`, internally one boosting round with 200 parallel trees. XGBoost 3.2 deprecates that convenience wrapper. Reproduction uses the equivalent supported `XGBRegressor(n_estimators=1, num_parallel_tree=200)`; a test verifies identical predictions with matching parameters. This is XGBoost's RF-style algorithm, not sklearn's conventional bagged forest. No early stopping or test-driven tuning.

`log(corrected USD) = log(forest USD) + global + make + model-within-make + age effect`.

Sequential empirical-Bayes pooling fits `log(observed/base)` on calibration OOF errors. Estimated between-group variance is group-mean variance minus average sampling variance, floored at zero. Group offsets shrink by `n*tau²/(sigma²+n*tau²)`. Unseen groups get zero group offset; sparse groups borrow strength. A regularized shared age slope is supplemented by model-age slopes with >=40 calibration rows and >=3 years' age variation. This is a sequential Gaussian approximation, not joint Bayesian MCMC. Correction standard errors omit hyperparameter uncertainty and are not sale-price intervals.

The scorecard includes MAE, RMSE, log MAE and signed bias. A 300-resample paired vehicle-group bootstrap reports a 95% interval for the MAE difference, not for a car's price. Both historical tests improved MAE/RMSE after pooling; auction signed bias increased. All measures and limitations remain visible. No automatic production promotion occurs.

## Today or another date

Shared valuation controls now default to **Today**, recomputed in Europe/Berlin on every rerun. A custom date and month-end scenario remain selectable. In the forest panel, select a historical example and click **Compare hierarchical estimate** to see today's date-adjusted scenarios and the evidence waterfall.

`selected-date price = historical forest price × fitted age/mileage ratio × monthly market-index ratio`.

The existing monotone depreciation curves pool make/model/body effects. Original year, source date and recorded mileage are matched from the source archive using the stored row identity and predictor checks; archive SHA-256 must match training provenance. This is metadata enrichment, not retraining or modification of historical scores. The default holds recorded mileage constant; visitors can enter an assumed mileage at the selected date. Missing recorded mileage, unsupported generations, and ages outside the curve's 0–30-year domain are declined explicitly.

Asking anchors use their source-month index. Auction anchors already use January 2015 reference dollars, so that reference is applied once. The release-aware BLS/FRED used-car index uses observed monthly values when available, and the component-backtest-selected **damped Holt** method beyond the observation cutoff, capped at six forecast months. The page shows the last observed month, forecast method/horizon, ageing/mileage multiplier and market multiplier. Live refresh is attempted; the dated snapshot is a fallback. This is not a daily transaction feed.

Historical MAE does not validate current-car transport. Component backtests do not establish a full current-price MAE or individual prediction interval. Cross-sectional depreciation is not identified causal ageing, and retail CPI is a proxy for auction-market movement. Historical test scores remain separate from selected-date scenarios.

## Reproduction and deployment scope

```
python -m scripts.prepare_full training-cache/ask.zip --output training-cache/full-cohort.jsonl --workers 4
python -m carbids.data --input training-cache/bid.zip --output training-cache/bid-cohort.jsonl
python -m scripts.train_original_residual_comparison --engine forest --output training-cache/forest-assessment-fit
python -m scripts.stage_forest_results
python -m scripts.enrich_forest_examples
```

Download original archives to the shown cache paths. Per-market provenance records source/canonical hashes, counts, exclusions and dates. No descriptions or raw VINs enter released examples; identities are hashed. `models/forest-original/{ask,bid}` contains evaluation, hierarchy weights, provenance and small display examples. Large base forests and full RF residual tables are offline reproduction outputs, with hashes/commands in `release-scope.json`. The public panel displays held-out predictions and transported scenarios, not fresh forest inference for arbitrary cars. The main calculator serves ordinary car inputs and descriptions.

Optional uploaded experiments remain session-local, capped at 10,000 rows. Consent-gated Turso `hierarchical_estimations` retains explicit historical comparisons for 90 days: visitor/session, UTC timestamp, city/country, market, base/corrected prices, correction/support statistics and model fingerprint. Descriptions/raw inputs are absent. Date scenarios are not misrecorded as historical predictions. GA receives bounded panel/action names. No private database records appear in the GUI.
