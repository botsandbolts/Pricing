# Original-data random forest and hierarchical laboratory

The **Hierarchical model** panel loads original-cohort results automatically. Its live editor combines **XGBoost random forest** with a **joint Bayesian residual correction** and shows validation-calibrated historical outcome ranges. The previous sequential empirical-Bayes correction remains as a benchmark. SGD has been removed. CatBoost is not used in this experiment. The normal landing-page calculator and MiniLM text features keep their existing production models.

## Data and historical assessment

Sources: Craigslist Used Cars / Tom, [Zenodo 10457828](https://zenodo.org/records/10457828), CC-BY 4.0; [Used Car Auction Prices / tunguz](https://www.kaggle.com/datasets/tunguz/used-car-auction-prices), CC0. Asking advertisements are not confirmed retail sales. Auction prices are platform-reported, not independently settlement-verified.

All source records are scanned and 100% of eligible canonical cars are assessed. No 10% sampling or residual trimming. Existing canonical preparation retains dated, deduplicated asking listings priced $1,000–150,000 with cleaned descriptions >=80 characters; auctions require valid VIN/price provenance and one canonical sale per vehicle. Predictor checks require valid year/date/mileage; missing mileage remains eligible for forest training. Known auction VIN overlap with asking descriptions is excluded.

| Market | Eligible cars | Test cars | Forest test MAE | Empirical-Bayes test MAE | Joint-Bayes test MAE | Historical 90% range test coverage |
|---|---:|---:|---:|---:|---:|---:|
| Craigslist asking | 211,127 | 31,682 | $4,722 | $4,065 | $4,085 | 89.9% |
| Wholesale auction | 550,181 | 83,603 | $2,240 | $2,127 | $2,116 | 92.4% |

Historical price bases differ: 2021 advertised asking dollars versus January 2015 CPI-reference auction dollars. Do not directly rank these two MAEs as identical tasks or current-car guarantees. Auction normalization uses the project's two-month-lag index recipe; current-vintage revisions remain a limitation.

Chronological calibration/validation/test cohorts are preserved. Calibration predictions use three vehicle-group OOF folds, not rolling-origin folds. Category vocabularies are learned inside each training fold. Validation/test predictions come from a calibration-only fit; hierarchy parameters come only from OOF calibration errors. No held-out labels fit a model or select parameters. Every eligible test/validation row remains in the scores. The browser displays at most 1,000 reproducibly selected test examples, clearly distinguished from the complete scoring cohort.

## Forest and hierarchy recipe

The forest fits natural-log price with squared error: 200 parallel trees in **one boosting round**, learning rate 1, depth 8, minimum child weight 20, lambda 10, row and per-node column subsampling .8, histogram method, 128 bins, four threads, seed 42. Native categorical vocabularies handle vehicle make/model/body/transmission/state/fuel/drive/title/cylinders/color where present; numerical facts include age, log mileage and auction grade. Unseen category levels become missing. There is no seller identity, MMR, target-price leakage or text encoder. Predictions are exponentiated and clipped to $1–10 million; no claim to arithmetic expected sale price or automatic Duan smearing.

The release used `XGBRFRegressor`, internally one boosting round with 200 parallel trees. XGBoost 3.2 deprecates that convenience wrapper. Reproduction uses the equivalent supported `XGBRegressor(n_estimators=1, num_parallel_tree=200)`; a test verifies identical predictions with matching parameters. This is XGBoost's RF-style algorithm, not sklearn's conventional bagged forest. No early stopping or test-driven tuning.

`log(corrected USD) = log(forest USD) + global + make + model-within-make + age effect`.

The earlier sequential empirical-Bayes benchmark pools `log(observed/base)` on calibration OOF errors. Estimated between-group variance is group-mean variance minus average sampling variance, floored at zero. Group offsets shrink by `n*tau²/(sigma²+n*tau²)`. Unseen groups get zero group offset; sparse groups borrow strength. A regularized shared age slope is supplemented by model-age slopes with >=40 calibration rows and >=3 years' age variation. This is a sequential Gaussian approximation, not joint Bayesian MCMC. Its correction standard errors omit hyperparameter uncertainty and are not sale-price intervals. It remains on the scorecard; the live car editor uses the joint Bayesian correction below.

The scorecard includes MAE, RMSE, log MAE and signed bias. A 300-resample paired vehicle-group bootstrap reports a 95% interval for the MAE difference, not for a car's price. Both historical tests improved MAE/RMSE after pooling; auction signed bias increased. All measures and limitations remain visible. No automatic production promotion occurs.

### Joint Bayesian correction and historical outcome ranges

The installed car editor fits a **joint posterior**, separately for asking and auction prices. The forest remains a held-fixed calibration-only base model. Group-out-of-fold log-price errors obey:

`r_i = market + age_per_decade × (age_i − mean_age)/10 + make_i + model_within_make_i + observation_noise_i`.

Independent proper priors: market `Normal(0,1)`, age per decade `Normal(0,0.2)`, make and model effects `Normal(0,tau_make²)` and `Normal(0,tau_model²)`. Observation and group variances have inverse-gamma `(shape, scale)` priors `(2,0.1)`, `(2,0.04)` and `(2,0.04)` respectively. Two blocked Gibbs chains jointly sample effects and variances. The global mean and group effects are updated as a conditional block to improve mixing. The release stores 100 thinned posterior draws per chain from 400 sampled draws; diagnostics use all samples and report R-hat for market, age and variances. Sparse groups shrink to their parents. Unseen groups add population variation to uncertainty.

For an individual historical car, predictive log variance combines joint posterior variation and observation noise. A **separate chronological validation set** chooses the finite-sample 90% quantile of absolute standardized log errors to scale the historical posterior predictive interval. The **untouched chronological test set** reports coverage, median width, MAE, RMSE and bias. The center exponentiates the mean log correction, a median-like price rather than the arithmetic expected sale price. A prediction range concerns one outcome, not the uncertainty of the model mean. Validation is marginal across the historical cohort, with no coverage guarantee for every subgroup.

The same ageing and monthly market factors transport the point and historical range to the selected date. **There is no verified 90% coverage for today's prices**: market forecasts, inventory shifts, unobserved generations and age/mileage transport introduce uncertainty absent from this historical interval. The GUI says this explicitly and reports the actual untouched-test coverage. Craigslist asks are advertised prices, not completed retail sales.

Reproduce from the original archives and group-held-out forest residuals:

```
python -m scripts.train_original_residual_comparison --engine forest --output training-cache/bayesian-refit
python -m scripts.train_bayesian_hierarchy --input training-cache/bayesian-refit --warmup 450 --draws 400
```

The first command needs the canonical input files described below. Released posterior files are checksum-verified non-pickle NumPy arrays split into Git-sized chunks, with priors, diagnostics and test coverage in JSON. The new RF refit matched every price in all 1,000 previously published held-out inspection examples per market. No test outcome fitted the posterior or selected interval width.

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

Download original archives to the shown cache paths. Per-market provenance records source/canonical hashes, counts, exclusions and dates. No descriptions or raw VINs enter released examples; identities are hashed. `models/forest-original/{ask,bid}` contains evaluation, hierarchy weights, provenance and small display examples. The calibration-only base forests are now installed for arbitrary car-form inference as checksum-verified compact tree/bitset artifacts. Full residual tables remain offline reproduction outputs. Each serving refit must reproduce all 1,000 published held-out example predictions within 0.002% / $0.02 tolerance before packaging. The shared asking/auction form shows both prices for one selected date; older-generation auction analogues require explicit opt-in. The original MiniLM calculator remains available.

Optional uploaded experiments remain session-local, capped at 10,000 rows. Consent-gated Turso `hierarchical_estimations` retains explicit historical comparisons for 90 days: visitor/session, UTC timestamp, city/country, market, base/corrected prices, correction/support statistics and model fingerprint. Descriptions/raw inputs are absent. Date scenarios are not misrecorded as historical predictions. GA receives bounded panel/action names. No private database records appear in the GUI.

## Full two-market car form

The forest panel opens with the full vehicle editor: year, make/model, mileage, state, body, condition, fuel, transmission, drive, title, cylinders, paint and optional auction grade. The 400 description examples populate editable fields using a separate widget namespace, preserving the main calculator's state. Description wording is retained for convenience but is not a forest predictor; that limitation is shown next to the field.

One submit estimates both markets. Inputs and valuation dates must match the saved receipt or results are hidden as stale. Today follows Berlin's calendar; a custom date is available. Asking is anchored to May 2021 at the corresponding vehicle age; auction uses an age-five comparable in January 2015 reference dollars. The hierarchy corrects each historical forest anchor, then the existing depreciation and release-aware market clock transport it. Mileage is interpreted at the selected date, held constant between anchor and valuation unless the user changes it. These anchors are model scenarios, not observed sales of the entered car. No test labels are used for serving refits.

Reproduce installed weights with `python -m scripts.refit_forest_serving` after canonical preparation. Native categorical schemas and original calibration counts must match the saved assessment. Exact tree traversal with packed categorical bitsets avoids the native loader’s large temporary memory allocation. All 2,000 published examples match the original XGBoost prices within $0.000001. Checksum-verified parts and non-pickle NumPy arrays keep loading bounded. Run `python -m scripts.export_compact_forests` after refitting to produce the serving artifacts.
