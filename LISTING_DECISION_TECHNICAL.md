# Listing decision tools — technical guide

## Three independent SVDs

Ask uses asking vehicles and optional MiniLM text similarity. Bid uses auction vehicles and an imputed MiniLM description-price proxy. Hybrid uses both markets' vehicle attributes only, including a separate price/text-free imputer. Ask and Bid each return five neighbours; Hybrid returns five of each, mixed and sorted by the same cosine similarity. Ask rows are purple and auction rows blue.

Adjusted advert/auction prices assist Ask/Bid missing-feature imputation, then are dropped before SVD. The only price-derived scalar SVD feature is the auction description-price proxy. Actual auction descriptions do not exist and are not invented. Detailed neighbour records show source and adjusted prices, year/mileage/specification differences and same-market price differences from the queried car. No FM or LambdaRank is used. See COMPARABLE_INSPECTOR.md for the complete feature and missing-data contract.

## Architecture and isolation

`listing_inspector.py` is a separate horizontal workspace. `market_review.py` contains bounded CSV parsing, duplicate matching, review rules and sequential batch orchestration. `listing_insights_ui.py` presents the tools. Offline research lives in `listing_statistics.py` and `scripts/build_listing_insights.py`. Production estimators and imputation chains are fitted offline. Small SVD reference indexes are constructed lazily and cached when a completed query first uses a matching mode. Production weights, cleaning masks, calibration and default pricing are unchanged.

### Comparable retrieval and missing values

Three independent 32-dimensional SVDs and exact FAISS IndexFlatIP search use L2-normalized vectors. Ask adds optional 80%/20% SVD/MiniLM semantic blocks. Bid alone adds the imputed log description-price proxy. Hybrid uses vehicle attributes only and searches both subsets in the same combined projection. No direct advert/auction price column enters any SVD.

Offline price-assisted IterativeImputer chains use native-categorical LightGBM classifiers and numerical regressors for Ask/Bid; the auxiliary price column is dropped from their output. Original observations are preserved. Hybrid's imputer never sees price or text. Portable hashed JSON stores the fitted schedules and trees; no serving fit or pickle load occurs. Random-cell reconstruction audits do not establish missing-not-at-random quality. See COMPARABLE_INSPECTOR.md for weights, distance, context-date handling and feature provenance.

### Specific-car selector

`data/vehicle-examples.json` contains exactly 1,000 individual historical vehicle records, with unique source IDs, year, make/model, mileage and available details/text. No popularity ordering or generic make/model preset is used. Selecting a record updates the complete editable Inspector form and clears stale comparison/passport results. If the compiled example file is not installed yet, the same deterministic records are assembled from existing local references. Historical auction examples have no description and never inherit previous text. Missing mileage stays empty. The selector targets 500 asking and 500 auction records and shuffles them in a stable, deterministic order; when only 400 asking references exist, 600 auction records complete the list.

### Duplicate matching

100 rows / 2 MB maximum; required make, model, integer year 1900–2021 and finite mileage 0–2 million. Optional prices must be positive finite dollars. Valid-format supplied VINs are SHA-256 hashed, never exported. Pairwise review has at most 4,950 pairs. A VIN match qualifies regardless of specification conflict. Without one, require equal normalized make/model/year, mileage difference <=10,000, descriptions >=80 characters and token-set Jaccard >=0.85. This catches repetition without treating a common dealer template alone as identity evidence. Rules miss paraphrases and do not establish fraud or chronology.

### Appraisal and batch review

Explicit actions call the existing `hierarchy_sparse_live.estimate`, not a new pricing estimator. Batch limits are 25 vehicles, evaluated sequentially. Per-row failures yield unavailable statuses. No bulk model fitting or GPU work occurs in Streamlit. Outputs are invalidated by content hash, date, generation assumption and production release identity. The single-car passport is independent of the landing-page saved receipt. CSV exports neutralize formula-leading user strings.

Review labels use asking price versus the existing asking range, rare-model support (<12), unseen categories, range width/point >1. They are deterministic triage rules, not fitted outcome probabilities. Current date adjustment does not validate contemporary interval coverage. Auction estimates are separate outputs; their difference from asking is not a profit label.

### Offline controlled experiments

Verified source ZIP hashes are checked in CI, then existing canonical preparation recipes run. Deterministic SHA-256 rank sampling is applied independently inside original chronological partitions, with caps of 40,000 train / 10,000 validation / 10,000 test per market. Original counts and canonical-file hashes are retained. No residual trimming is performed for this audit. Cross-partition vehicle overlap causes failure.

The audit baseline is Ridge(alpha=10, solver=lsqr) on log dollars. Sparse one-hot controls cover make, make/model, state, body, fuel, transmission, drive, title, retail condition, cylinders, colour and month; categories occurring <10 times are pooled by the encoder. Standardized numerical controls include age, age squared, log(1+mileage), its square, age×log mileage, calendar day, auction grade and its missing indicator. Preprocessing is fitted on training only. Dollar predictions are exp(log prediction), clipped to the declared numerical support; they target a log-location and do not apply a mean-price smearing correction.

Regional effects are state coefficients relative to the training-count-weighted mean of supported states (>=100 training vehicles). Forty training-vehicle bootstrap refits retain fixed encoding and penalty and produce exploratory 90% percentile intervals. There is no claim of causal identification, full covariate overlap, simultaneous coverage or a pairwise region-difference interval. Sampling, regularization and omitted attributes limit interpretation.

The asking description experiment adds 384 frozen int8 all-MiniLM-L6-v2 vectors (up to 128 tokens after existing price/contact cleanup) to the same tabular Ridge controls. No hand-written keyword flags or condition labels are used. Encoding is offline on CPU; the production description-price head and pricing weights are unchanged. Baseline and augmented dollar MAE use the same unchanged historical test rows. A 300-draw paired vehicle-group bootstrap estimates a 95% interval for the difference in absolute error. Negative improvement is reported, not hidden. Source test data have been examined by previous project releases, so this is not a new external benchmark. Embedding dimensions have no standalone word-premium interpretation; no causal text effect is claimed.

A separate symmetric absolute-log-residual radius is calibrated on validation and its historical test coverage recorded for the audit baseline. It does not replace production Bayesian intervals. Research-training outlier diagnostics use fresh three-fold GroupKFold predictions with fold-specific encoders; the top residual quartile is flagged only. No production record is deleted.

### Testing

`test_market_review.py` covers invalid/missing inputs, size limits, VIN conflicts, template-only false matches, spreadsheet export safety, failure-isolated batches, historical partition leakage, controlled experiment output and preset callbacks. Comparable, BART and automatic-summary regressions remain included. CI verifies real original-data report rendering before publishing small aggregate JSON files. Browser validation is separate from a successful source push.


## Historical prices adjusted to the valuation date

Comparison cards retain the original observed asking/auction price and observation date, then apply the existing depreciation age/mileage ratio and the ratio of the aggregate used-car market index at the valuation and observation dates. The default valuation date follows today. Optional mileage normalization puts each reference at the entered car’s mileage; turning it off holds each reference’s stated mileage fixed. Neither assumption tracks the physical vehicle’s subsequent use or condition. National retail used-car CPI is also a wholesale proxy; when observations lag, the existing damped market forecast is disclosed. In Ask/Bid only, adjusted prices and the query estimate assist missing-feature imputation and are then dropped before SVD. This indirect conditioning prevents treating neighbours as independent validation. Unsupported age ranges or missing source prices/dates/mileage yield an explicit unavailable adjustment, not a guessed current price. These scenarios do not have validated current-vehicle accuracy or prediction coverage.


## Price-assisted completion, price-free SVD

Ask/Bid use the queried car’s selected-market model estimate as auxiliary imputation context where supported. Reference and query context share the saved imputer's dated price basis. The actual price column is dropped before SVD; only Bid retains the separate imputed MiniLM description-price proxy. Hybrid excludes every price/text signal. Displayed historical prices are independently adjusted to the selected date. These are scenarios, not verified current offers or an independent valuation check.

## Missing auction description-price feature

Auction records contain no seller description. Their description-price feature starts as NaN, not zero or invented text. A shared sklearn IterativeImputer with small native-categorical LightGBM regressors learns from observed asking-description prices, vehicle attributes and reference-2021 price context. The asking signal is the mean of the two installed MiniLM price-head dollar predictions. Known asking signals are preserved; missing asking and all auction signals are imputed. The inferred quantity is an asking-style statistical proxy, not an observed auction description or auction offer. It is used only in Bid retrieval, never Ask, Hybrid or production valuation.

The reconstruction audit masks asking signals and compares dollar error with a median baseline. It does not validate transfer to auction descriptions, which do not exist. Models and schedules are stored in hashed compressed JSON with LightGBM text trees. Hybrid uses a separate vehicle-only imputer, excluding price and text even during missing-value completion.

