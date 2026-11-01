# Listing decision tools — technical guide

## Architecture and isolation

`listing_inspector.py` is a separate horizontal workspace. `market_review.py` contains bounded CSV parsing, duplicate matching, review rules and sequential batch orchestration. `listing_insights_ui.py` presents the tools. Offline research lives in `listing_statistics.py` and `scripts/build_listing_insights.py`. Neither visiting a panel nor changing a tab fits a model. Production weights, cleaning masks, calibration and default pricing are unchanged.

### Comparable retrieval and missing values

FAISS IndexFlatIP performs exact five-neighbour search on L2-normalized 32-dimensional SVD representations of weighted one-hot categories plus standardized numeric inputs. Optional independently normalized SVD/MiniLM blocks are combined at 80%/20% weights and normalized again. Inner product equals cosine similarity. There is no intermediate candidate stage, LambdaRank/FM, reranking or same-make/model/year exclusion. Only invalid representations and explicit self records are omitted. See COMPARABLE_INSPECTOR.md for representation and reference details.

Offline IterativeImputer uses LightGBM regression for numeric values and native-categorical LightGBM classifiers for optional categorical values. Three categorical and up to five numeric chained iterations use small trees, two CPU threads, no prices and no posterior sampling. Observed values are restored; all-missing unsupported categories remain unknown. Missingness-derived values affect retrieval only, with disclosure on each card. A held-out random-cell masking audit is not proof of missing-not-at-random reconstruction quality. Portable JSON stores the fitted chain and LightGBM text models; a round-trip test checks agreement with sklearn. No serving-time imputer fitting or unsafe pickle load occurs. Reference and model hashes prevent applying the artifact to another catalogue.

### Specific-car selector

`data/vehicle-examples.json` contains exactly 1,000 individual historical vehicle records, with unique source IDs, year, make/model, mileage and available details/text. No popularity ordering or generic make/model preset is used. Selecting a record updates the complete editable Inspector form and clears stale comparison/passport results. If the compiled example file is not installed yet, the same deterministic records are assembled from existing local references. Historical auction examples have no description and never inherit previous text. Missing mileage stays empty. A subsequent larger asking catalogue supplies sufficient asking examples for an all-asking set.

### Duplicate matching

100 rows / 2 MB maximum; required make, model, integer year 1900–2021 and finite mileage 0–2 million. Optional prices must be positive finite dollars. Valid-format supplied VINs are SHA-256 hashed, never exported. Pairwise review has at most 4,950 pairs. A VIN match qualifies regardless of specification conflict. Without one, require equal normalized make/model/year, mileage difference <=10,000, descriptions >=80 characters and token-set Jaccard >=0.85. This catches repetition without treating a common dealer template alone as identity evidence. Rules miss paraphrases and do not establish fraud or chronology.

### Appraisal and batch review

Explicit actions call the existing `hierarchy_sparse_live.estimate`, not a new pricing estimator. Batch limits are 25 vehicles, evaluated sequentially. Per-row failures yield unavailable statuses. No bulk model fitting or GPU work occurs in Streamlit. Outputs are invalidated by content hash, date, generation assumption and production release identity. The single-car passport is independent of the landing-page saved receipt. CSV exports neutralize formula-leading user strings.

Review labels use asking price versus the existing asking range, rare-model support (<12), unseen categories, range width/point >1, and . They are deterministic triage rules, not fitted outcome probabilities. Current date adjustment does not validate contemporary interval coverage. Auction estimates are separate outputs; their difference from asking is not a profit label.

### Offline controlled experiments

Verified source ZIP hashes are checked in CI, then existing canonical preparation recipes run. Deterministic SHA-256 rank sampling is applied independently inside original chronological partitions, with caps of 40,000 train / 10,000 validation / 10,000 test per market. Original counts and canonical-file hashes are retained. No residual trimming is performed for this audit. Cross-partition vehicle overlap causes failure.

The audit baseline is Ridge(alpha=10, solver=lsqr) on log dollars. Sparse one-hot controls cover make, make/model, state, body, fuel, transmission, drive, title, retail condition, cylinders, colour and month; categories occurring <10 times are pooled by the encoder. Standardized numerical controls include age, age squared, log(1+mileage), its square, age×log mileage, calendar day, auction grade and its missing indicator. Preprocessing is fitted on training only. Dollar predictions are exp(log prediction), clipped to the declared numerical support; they target a log-location and do not apply a mean-price smearing correction.

Regional effects are state coefficients relative to the training-count-weighted mean of supported states (>=100 training vehicles). Forty training-vehicle bootstrap refits retain fixed encoding and penalty and produce exploratory 90% percentile intervals. There is no claim of causal identification, full covariate overlap, simultaneous coverage or a pairwise region-difference interval. Sampling, regularization and omitted attributes limit interpretation.

The asking description experiment adds 384 frozen int8 all-MiniLM-L6-v2 vectors (up to 128 tokens after existing price/contact cleanup) to the same tabular Ridge controls. No hand-written keyword flags or condition labels are used. Encoding is offline on CPU; the production description-price head and pricing weights are unchanged. Baseline and augmented dollar MAE use the same unchanged historical test rows. A 300-draw paired vehicle-group bootstrap estimates a 95% interval for the difference in absolute error. Negative improvement is reported, not hidden. Source test data have been examined by previous project releases, so this is not a new external benchmark. Embedding dimensions have no standalone word-premium interpretation; no causal text effect is claimed.

A separate symmetric absolute-log-residual radius is calibrated on validation and its historical test coverage recorded for the audit baseline. It does not replace production Bayesian intervals. Research-training outlier diagnostics use fresh three-fold GroupKFold predictions with fold-specific encoders; the top residual quartile is flagged only. No production record is deleted.

### Testing

`test_market_review.py` covers invalid/missing inputs, size limits, VIN conflicts, template-only false matches, spreadsheet export safety, failure-isolated batches, historical partition leakage, controlled experiment output and preset callbacks. Comparable, BART and automatic-summary regressions remain included. CI verifies real original-data report rendering before publishing small aggregate JSON files. Browser validation is separate from a successful source push.
