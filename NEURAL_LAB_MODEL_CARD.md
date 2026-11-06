# NEURAL LAB — DeepCross AFT + MiniLM

## Audience and purpose
A research challenger for **seller-advertised used-car asking prices**, not transaction, trade-in or wholesale prices. Retains Car Price Studio production calculator unchanged. Experimental Models is a separately branded, consent-aware alpha panel.

## Business
Model output is a historical asking-price anchor (May 2021). Date-adjusted estimates multiply that anchor by an age-depreciation ratio and monthly used-car market index; they have **no validated current-date test MAE**. A hypothetical listing price does not guarantee a sale. The receipt separates historical estimate, ageing, market factor, and the current selected-date scenario.

## Dataset and validation
Historical Craigslist 2021 asking listings: **211,127** structurally eligible cars; unchanged chronological train 147,777 / validation 31,668 / test 31,682. Selected from 36 experimental runs by external validation MAE. Chosen config DeepCross + exact-observation AFT + MiniLM, main dropout 0.12, full training cohort (133,000 fitting and 14,777 internal early stopping). **Validation MAE 3,046.614 USD; test MAE 3,261.537 USD.** Previous XGBoost full-data published test MAE 3,377.730 USD is an aggregate same-size comparator, not independently matched row-level predictions. Single-seed selected result; no replicated multi-seed guarantee.

## Numeric, categorical and text inputs
7 robust-scaled numeric factors: year, age at 2021, log(1+mileage), mileage per age-year, log description characters/words/punctuation. 11 categorical embeddings: make, model, type, transmission, state, fuel, drive, title, cylinders, paint and condition. Unknown category has explicit id 1.
Unsupervised TF-IDF unigram/bigram training-only atlas of 30,000 source listings yields 24 train-scaled SVD factors. Pinned frozen all-MiniLM-L6-v2 revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 produces 384-dimensional text embeddings, fed through Ridge alpha 100 price heads. Each of three heads trains on one disjoint third of 133,000 fit rows, giving two out-of-bag head predictions per fit car. Internal early stopper and untouched holdouts receive the mean of all three training-only heads. This scalar is log-transformed, median/IQR normalized and clipped.

## Network, likelihood, uncertainty
Parallel low-rank explicit cross interactions and residual deep layers; two-output head learns log-location mu and positive scale sigma. AFT-style exact-observation log-normal negative log-likelihood: mean(0.5*((log(y)-mu)/sigma)^2+log(sigma)). No genuine survival intervals, observed censoring windows or auction descriptions. No model-derived or current-price prediction interval is asserted. What-if charts are model responses, not causal effects or SHAP.

## Deployment and reproducibility
Versioned local PyTorch state_dict weights; trusted compressed joblib TF-IDF/SVD; JSON category maps and scalers; three original Ridge head coefficients; pinned existing MiniLM ONNX encoder. A packaging CI job must rebuild feature projections on the same immutable training dataset, compare original stored tensors and original held-out checkpoint predictions with live inference before installing. Fail closed with a clear unavailable state. No database used for model files. Slower alpha inference is acceptable, with a truthful stage-based humorous progress bar.

## UX
Separate Neural Lab navigation and responsive branding; full vehicle editor and real editable redacted examples; date controls and historical/time-adjusted receipts; mileage, condition, description, and timeline scenarios; downloadable price receipt (description omitted); business-level and scientific documentation; wholesale bid deliberately excluded pending its own verified model.

## Analytics / consent
Existing Turso feature_runs and pricing_queries tables are used **only with consent**. Model family and version, success/error status, duration, date-adjusted asking-price event, and allowlisted vehicle fields. No text, embeddings, VIN, contacts or file models are saved. Tracking failures do not interrupt estimation. There is no new model-storage database.

## Known limitations
2021 seller demands may not be realized transactions; novel trims, rare manufacturers, damaged vehicles and out-of-distribution text have unknown errors. Using an aggregate current market index and cross-sectional depreciation cannot establish verified current-price accuracy. Repeat benchmark across random seeds, verify Streamlit deployment resources, and assess fair coverage before any production promotion.

Sources: reports/deepcross-aft-tuning/report.json and leaderboard.csv; Actions run 42252201025.
