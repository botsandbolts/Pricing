# Pick your garage

## Two tests, one shortlist
Speed Date · 10 cars, no small talk: ten real representatives for a short introduction to your taste.

The Long Drive · 25 cars, zero commitment: twenty-five real representatives and finer density profiles. More judgments are requested; better recommendations are not assumed without independent user testing.

Both tests return ten historical adverts, exclude the representatives you rated, preserve separate ratings when switching, and show original asking prices beside supported date-adjusted scenarios. These are not live offers.

## Geometry and density profiles
Category encoding is smoothed leave-one-out mean log asking price, alpha 20. Both category sums and global prior exclude the row. Constant and identifier-like columns are suppressed. Year, log mileage and missingness join the matrix, then StandardScaler and clipping at six standard deviations. No UMAP, SVD or one-hot features enter this geometry.

The short test uses the completed full-cohort HDBSCAN fit: eleven EOM clusters merged to ten macroprofiles. The long test reuses its saved single-linkage hierarchy, recondenses when necessary and chooses a leaf-selection epsilon supporting exactly 25 clusters. The neighbour graph is not rebuilt. The release fails if 25 real clusters cannot be selected. The internal hierarchy-selection adapter is tested with hdbscan 0.8.44 pinned in the release environment.

Every representative is an actual non-noise car. Short-test medoids retain their original Euclidean objective. Long-test medoids minimize summed L1 distances. For up to 2,000 members the medoid search is exact; larger groups evaluate a deterministic 256-candidate approximation against every member. No centroid is presented as a real car.

## Soft memberships and ranking
Each test has its own n_cars by K membership matrix, K=10 or 25. Native HDBSCAN membership strengths are retained without forcing uncertain cars into a group. Unassigned mass is one minus the row sum. For ratings r mapped to 0..3, inferred relevance is M @ r + (1-row_sum(M))*mean(r).

XGBRanker uses rank:ndcg, eval_metric ndcg@10, native categorical features and linear NDCG gains. Soft relevance is quantized to 0.1 steps for integer training labels; displayed held-out NDCG uses unrounded relevance. Vehicle attributes, original log asking price, supported date-adjusted log price and its missing indicator enter XGBoost. No user identity, cluster IDs or membership columns are predictors. Each advert occurs once; no Cartesian expansion.

A deterministic approximately 20% vehicle-ID split chooses the iteration count, with rated representatives kept in training. At least 16 trees are fitted for production, and constant predictions fail the release gate. The final model fits all rows once and produces ten recommendations. NDCG measures agreement with inferred labels, not independently measured satisfaction. The long test is not claimed to outperform the short test. Additional real ratings are needed to assess preference quality.

## Exact FAISS L1
FAISS IndexFlat with METRIC_L1 returns the sum of absolute coordinate differences in the frozen scaled feature space. Lower is closer. There is no square, square root, division by dimension, or unit-vector normalization. These are not dollar errors or liking probabilities. FAISS supplies nearest-medoid explanations; HDBSCAN supplies density memberships; XGBoost orders recommendations. Existing Listing Inspector SVD/cosine tools are separate and unchanged.

## New listings without refitting
Trusted offline artifacts contain the frozen category statistics, fitted scaler, density prediction data, profile mapping and medoid vectors for each test. New categories fall back to the training global prior. New listing targets are not required to encode features. Membership prediction uses the corresponding fixed hierarchy; L1 explanations use the same fixed medoids. The 25-profile artifact uses membership_vector_flat so epsilon-selected labels and prediction data agree. Neither path refits HDBSCAN.

Run python -m scripts.assign_taste_listings TRUSTED-BUNDLE.joblib daily.jsonl --output assigned.jsonl. Outputs contain memberships, unassigned mass, medoid_l1 and nearest profile ordering. A daily inventory feed has not been connected automatically. The memory-sensitive web app loads compact arrays and requested advert rows, not the full HDBSCAN object.

## Sources, dates and limits
211,844 canonical Craigslist adverts from Tom, Zenodo 10457828, CC BY 4.0. This is price-informed, transductive catalogue exploration; source descriptions and recorded mileage are not verified facts. Date-adjusted scenarios reuse existing depreciation curves and CPI/Holt market transport. Current-date individual accuracy is unvalidated. Existing pricing, Bayesian, BART, sharing and comparison panels remain separate.
