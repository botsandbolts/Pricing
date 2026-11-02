# Pick your garage

**UMAP neighbourhood embedding → HDBSCAN-based profiles → signed rating evidence and FAISS L2 safeguards → XGBoost ordering.** The recommender is a hybrid. HDBSCAN discovers density profiles; XGBoost does not perform the clustering. The separate price estimators are not the recommender.

## Two tests, one shortlist
Speed Date · 10 cars, no small talk: ten real representatives for a short introduction to your taste.

The Long Drive · 25 cars, zero commitment: twenty-five real representatives and finer density profiles. More judgments are requested; better recommendations are not assumed without independent user testing.

Both tests return up to ten historical adverts, exclude the representatives you rated, preserve separate ratings when switching, and show original asking prices beside supported date-adjusted scenarios. These are not live offers.

## Geometry and density profiles
Category encoding is smoothed leave-one-out mean log asking price, alpha 20. Both category sums and global prior exclude the row. Constant and identifier-like columns are suppressed. Year, log mileage and missingness join the matrix, then StandardScaler and clipping at six standard deviations. These original scaled features enter the five-dimensional UMAP preprocessing described below. No SVD or one-hot features are used here.

The UMAP release fits HDBSCAN in its five-dimensional embedding and selects separate cuts with exactly 10 and 25 profiles. It reuses that fitted hierarchy between cuts. The release fails if the required real clusters cannot be selected. Before this verified release is installed, the legacy catalogue remains available: its short test merged eleven original-space groups into ten profiles. The hierarchy-selection adapter is tested against pinned hdbscan 0.8.44.

Every representative is an actual non-noise car. Short-test medoids retain their original Euclidean objective. Long-test medoids minimize summed L2 distances. For up to 2,000 members the medoid search is exact; larger groups evaluate a deterministic 256-candidate approximation against every member. No centroid is presented as a real car.

## Soft memberships and ranking
Each test has its own n_cars by K membership matrix, K=10 or 25. Native HDBSCAN membership strengths are retained without forcing uncertain cars into a group. Ratings 1, 2, 3, 4 map to -1, -1/3, +1/3, +1. Unassigned membership contributes zero, not average approval. Positive and negative membership evidence are tracked separately. A provisional, unvalidated 75/25 blend combines density evidence with distance evidence: inverse-square L2 affinity 1/(1+d²), row-normalized and multiplied by its maximum affinity to reduce confidence far from every representative. Negative ratings subtract evidence. This is an explicit preference heuristic, not calibrated satisfaction probability.

XGBRanker uses rank:ndcg, eval_metric ndcg@10, native categorical features and linear NDCG gains. Soft relevance is quantized to 0.1 steps for integer training labels; displayed held-out NDCG uses unrounded relevance. Vehicle attributes, original log asking price, supported date-adjusted log price and its missing indicator enter XGBoost. No user identity, cluster IDs or membership columns are predictors. Each advert occurs once; no Cartesian expansion.

A deterministic approximately 20% vehicle-ID split chooses the iteration count, with rated representatives kept in training. At least 16 trees are fitted for production, and constant predictions fail the release gate. The final model fits all rows once and produces up to ten recommendations. NDCG measures agreement with inferred labels, not independently measured satisfaction. The long test is not claimed to outperform the short test. Additional real ratings are needed to assess preference quality.

## Exact FAISS L2
FAISS IndexFlatL2 returns squared L2, which is converted to Euclidean distance in the frozen scaled feature space. Lower is closer. FAISS returns squared L2; we take its square root. No division by dimension or unit-vector normalization is applied. These are not dollar errors or liking probabilities. FAISS supplies nearest-medoid explanations; HDBSCAN supplies density memberships; XGBoost orders recommendations. Existing Listing Inspector SVD/cosine tools are separate and unchanged.

## New listings without refitting
Trusted offline artifacts contain the frozen category statistics, fitted scaler, density prediction data, profile mapping and medoid vectors for each test. New categories fall back to the training global prior. New listing targets are not required to encode features. Membership prediction uses the corresponding fixed hierarchy; L2 explanations use the same fixed medoids. Both UMAP profile artifacts use membership_vector_flat so epsilon-selected labels and prediction data agree. Neither path refits HDBSCAN.

Run python -m scripts.assign_taste_listings TRUSTED-BUNDLE.joblib daily.jsonl --output assigned.jsonl. Outputs contain memberships, unassigned mass, medoid_l2 and nearest profile ordering. A daily inventory feed has not been connected automatically. The memory-sensitive web app loads compact arrays and requested advert rows, not the full HDBSCAN object.

## Sources, dates and limits
211,844 canonical Craigslist adverts from Tom, Zenodo 10457828, CC BY 4.0. This is price-informed, transductive catalogue exploration; source descriptions and recorded mileage are not verified facts. Date-adjusted scenarios reuse existing depreciation curves and CPI/Holt market transport. Current-date individual accuracy is unvalidated. Existing pricing, Bayesian, BART, sharing and comparison panels remain separate.

## Dislike protection
A candidate needs positive net evidence. If both liked and disliked representatives exist, its nearest liked representative must be strictly closer in L2 than its nearest disliked one. These checks apply before the XGBoost shortlist and cannot be overridden by a tree score. Representatives themselves are excluded. If fewer than ten candidates qualify, fewer are shown; no disliked fallback fills the list. At least one I like it or Love it rating is required. Tree-score ties use inferred preference as a secondary ordering. Each result shows positive evidence, negative evidence, and nearest liked/disliked distances. Signed evidence is shifted to nonnegative 0..3 relevance solely for LambdaMART/NDCG. These safeguards operate in model feature space and do not guarantee a person will like every recommended car.

## UMAP density upgrade
The serving catalogue now prefers the complete, verified `data/taste-garage-umap` release. Until all its required arrays/manifests are installed, the previous tested catalogue stays active. Training uses the entire original canonical cohort, not a sample. Original scaled features enter UMAP with n_neighbors=30, min_dist=0, n_components=5, Euclidean metric, random_state=42, 200 epochs and low_memory=True. HDBSCAN then fits that five-dimensional representation with min_samples=10 and min_cluster_size=500. The 10/25-profile tests select separate flat cuts from this new hierarchy, never reusing the old memberships. Their medoids remain actual cars selected within the new clusters using original-feature L2 distance.

The frozen UMAP transform and density model are exported as offline artifacts for new-listing assignment without refitting. They are excluded from the web process. The GUI needs only compact precomputed memberships, original-feature medoid distances and advert rows. New listing prediction runs frozen encoding/scaling → UMAP.transform → HDBSCAN soft memberships. FAISS and dislike safeguards continue to use the original scaled features, not UMAP coordinates. Existing pricing and description models are unchanged.

`cluster-comparison.json` reports the old/new largest profile shares, all profile sizes and native noise fractions. These describe structure and coverage, not personal relevance. The original category encoding remains price-informed. UMAP cannot manufacture missing preference signals, may distort density and can introduce artificial splits; neither a balanced partition nor inherited-label NDCG proves better recommendations. Use additional independent shortlist ratings to assess personal relevance.

Method reference: https://umap-learn.readthedocs.io/en/latest/clustering.html


## Cosine preference comparison
New UMAP catalogues use exact FAISS IndexFlatIP on L2-normalized original scaled vectors. Distance is 1 minus cosine similarity (0–2), with zero vectors assigned neutral similarity zero. Both representative affinity and the nearest-liked versus nearest-disliked exclusion use this same distance. This changes preference comparison, not the Euclidean UMAP/HDBSCAN density metric or the original-space medoid objective. Cosine emphasizes direction rather than magnitude and is not a proven relevance improvement. Legacy catalogues retain their labelled Euclidean distances until the verified cosine assets replace them.
