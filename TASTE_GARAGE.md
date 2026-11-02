# Pick your garage: soft HDBSCAN

## Business guide
Rate ten real representative cars from very bad to very good. Each historical car has ten overlapping profile affinities, rather than a forced cluster rating. Your reactions blend through these affinities to form an inferred preference. Unassigned membership receives your average rating. XGBoost learns the ordering and returns ten historical adverts; these are not live stock.

## Technical guide
The full canonical Craigslist cohort is encoded using smoothed leave-one-out category means of log advertised price (alpha 20); both the category sum and global prior exclude the row. Constant and identifier-like category columns are suppressed. Year, log mileage and mileage missingness join the dense matrix. StandardScaler and clipping at six standard deviations precede Euclidean HDBSCAN. No UMAP or SVD is used.

HDBSCAN EOM varies min_cluster_size with min_samples=10, seeking approximately ten native groups. If it first supports more than ten, Ward merging of native centroids produces exactly ten macroprofiles. This is explicitly different from claiming ten native density clusters. Soft native memberships from HDBSCAN membership_vector are summed across merged groups, in bounded batches, to form M with shape n_cars by 10. Row sums may be below one; this noise mass is retained. Argmax is only a display association, not a training label. No HDBSCAN object is loaded by the web app.

A representative is a real non-noise advert selected by sum of Euclidean distances to group members. Groups up to 2,000 use exact medoids. Larger groups use a deterministic 256-candidate approximate medoid, evaluated against every member; this approximation bounds CPU and memory and is disclosed in the technical UI. A centroid is never presented as a car.

For ratings r in 0..3, soft relevance is M @ r + (1 - row_sum(M)) * mean(r). The XGBRanker uses rank:ndcg, eval_metric ndcg@10, ndcg_exp_gain=False. It requires integer labels, so training relevance is rounded to 0.1 steps (integers 0..30), preserving blended preferences rather than taking an argmax rating. The displayed linear-gain NDCG uses unrounded soft relevance. Native categorical vehicle features, original log advertised price, supported log date-adjusted price and its missing indicator are predictors. Memberships, profile IDs and user features are not predictors. Each original advert appears exactly once, with no Cartesian expansion.

An approximately 20% deterministic vehicle-ID split chooses the iteration count. The ten directly rated medoids remain in training and are excluded from recommendations. The final model refits all rows once. Validation labels are inferred from only ten real judgments; NDCG is weak-label agreement, not measured satisfaction. Price-informed geometry is transductive, using the full catalogue. Additional shortlist ratings can assess displayed ordering but cannot measure catalogue-wide recall.

## Dates and sources
Today follows the calendar. Historical advertised prices remain visible beside depreciation/CPI/Holt date-adjusted scenarios. Current-date accuracy is unvalidated. Craigslist Used Cars: Tom, Zenodo 10457828, CC BY 4.0. Source descriptions are excerpts, not verified vehicle facts. Existing pricing, Bayesian, BART and comparison panels remain separate.

## New daily listings: no refit
The separate frozen-hdbscan-daily-listing-prediction Actions artifact contains the training category statistics, fitted scaler, HDBSCAN prediction data, fixed macro mapping and actual medoid vectors. It is intentionally excluded from the memory-sensitive web app. New rows use reference-category smoothed means (their own target is never needed), saved scaling and hdbscan.membership_vector. Unseen categories fall back to the training global prior. Euclidean distances to fixed medoids are additional explanations, not substitutes for density memberships. No UMAP transform is involved. Training rows retain leave-one-out encoding; inference uses the full frozen reference statistics.

Run python -m scripts.assign_taste_listings frozen-profiles.joblib daily.jsonl --output assigned.jsonl using the trusted build artifact. This creates affinities for supplied new listings; it does not claim that a daily source feed has been connected or automatically append new inventory to the web catalogue. Refit only when intentionally updating the reference model to address drift.

## FAISS RMSE distance
Exact IndexFlatL2 searches the fixed real medoid vectors. FAISS returns squared L2; displayed RMSE is sqrt(squared_L2 / number_of_scaled_features). Vectors are not unit-normalized, so this is not cosine distance. Lower is nearer; units are standardized feature units, never dollars. HDBSCAN memberships still come from its density prediction, and XGBoost preference scores order the shortlist. New-listing inference uses the same FAISS distance without refitting. The offline clustering result is reused when changing ranking or distance presentation.

## Ranking release checks
A ranking with constant predictions is rejected. Lower minimum child Hessian and moderate regularization allow LambdaMART to split despite top-k pair sampling in a single large query. At least 16 trees are fitted after validation chooses the best iteration. Score range and comparison with random ordering are reported; weak-label NDCG is still not independent evidence of user satisfaction.
