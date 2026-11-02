# Pick your garage · one-hot HDBSCAN + mileage + cosine

The one-hot release rebuilds HDBSCAN profiles, real-car medoids, soft memberships and cosine comparison matrices for both the 10- and 25-car tests. It uses the same 211,844 canonical adverts. No UMAP or other dimensionality reduction is used.

## Clustering features

- Common values in each categorical field get independent 0/1 indicators. A category is common when its count is at least `max(100, ceil(0.001 × reference_rows))`: 212 cars in the installed cohort. Manufacturer, model, body type, fuel, transmission, drive, title status, condition, cylinders, paint colour and state are considered.
- Rare and unseen values share an explicit Other bucket within each retained field. A missing categorical value is handled by the same frequency rule. Constant grouped fields are omitted. The frozen vocabulary is reused for future listings.
- **Vehicle year, log(1 + mileage in miles), and a missing/invalid-mileage indicator are included.** Missing numerical values use medians fitted on the reference cohort. Numerical features are standardized with the frozen reference scaler and clipped to six standard deviations.
- One-hot indicators are not variance-scaled, so uncommon retained categories are not amplified by inverse-frequency scaling. Asking prices, descriptions, user attributes and preference ratings are excluded from the HDBSCAN geometry. There is no leave-one-out/target encoding in this release.

HDBSCAN uses Euclidean density geometry directly on these features. The short test selects or merges native density groups into ten macroprofiles. The long test selects 25 finer density groups from the newly fitted hierarchy. Representatives are actual non-noise cars, selected using the Euclidean medoid objective; larger groups use a deterministic candidate approximation. The serving GUI loads compact arrays, not the heavy fitted clustering object.

## Ratings and ordering

Rate each representative from 1 (No thanks) to 4 (Love it). Ratings map to signed evidence -1, -1/3, +1/3, +1. Density evidence has 75% weight and representative-distance affinity 25%. These are provisional heuristic weights, not calibrated liking probabilities. Unassigned membership contributes zero.

Exact FAISS IndexFlatIP operates on L2-normalized copies of the same one-hot/numeric vectors. Cosine distance is 1 minus inner product, between 0 and 2. Zero vectors produce neutral similarity zero. The same distances drive preference affinity, exclusion and displayed explanations. A candidate needs positive net evidence and, when both liked and disliked examples exist, must be strictly closer to a liked representative than a disliked one; ties are excluded. Rated prototypes are never recommendations.

A separate vehicle-only XGBRanker orders eligible candidates with rank:ndcg and ndcg@10. It retains native categorical inputs plus original and supported date-adjusted log asking prices. The change to one-hot encoding applies to clustering and cosine comparisons, not XGBoost. There are no user attributes or Cartesian row expansion. Up to ten historical adverts are returned; they are not live inventory.

## Validation and daily listings

The release checks target-free encoding, rare/unseen handling, mileage effects, frozen future-listing transforms, real medoids, cosine distances, positive evidence and nearest-dislike safeguards. Both full-catalogue Streamlit rating flows must produce distinct eligible recommendations before publication. Mechanical held-out NDCG measures agreement with inherited preferences, not independently measured satisfaction. Optional additional ratings provide a limited direct check; a longer test is not assumed to be better.

New listings pass through the frozen common-category vocabulary and numerical scaler, then HDBSCAN membership prediction and FAISS cosine comparison without refitting. There is no automatic live-inventory feed. The vocabulary and scaler were fitted on the whole reference catalogue, so this is transductive catalogue exploration, not an external benchmark.

Run `python -m scripts.rebuild_onehot_garage`, then `python -m scripts.verify_taste_release --bundle training-cache/taste-incremental/frozen-profiles.joblib`. Publish the code, matching 10/25 assets and verification receipt together. Keep the source-catalogue hash and SQLite integrity check. Older leave-one-out helpers exist only for historical artifacts/tests and are not the production one-hot build path. The manual one-hot workflow can repeat the rebuild and save frozen prediction artifacts. Routine code checks do not overwrite a newer catalogue with an old hard-coded artifact.
