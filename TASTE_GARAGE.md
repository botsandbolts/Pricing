# Pick your garage · Lemon-only mixed-data UMAP + HDBSCAN

The catalogue contains **110,832 retained Lemon Squeezer training cars**. It excludes the discarded training rows and all pricing validation/test rows. The saved cross-fitted `predicted_text_price` scalar is reused by row ID. No MiniLM embeddings or fresh text-model inference are computed for this recommender.

## Inputs

Common categorical values get independent one-hot indicators; rare and unseen values share an Other bucket per field. Common means at least `max(100, ceil(0.001 × retained_rows))`, or **111 cars** for this release. Constant grouped fields are omitted. Category encoding uses no price-target statistics.

The numeric view includes vehicle year, **log(1 + mileage)**, a missing-mileage indicator, log asking-price scenarios for **November 2026**, and log of the **existing text-price scalar**. Unsupported asking-price transport and unavailable future text-price values have explicit missing indicators. Training requires the stored positive text-price value for every retained row. No tree-model `predicted_price` is substituted for `predicted_text_price`.

The November scenario uses 1 November 2026 as the monthly reference date, existing depreciation curves and the dated market-index transport. Recorded mileage is held constant. This is an extrapolated scenario, not a live sale value. The existing text-price scalar is retained on its original historical dollar basis. Numeric values use frozen medians and 1st/99th percentile ranges, clipped to [0,1]; flags remain [0,1]. Clipping is a feature transform, not an additional row filter.

## Mixed distance and clustering

UMAP uses `0.5 × Dice(one-hot) + 0.5 × mean absolute numeric difference`. Dice ignores joint-zero category indicators. Numeric differences are computed after robust range scaling. The 50/50 view weights are explicit heuristic choices, not a proven optimal metric. Arbitrary category integers and Euclidean distance across an unscaled mixed matrix are not used.

Every retained row enters an approximate mixed-distance neighbour graph. UMAP uses 30 neighbours, min_dist=0, five output dimensions, 200 epochs and seed 42, following the clustering-oriented setup in the [UMAP clustering guide](https://umap-learn.readthedocs.io/en/latest/clustering.html). The [mixed-view documentation](https://umap-learn.readthedocs.io/en/latest/composing_models.html) discusses applying different metrics to categorical and numerical inputs. UMAP can alter density and split structure; it does not establish that the resulting profiles are objectively correct.

HDBSCAN uses Euclidean distance in the five-dimensional UMAP output, with min_samples=10 and initial min_cluster_size=500. The 10- and 25-car tests select separate cuts of that fitted hierarchy. Representatives are actual non-noise cars, never UMAP centroids.

## FAISS cosine and ranking

For representative comparisons, categorical and numeric blocks are individually normalized, equally weighted, concatenated and normalized again. **Exact FAISS cosine distance** operates in these balanced original features, including mileage and both price features, rather than on UMAP coordinates. Each representative is the actual within-cluster sum-cosine-distance medoid. All members are considered for this objective.

Ratings map to -1, -1/3, +1/3 and +1. Soft HDBSCAN membership evidence has 75% weight; representative-distance affinity has 25%. Unassigned mass contributes zero. A candidate needs positive net evidence and must be strictly closer to a liked representative than a disliked one when both exist. Distance ties and rated prototypes are excluded. A separate native-categorical XGBoost ranker orders eligible cars; the one-hot change applies to the unsupervised profile/comparison features. Up to ten historical adverts are shown, not live stock.

## Verification and deployment

Source IDs, the original canonical-cohort hash, retained-ID digest and existing text-price export are checked. The release verifies mileage/text-price features, frozen future-listing transforms, real representatives, cosine/dislike safeguards and both complete Streamlit rating flows. Mechanical NDCG measures agreement with inferred labels, not independent customer satisfaction.

New listings use the frozen one-hot vocabulary and numeric ranges, then the saved UMAP transform and HDBSCAN membership prediction without a refit. Supply an existing `predicted_text_price` scalar when available; the recommender does not call MiniLM to create it. Missing feature values are flagged and imputed using reference medians. The GUI loads compact arrays and selected advert rows, not the heavy offline models.

Publish the matching code, both verified profile assets, SQLite catalogue and `data/taste-garage-release.json` pointer together. The previous catalogue remains the fallback until the new release is complete. Routine code checks do not overwrite new assets with old fixed-run models.


## Reciprocal rating standardisation
Session ratings use negative population-standardized reciprocals: score = −(1/r − mean(1/r))/std(1/r). Higher ratings always have higher scores. Equal ratings safely yield zero (the UI requires variation). Zero indicates the session average, rather than a fixed like/dislike threshold. Explicit nearest-liked/disliked exclusion still uses ratings 3–4 versus 1–2. Before integer ranking labels are formed, signed evidence is bounded and mapped to 0–3; no negative relevance labels reach XGBoost. Saved session recommendations are invalidated when this scoring version changes. Similarity models and trained catalogue assets are unchanged.


## Current RMSE / inverse release
Replaces vector-normalized cosine preference comparisons with per-feature RMSE on the original scaled mixed features: sqrt(sum((x-y)^2)/D), computed by FAISS IndexFlatL2. No unit normalization is applied; vector magnitude matters. Existing clusters, soft memberships and representatives are preserved. Rating evidence uses 5/12 − 1/rating: an inverse transform plus a fixed neutral offset between ratings 2 and 3, with no z-score, unit normalization or session-dependent centering. This keeps ratings 1–2 negative and 3–4 positive, with 4 best. Earlier standardized-reciprocal descriptions are superseded. RMSE measures feature differences, not dollar prediction error.


## Current StandardScaler release
FAISS vectors: sklearn StandardScaler is fitted per feature column on all 110832 catalogue rows. Saved means and scales transform the entire catalogue, its representatives and new queries. Constant columns have scale 1. There is no row unit normalization. Distances are RMSE in standardized feature units and the selected shortlist displays lowest RMSE first. HDBSCAN memberships and original representatives stay unchanged. The frozen scaler is saved as faiss-scaler.json; a matching offline feature encoder and scaler are exported together for new-listing inference.

Session ratings: inverse-transform all 10 or 25 ratings, fit a separate sklearn StandardScaler across that session, then negate, yielding −z(1/rating). Rating 4 remains highest; zero now means session average. Explicit nearest-liked/disliked safeguards still use ratings 3–4 versus 1–2. Earlier inverse-only/fixed-offset descriptions are superseded. No independently measured relevance improvement is claimed.


## Current release: cosine, no extra column scaling
FAISS uses existing encoded catalogue vectors directly; the added column StandardScaler and block normalization are disabled for comparisons. FAISS IndexFlatIP uses L2 normalization solely to calculate cosine similarity. Distance = 1 − cosine similarity. Reciprocal rating costs are 1/r: 4=0.25, 3=1/3, 2=0.5, 1=1, without session standardization. Signed preference evidence uses 5/12 minus cost. Selected recommendations display ascending cosine distance to the nearest liked representative. This supersedes the StandardScaler/RMSE release; existing unsupervised cluster geometry is unchanged.
