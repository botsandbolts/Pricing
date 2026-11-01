# Pick your garage: ten reactions, vehicle-only recommendations

## Business guide

The new horizontal Pick your garage panel is separate from valuation and the three neighbour-search SVDs. It shows ten real historical asking adverts representing different vehicle profiles. Rate each from 1 (very bad) to 4 (very good), then press Find my ten cars. At least two ratings must differ. The result is a ranked ten-car shortlist with specifications, original dates/prices and date-adjusted asking-price scenarios. These historical cars are not live inventory.

The ten actual ratings propagate to other adverts assigned to each profile. XGBoost learns from vehicle features and advertised prices. There is one row per original advert, no prototype-by-catalogue Cartesian expansion, user IDs, demographic data, personal features or collaborative model. Ratings and fitted-session results are not saved in a user database or sent as analytics parameters. Changing a rating invalidates the previous shortlist until the user rebuilds it.

The displayed held-out NDCG@10 measures agreement with inherited profile labels. It is not a measured personal satisfaction rate: hundreds of thousands of labels can still come from only ten actual judgments. Optional additional ratings evaluate ordering within the ten shown recommendations; they are not used to train that displayed shortlist and do not establish catalogue-wide recall.

## Offline profiles

The source is the complete deduplicated canonical Craigslist asking cohort, from the checksum-verified original Zenodo archive. All source partitions contribute to this separate exploratory catalogue; production train/test results and models remain unchanged. Native density noise is retained and identified, not silently deleted.

Clustering preprocessing uses one-hot representations of vehicle categories and standardized year, log mileage and a mileage-missing indicator. Optional categories have explicit missing levels; missing numerical values use catalogue medians. TruncatedSVD (up to 32 dimensions, five randomized iterations, seed 42) reduces the sparse matrix, followed by L2 normalization. This is preprocessing for the separate recommendation experiment, not a change to the three existing neighbour SVDs. Advert prices, seller descriptions and all personal attributes are absent from clustering and ranking predictors.

UMAP fits the full canonical catalogue in five dimensions, cosine input distance, 30 neighbours, minimum distance 0.05, 100 epochs, seed 42, one deterministic optimization thread and low-memory mode. HDBSCAN uses Euclidean UMAP distance, leaf selection and min_samples=10. Its initial min_cluster_size is max(30,N/100); if it supports fewer than ten dense groups, progressively smaller thresholds are tried. Failure to support ten groups blocks publication rather than inventing groups.

If more than ten native leaf groups exist, Ward clustering merges their centroids into ten macroprofiles. The interface and manifest disclose this aggregation. Original HDBSCAN noise is assigned to the nearest macroprofile centroid and remains flagged as exploratory. Each displayed prototype is the actual non-noise advert nearest the macroprofile centroid. It is not an averaged, fabricated car. UMAP geometry and density grouping do not establish that every car in a profile has the same appeal.

## Session ranker

Ratings map from 1–4 to relevance 0–3. Each advert inherits its assigned profile's rating. The structured predictors are manufacturer, model, body, fuel, transmission, drive, title, condition, cylinders, paint colour, state, model year, log mileage and a mileage-missing indicator. Native XGBoost categorical partitioning uses max_cat_to_onehot=1; no one-hot matrix is fed to the ranker. Original log advertised price, date-adjusted log price and an adjustment-missing indicator are additional ranker predictors. Prices remain excluded from UMAP/HDBSCAN. Cluster IDs, prototype IDs, descriptions and qid are excluded from predictors.

XGBRanker uses objective='rank:ndcg', eval_metric='ndcg@10', hist trees, depth 3, learning rate 0.07, lambda 20, alpha 1, min_child_weight 20, max_bin 64, max_cat_threshold 32, top-k LambdaMART pairs with 10 pairs per sample, two CPU threads and seed 42. An all-zero qid marks the single session's ranking query group; it is not a user feature. Up to 96 trees and 12-round early stopping limit overfitting and compute.

A stable SHA-256 vehicle-ID split reserves roughly 20% for early stopping, with the ten directly rated prototype rows kept in training. Validation is on inferred cluster labels, not independent real preferences. Feature vocabulary and unsupervised geometry use the full catalogue: this is a transductive experiment, not an external benchmark. After selecting the iteration count, a final ranker fits each original advert exactly once and scores the full catalogue. The rated prototypes are excluded from the ten recommendations. The UI reports exponential-gain NDCG@10, a deterministic random-order baseline, and the actual judgment count.

## Deployment and limits

UMAP/HDBSCAN never fit inside Streamlit. Offline CI exports compact categorical/numeric feature arrays and an indexed, compressed SQLite catalogue. Serving reads only the ten displayed records. Artifacts have SHA-256 hashes and a 90 MiB SQLite size gate. No pickled UMAP object or full description matrix is loaded at runtime. Shared caches contain immutable catalogue data; ratings and results remain in session state. CPU fit and peak process RSS are recorded by a full-catalogue artificial-rating smoke test, clearly labelled as a mechanical check rather than observed preference quality.

Date-adjusted price scenarios use the existing depreciation curves and saved CPI/Holt clock. National retail CPI also serves as a market proxy, and contemporary per-car accuracy is unvalidated. Missing source facts or unsupported age ranges produce unavailable date adjustment rather than invented prices. Data attribution: Tom, Used Cars Dataset, Zenodo 10457828, CC BY 4.0. Common contact/VIN/email strings are redacted from 100-character description excerpts.
