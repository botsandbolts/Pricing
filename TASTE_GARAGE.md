# Pick your garage · HDBSCAN + cosine

The production recommender reuses the existing HDBSCAN models, original real-car representatives and soft memberships for the 10- and 25-car tests. There is no UMAP transform or clustering retraining.

## Business guide
Rate each representative from 1 (No thanks) to 4 (Love it). Soft memberships propagate positive and negative reactions. FAISS cosine comparisons check whether a car is closer to a liked representative than a disliked one. Eligible cars have positive net evidence; XGBoost orders them. Prototypes are excluded from recommendations. Historical adverts are not live inventory.

## Technical guide
Frozen smoothed leave-one-out category encodings and standardized numeric features feed HDBSCAN directly. Exact FAISS IndexFlatIP operates on L2-normalized original vectors; distance is 1 minus cosine similarity, between 0 and 2. Zero vectors produce neutral similarity zero. Direction matters more than magnitude. Inputs are copied before normalization. The same cosine distances feed preference affinity, exclusion and displayed explanations. The medoid objective and existing cluster geometry remain Euclidean; neither is refit.

Ratings map to signed evidence -1, -1/3, +1/3, +1. Density evidence has 75% weight; representative-distance affinity has 25%. A nearest-disliked representative blocks a recommendation, including distance ties. A separate vehicle-only XGBRanker uses rank:ndcg and ndcg@10; no user attributes or Cartesian expansion. Original and date-adjusted log asking prices enter ranking. Mechanical held-out NDCG is agreement with inherited preferences, not measured user satisfaction. Optional additional ratings provide a limited direct check.

## Release checks
Reuse saved original 10/25 profile artifacts. Recompute cosine matrices only. Verify full-cohort ranking, new-listing prediction, source integrity, liked/disliked safeguards and both real Streamlit rating flows before publishing assets. Changing the manifest refreshes cached catalogue and rating state. No heavy HDBSCAN model loads in the GUI.
