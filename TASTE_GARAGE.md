# Drive Chemistry
Rate a few cars. Find your type. Skip the awkward small talk.

## For drivers
Drive Chemistry is the car-discovery experience inside Car Price Studio’s “Pick your garage” panel. It is free to explore without registration. Choose Speed Date (10 cars) or The Long Drive (25 cars), react to every car from 1 — No thanks to 4 — Love it, then build a shortlist of up to ten other historical cars.

Each taste-test car is a real advert representing a learned group. Rate the whole car, including its price. The longer test offers finer profiles; it is not a guarantee of better recommendations. You can switch tests without losing the ratings saved for that test and catalogue version.

The current discovery order is **largest cosine distance first** to the nearest representative you liked. It puts more different selected cars ahead of closer ones. It is not “best match first.” Every displayed car has already passed the preference checks. This order follows the requested product setting.

Open “Why this car?” for the positive and negative taste signals, related profiles and distances. Open the original advert excerpt for specifications. “Did we find your type?” collects additional ratings for a limited check of the displayed ordering. Those ratings do not change the list already displayed.

## For businesses
The experience demonstrates discovery from a small amount of explicit preference feedback, without user IDs, demographics or a history of clicks. Vehicle groups are prepared offline; session reactions personalize filtering and ranking. It can illustrate alternatives in a classified-listing or dealership journey.

The catalogue contains retained historical Craigslist asking adverts, not live stock. Price displays are date-adjusted asking-price scenarios at recorded mileage, not verified transactions, binding offers or guaranteed present-day values. Changing the date updates the scenario and ranking price features; it does not refit the vehicle groups. Seller claims and condition are not verified.

Broad clusters and omitted preferences can produce poor matches. Coverage and automatic ranking scores do not prove customer satisfaction, conversion uplift or commercial accuracy.

## Technical design
### 1. Reference data and features
The current catalogue uses 110,832 retained Lemon Squeezer training/calibration cars; validation and test rows of the pricing experiment are excluded. Common nominal categories use one-hot indicators. Rare, missing-if-rare and unseen values share a per-field Other bucket. The seven numeric columns include year, log mileage, mileage-missing, log November-2026 asking-price scenario, scenario-missing, stored cross-fitted MiniLM description-price scalar and its missing indicator. Numeric values use frozen robust range scaling and imputation. No raw text embeddings, newly computed transformer prices or preference ratings enter clustering.

### 2. Soft vehicle groups
HDBSCAN fits directly in the encoded feature space. Weighted Manhattan distance equals:
- 50% of the fraction of categorical fields that differ;
- 50% of the mean absolute difference across bounded numeric features.

For F categorical fields, each one-hot indicator has weight 0.25/F: changing a category flips two indicators, counting one mismatch. Each of N numeric columns has weight 0.5/N. There is no additional column StandardScaler, reduced-dimensional embedding or full all-pairs distance matrix. The KD-tree Boruvka path uses Manhattan distance, min_samples=10 and an initial min_cluster_size=500. Supported hierarchy cuts produce separate 10/25-profile tests. The workflow never invents clusters to meet a requested count.

Soft memberships are nonnegative and sum to at most one. Remaining mass is unassigned. Density noise can still have soft affinities; it is not forced into a hard group. The saved encoder and distance weights transform new cars for membership prediction without refitting HDBSCAN.

Representatives are real non-noise members, selected by minimum summed cosine distance among their group’s original encoded vectors. They are examples, not synthetic average cars.

### 3. Personal preference
Raw ratings preserve their direction. Signed evidence is r − 2.5, giving −1.5, −0.5, +0.5, +1.5. There is no reciprocal transform or session StandardScaler. Soft group evidence receives 75% weight and representative-distance affinity receives 25%. Unassigned membership adds no preference evidence. Candidates need positive net evidence. When both liked and disliked representatives exist, the nearest liked representative must be strictly closer than the nearest disliked one; ties fail. Taste-test representatives are excluded.

### 4. Similarity, ranking and presentation
FAISS IndexFlatIP computes exact cosine after the L2 normalization necessary for cosine. Inputs are original encoded feature vectors without added column scaling or block normalization. Distance = 1 − cosine similarity. Smaller distance means more similar.

A session XGBRanker uses rank:ndcg and ndcg@10, native categorical partitioning, max_depth=3 and two CPU threads. Predictors are vehicle attributes and original/date-adjusted log asking-price features. User, demographic, cluster and prototype IDs are excluded. Each original car appears once; there is no Cartesian expansion.

Inherited signed preference evidence is bounded and shifted to nonnegative relevance. Ranking labels are quantized in 0.1 steps; displayed inherited-label NDCG uses unrounded relevance and linear gains. A deterministic vehicle-ID holdout of roughly 20% supports early stopping; representatives stay in training. The final ranker fits every catalogue row. It selects up to ten eligible cars, and the display then sorts this selection by descending nearest-liked cosine distance. Display sort and learned ordering are separate.

### 5. Evaluation, privacy and release
Held-out NDCG measures agreement with weak preferences inherited from the same representative reactions, before display sorting. It is not independent satisfaction or catalogue-wide relevance. Optional additional ratings check only ordering within the displayed cars.

Ratings and the trained personal ordering stay in the Streamlit session. No registration is required for discovery. The broader app’s analytics and cookie controls apply separately.

New catalogue assets publish only after source-alignment checks, soft-membership/new-point checks, full-catalogue ranking and both real Streamlit rating flows pass. UI revisions preserve stable control keys, ordered mobile cards, accessible reaction labels, progress announcements and mode-specific rating memory.

Data: Craigslist Used Cars, Tom / Zenodo 10457828, CC BY 4.0. Price scenarios use saved depreciation curves and the CPI/Holt market clock.
