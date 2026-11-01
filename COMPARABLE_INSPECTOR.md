# Comparable cars: SVD + exact FAISS cosine retrieval

The separate Listing Inspector workspace leaves regular and Bayesian pricing models unchanged. Select one of 1,000 specific historical vehicle examples, or enter your own car. Examples load year, make, model, mileage, available specifications and seller-description text. There is no make/model popularity selector.

## Representation and distance

Categorical inputs are make, model, body, fuel, transmission, drive and title status. Sparse one-hot feature-block weights are 1.4, 2, 1, 0.7, 0.7, 0.7, 0.7. Year and mileage are standardized using reference-set statistics and included at weight 0.2 each. Truncated SVD uses 32 dimensions, five randomized iterations and seed 42. Resulting vectors are L2-normalized.

Without description input, FAISS IndexFlatIP searches the unit SVD vectors directly. Inner product therefore equals cosine similarity. With description input, combine sqrt(0.8) times the unit SVD vector and sqrt(0.2) times the unit MiniLM vector, then L2-normalize the complete vector before the same exact FAISS search. Cosine distance is 1 minus the returned similarity. Scores are relative similarity, not probabilities of compatibility or valuation correctness.

There is no same-make, model-name, generation or year compatibility filter. Volkswagen can return Audi. Numerical attributes influence SVD geometry rather than excluding cars. Only invalid/zero representations and an explicitly excluded self-record are omitted. Search requests five neighbours directly, with no 500-candidate stage, LambdaRank, FM or second-stage reranking. Fewer results are returned only when fewer usable reference records exist.

## Missing features

Offline sklearn IterativeImputer fits chained native-categorical LightGBM classifiers for optional categorical fields and LightGBM regression for year/log-mileage. Original observations are restored after transformation. Unsupported all-missing categorical fields stay explicit unknown categories; a complete numerical matrix is not a claim that unknown facts have become known. Learned schedules are exported as portable compressed JSON containing LightGBM text models, not pickle.

Only retrieval vectors use inferred values. Cards retain source fields and name which retrieval inputs were estimated. Inferred title status, mileage or transmission is never a verified listing fact and never enters the production valuation model through this feature. Hashes bind imputation artifacts to the exact reference set. A held-out randomly masked-cell audit reports categorical reconstruction against a mode baseline; it does not establish performance under real missing-not-at-random patterns.

## Reference data

The larger asking catalogue uses up to 5,000 deterministic original training adverts, with historical prices and dates retained for comparison, common contact/VIN strings redacted and description excerpts capped at 1,200 characters. It falls back to the existing 400 historical asking examples until installed. Auction references are 4,000 sampled 2015 records with original auction dates and prices. Markets remain separate. References may overlap production-model training, are not an independent accuracy benchmark and imply no current availability.

Frozen int8 all-MiniLM-L6-v2 descriptions use up to 128 tokens. CPU reference vectors are precomputed as compressed float16 arrays. Integrity and row-order hashes are checked before use. The UI never downloads a model; unavailable text matching leaves categorical/numerical retrieval available.

## Evaluation

`benchmark_comparables.py` compares 32/64/128 dimensions for reference memory and local search latency. Cross-brand and distant-year results are diagnostics, not violations. The 32-dimension choice minimizes footprint; no judged relevance or NDCG superiority is claimed. No price is used to fit or search the similarity index.

Tests cover exact agreement between FAISS and brute-force cosine scores, the five-result limit, cross-brand retrieval, price invariance, invalid inputs, self exclusion, portable imputation equivalence, original-fact preservation, complete presets and UI reruns. CI verifies real MiniLM inference and builds compact reference assets before publication.


## Historical prices adjusted to the valuation date

Comparison cards retain the original observed asking/auction price and observation date, then apply the existing depreciation age/mileage ratio and the ratio of the aggregate used-car market index at the valuation and observation dates. The default valuation date follows today. Optional mileage normalization puts each reference at the entered car’s mileage; turning it off holds each reference’s stated mileage fixed. Neither assumption tracks the physical vehicle’s subsequent use or condition. National retail used-car CPI is also a wholesale proxy; when observations lag, the existing damped market forecast is disclosed. Actual prices are never similarity features. Unsupported age ranges or missing source prices/dates/mileage yield an explicit unavailable adjustment, not a guessed current price. These scenarios do not have validated current-vehicle accuracy or prediction coverage.
