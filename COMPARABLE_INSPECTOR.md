# Comparable cars: three SVDs and exact FAISS cosine search

## Business guide

Choose one of 1,000 specific historical cars, mixed between asking adverts and auction records in a stable shuffled order, or enter your own car. The complete editable form prices it with the same installed regular XGBoost or Bayesian models used on the landing page.

| Mode | Reference cars | Features after completion | Results |
|---|---|---|---|
| Ask | Asking adverts | Vehicle attributes; optional MiniLM text similarity | Five asking adverts |
| Bid | Wholesale auction records | Vehicle attributes and an imputed MiniLM description-price signal | Five auction records |
| Hybrid | Both populations | Vehicle attributes only | Five asking and five auction neighbours |

There are three separately fitted SVDs. Hybrid fits one joint vehicle representation, searches each market subset in those same coordinates and merges its results in descending cosine similarity. Asking records are purple; auction records are blue. Bid means a wholesale auction sale record, not a live buyer offer.

Each neighbour shows its original price/date, a price scenario adjusted to the valuation date, model-year and mileage differences, differing specifications, and a price difference against your car's estimate for the same market. Expand the individual record for every compared specification and imputation provenance. The default valuation date is today. Similarity scores are not confidence probabilities. Cross-brand neighbours are allowed; there is no make/model/year compatibility exclusion, second-stage ranker, LambdaRank or FM.

## Prices help complete missing facts, then leave the SVD

For Ask and Bid, adjusted historical prices are auxiliary predictors in an offline sklearn IterativeImputer with LightGBM. The price column is removed before SVD. Completed mileage or categories may therefore depend on price context, but price is never a direct similarity dimension. Original observed facts are always preserved. The query uses its model estimate as auxiliary context when supported, on the saved imputer's declared date basis; without a supported query price, the imputer uses its training median context. Inferred fields remain estimates, never verified vehicle facts.

Hybrid has its own vehicle-only imputer. No price, description, MiniLM vector or description-price signal enters hybrid completion or geometry.

The only price-derived scalar allowed in an SVD is Bid's description-price proxy. Auction descriptions are genuinely absent: their MiniLM price signal starts as NaN, not zero or fabricated text. A shared small LightGBM chained imputer transfers information from observed asking-description prices, vehicle attributes and reference-2021 price context. The two installed asking MiniLM heads are averaged in dollars. The imputed auction signal is an asking-style statistical proxy, not an observed auction description or bid. Ask uses optional raw MiniLM semantics, without a scalar description-price SVD column.

## Technical recipe

Categories: make, model, body, fuel, transmission, drive and title status. Sparse one-hot block weights: 1.4, 2.0, 1.0, 0.7, 0.7, 0.7, 0.7. Year and log(1+mileage) are standardized with reference statistics and weighted 0.35 each. Bid alone can add standardized log description-price and its missing indicator, weighted 0.50 and 0.10. No actual advert/auction price column is accepted into SVD.

TruncatedSVD uses up to 32 dimensions, five randomized iterations, seed 42 and two CPU threads. Vectors are L2-normalized. FAISS IndexFlatIP returns exact inner products, hence cosine similarity. Distance is 1 minus cosine similarity. In Ask mode with available query text, separately normalized SVD and MiniLM vectors are concatenated with square-root weights for an 80%/20% inner-product mixture and normalized again. Hybrid never attaches a text block. Search directly returns five records from the selected market; hybrid performs two five-neighbour searches in the same projection. Only invalid/zero representations and an explicitly excluded source record are omitted.

Price-assisted completion uses small native-categorical LightGBM classifiers in a three-iteration categorical chain, followed by LightGBM regressors in an up-to-five-iteration numerical chain. Observed cells are restored. Entirely unobserved categories remain unknown. Portable compressed JSON contains fitted LightGBM text trees and the exact sklearn chain schedule, with reference and artifact hashes; serving does not fit or load pickle. The artifact records its price-basis date. Prices shown to users are independently transported to their chosen date, not read from imputed values.

## Historical prices and limits

The asking reference catalogue retains up to 5,000 original training adverts; the auction catalogue contains 4,000 historical records. Prices, dates and source facts are retained. The 1,000-example selector targets 500 from each market and uses stable hash shuffling. References can overlap production training and are not an independent accuracy benchmark or live inventory.

Date adjustment uses existing cross-sectional age/mileage depreciation curves and aggregate retail used-car CPI; retail CPI is also a wholesale proxy. The existing damped Holt forecast covers the lag after the latest published index observation. Optional mileage normalization compares prices at your entered mileage; otherwise each reference's recorded mileage stays fixed. Neither scenario knows subsequent vehicle usage or condition. Unsupported ages or missing original price/date/mileage produce an explicit unavailable adjustment. No contemporary vehicle-level accuracy or interval coverage is claimed. Asking/auction gaps are not dealer profit.

Tests check exact FAISS versus brute-force cosine agreement, five-result bounds, cross-brand retrieval, self-exclusion, mixed-list sorting, price exclusion from SVD, hybrid price/text invariance, observed-fact preservation and portable imputer equivalence. CI checks real MiniLM inference, all three UI modes and installed feature contracts. No judged relevance, NDCG improvement or customer preference accuracy is claimed for this unsupervised neighbour search.
