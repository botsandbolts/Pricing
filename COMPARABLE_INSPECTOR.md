# Listing Inspector: comparable vehicle retrieval

The separate horizontal Listing Inspector panel preserves the existing regular and Bayesian pricing paths. It adds evidence-linked advert checks, seller questions, two historical comparison sets, and a user-assumption purchase contribution sandbox.

## Comparable retrieval

Sparse float32 one-hot encoding covers make, model, body, fuel, transmission, drive and title status. Missing values have an explicit marker; unknown query categories are ignored. Feature-block weights are 1.4, 2, 1, 0.7, 0.7, 0.7, 0.7. Randomized truncated SVD (seed 42; five iterations) maps these into 32 dimensions. Vectors are normalized.

Ranking = 0.45 categorical cosine + 0.30 model-label string similarity − 0.15 capped year distance/5 − 0.10 capped mileage distance/100,000. Separate hard guardrails require same make, no more than five model years apart, and model-label similarity >=0.45. Unknown makes receive no results. These are explicit engineering choices, not optimized relevance weights. Exact attributes remain visible. Generation/trim identity is not guaranteed.

When a valid offline description index is installed and a query has text, add 0.10 MiniLM cosine similarity. The frozen installed int8 all-MiniLM-L6-v2 encoder sees up to 128 tokens. Reference vectors are stored float16 in compressed NPZ, loaded without pickle, with content and row-order hashes. Queries are encoded on CPU with a bounded cache; no UI downloads. Semantic failure falls back to attribute matching. Auction records have no descriptions.

The asking reference set uses up to 5,000 original training adverts when its compact catalogue is installed, falling back to 400 manufacturer-balanced historical Craigslist training examples (April–May 2021; prices and common personal identifiers removed) and 4,000 sampled auction records from 2015. They are separate markets, may overlap training, and are not live listings. Prices never enter retrieval. Auction prices are shown with their historical dates; asking prices are unavailable in the curated text sample.

## Evaluation and selection

`python -m scripts.benchmark_comparables` compares 32/64/128 dimensions. The recorded engineering audit checks filtering, reference-vector footprint and local latency. All had zero guardrail violations; 32 was selected for memory, not demonstrated relevance superiority. Search averaged approximately 7 ms for asking references and 35 ms for auction references on the development machine, excluding first-load imports, encoding and UI rendering. Total benchmark-process peak RSS was approximately 192 MiB, not incremental app memory.

`tests/test_comparable_retrieval.py` tests target-price invariance, invalid queries, unfamiliar categories, self exclusion, semantic dimensions, real reference guardrails, UI submit and reruns. CI builds real MiniLM vectors and verifies actual semantic UI use. No judged NDCG, precision gain or price accuracy improvement is claimed. No FM or LambdaRank is used. Similar target prices are not relevance labels.

## Remaining scope

This release does not implement calibrated bargain probabilities, duplicate detection, regional-effect estimation, information-premium experiments or batch appraisal queues. Existing Bayesian uncertainty and valuation metrics belong to their own models, not this retrieval index. Rule-based advert checks can miss negation and complex language. Contribution calculations use user assumptions and exclude overhead and tax; they are not predicted realized margins.


Retrieval first selects up to 500 nearest SVD vectors, then applies explicit matching scores and same-make/year/model guardrails to return up to five eligible cars. No exact count is fabricated for a small or unsupported reference pool.
