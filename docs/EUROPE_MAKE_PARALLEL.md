# European deduplication: 12 parallel make-pure workers

Workflow file: .github/workflows/europe-make-parallel.yml

Partition the European asking-price snapshot by **normalized vehicle make**, not by arbitrary row number. This ensures all listings of the same make, including across countries and marketplaces, belong to exactly one partition.

The GitHub Actions workflow:

1. Runs synthetic unit/integration tests for shard balancing and global-source-ID restoration.
2. Downloads the original Parquet data **once**, counts by normalized make, and greedily balances complete makes across 12 bins.
3. Uploads twelve independent temporary Parquet artifacts containing original rows plus their immutable global row indices. Source artifacts have 1-day retention.
4. Runs **12 concurrent matrix jobs**, each downloading **only its own shard**. Each performs Granite 97M R2 embeddings, FAISS HNSW retrieval, conservative vehicle-identity linkage, and optional high-information thinning. Runtime parallelism depends on the GitHub account's available runner slots.
5. Uploads audit metadata and reversible global-row mappings, not raw vehicle listing files.
6. Aggregates all reports, verifying exactly-once global row coverage and reporting total identity merges, similarity thinning, review pairs and per-country results.

The full aggregate artifact contains combined_audit.json and global_training_mapping.csv. Every input has a zero-based global row ID, a canonical identity representative, a training representative and a Boolean keep decision.

**Similarity-thinned records are NOT proven duplicate vehicles.** The extra training subset is an experimental sampling approach, reversible and distinct from confirmed same-listing ID merges. Compare it with the conservative dataset on an identical held-out split with MAE, CVaR95, calibration and per-country checks.

Source snapshot licensing and underlying marketplace permissions need review before redistribution or commercial deployment. This workflow does not change the live American ASK/BID models, application dependencies or production UI.

## Local CLI

Partition: python -m pricing.europe_make_shards partition --input /data/europe.parquet --output /data/parts --shards 12

Link one shard: python -m pricing.europe_dedupe --input /data/parts/shard-00.parquet --output /data/audit-00 --granite --thin-review-candidates --no-canonical-export

Map original IDs: python -m pricing.europe_make_shards map --input /data/parts/shard-00.parquet --audit /data/audit-00

The workflow runs the preceding operations for all 12 makes-balanced partitions, then collects them in one global report. Cross-brand typos or brand reclassification cannot be matched by per-make blocking; a separate cross-brand integrity pass would be needed.
