# European listing deduplication: 50 concurrent make-and-year workers

The superseded 12-way make-only workflow has been replaced by a 50-way
make-and-year workflow. GitHub Actions uses a shared concurrency group
to request cancellation of the old run, and also explicitly calls GitHub's
run-cancel API for the old run ID.

## Strict block key

Each original listing is assigned to one **normalized make + year** block.
The year is read, in descending precedence, from production_year,
manufacturing_year, manufacture_year, production_date, model_year, year,
reg_year or first_registration. Missing/unparseable years enter a distinct
unknown-year group. The manifest exposes counts by year source; if the
dataset has only reg_year, that is explicitly a **first-registration proxy,
not a verified production year**.

All rows with the same normalized make and chosen year stay together in
exactly one balanced shard. The SQLite identity and semantic matching joins
explicitly enforce that their segment keys match. Thus no pair may cross
the make/year block, even where one shard contains many different blocks.
Brand aliases (e.g. VW and Volkswagen) share the same normalized key.

The first preparation job downloads the Parquet snapshot once and uses
two streaming passes to count make/year groups and generate 50 partition
Parquet artifacts. Each has its original zero-based row position and block
key. Fifty independent GitHub matrix workers process their assigned
partition using Granite 97M multilingual embeddings, FAISS HNSW and the
reversible information-completeness thinning rule (max-parallel=50).
Available GitHub runner concurrency limits may reduce actual simultaneity.

The final aggregation job verifies that every source row appears once,
no candidate cap was reached and all 50 shard counts match the manifest.
It emits combined_audit.json and global_training_mapping.csv.

### Interpretation

Identity-canonical deduplication and optional high-information training
thinning are kept separate. Similar looking listings removed by thinning
are **not** verified instances of the same physical vehicle. No permanent
listing data is committed to the repository; temporary input artifacts
have one-day retention. Raw and canonical input datasets are unchanged.

Only the research branch and GitHub Actions workflow change. Existing
American production app and ASK/BID models are untouched.
