# European used-car record linkage (offline research)

Scope: [KTHESAT/european-used-car-market-2m-listings](https://huggingface.co/datasets/KTHESAT/european-used-car-market-2m-listings) (October 2026 snapshot; 2,003,645 advertised listings, nine countries). The uploader labels the dataset **CC BY-NC-SA 4.0**; third-party marketplace extraction permissions have **not** been independently verified. Do not redistribute raw rows, derivatives or model weights without legal/provenance review. No dataset is committed to this repository.

This is a **candidate deduplication audit**, **not** a trained price model, confirmed vehicle-identity oracle, or automatic permission to use this data in production.

## Method

1. Read CSV or Parquet in batches into on-disk SQLite; preserve immutable original source row numbers and listing IDs.
2. Unicode-aware normalization of make/model/version, mileage, fuel, power, seller locality and registration; power_hp=0 is treated as missing.
3. Indexed source-ID, mileage/registration and locality blocking. Candidate-pair budget is finite; a cap is reported explicitly.
4. Optional [IBM Granite Embedding 97M Multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) 384-dimensional embeddings of **unique** normalized make/model/version strings (cached with dataset/model digest), with FAISS HNSW semantic neighbours. These suggest additional pairs across trim naming variants, but never prove identity.
5. Hard contradiction checks and conservative automatic merges: same source + listing ID when vehicle identity is compatible, or different source + identical full registration date + same city/country + exactly matching normalized version + equal power + odometer within 25 km + no conflicting fuel/gear/color.
6. Cluster coherence checks prevent transitive chains from joining incompatible cars. Keep original records, reversible mapping, automatic-pair evidence, review candidates and audit JSON.
7. Full canonical export preserves the original columns (Parquet to Parquet, CSV to CSV). A sampled run intentionally skips canonical export to avoid passing off a partial sample as a complete cohort.

**No one-hot/SVD, hashing or semantic embedding automatically deletes cars.** Lookalikes stay distinct unless the identity evidence is unusually strong. Review pairs are not removed.

## Install and run

From the repository root:

~~~bash
pip install -r requirements-europe-dedupe.txt
# Download only after reviewing dataset licence and marketplace terms.
python -m pricing.europe_dedupe \
  --input /path/to/train-00000-of-00001.parquet \
  --output /path/to/eu-linkage-2026-10
~~~

Optional semantic stage; installed separately so the live Streamlit app and US model memory footprints stay unchanged:

~~~bash
pip install sentence-transformers faiss-cpu
python -m pricing.europe_dedupe \
  --input /path/to/train-00000-of-00001.parquet \
  --output /path/to/eu-linkage-granite \
  --granite --embedding-cache /path/to/reusable-granite-label-cache.npz
~~~

The Granite encoder currently runs through SentenceTransformer in offline CPU/GPU batches. **INT8 ONNX is NOT enabled or claimed here**; perform numerical parity checks separately before changing inference backends.

For a **nonrepresentative smoke test only**, add --sample 20000. It selects the first 20k rows, does not export a canonical cohort, and MUST NOT be used to estimate the overall European duplicate rate.

Synthetic QA: pytest -q tests/test_europe_dedupe.py.

## Outputs

- **row_mapping.csv**: original row to canonical row, source + ID and cluster size.
- **automatic_pairs.csv**: merge edges and explicit reasons.
- **review_pairs.csv**: candidates for human review, never automatically merged.
- **canonical_listings.parquet** (or CSV): all original fields from retained representatives.
- **audit.json**: counts, candidate cap and method metadata.
- **linkage.sqlite**: local intermediate; do not publish.
- **granite_labels.npz**: optional private embedding cache.

The audit reports safe_to_train=false until provenance, manually labelled precision/recall and leakage checks pass. Group the subsequent training/validation/test split by resolved canonical vehicle identity. Preserve longitudinal asking-price histories separately; a canonical listing is not a transaction or realized sale.

## Before downstream training

Annotate a stratified random sample of automatic pairs, review pairs and hard negatives (same make/model/year/city and close mileage). Measure auto-merge precision with confidence intervals, duplicate-pair recall, by-source/by-country false merges and rare-model outcomes. Evaluate sensitivity to matching thresholds and whether one marketplace dominates. Maintain a manual reversal path for every auto-merge.

There are no VINs or actual image bytes in the published dataset schema, so true physical identity cannot always be established. Verify first-registration date conventions, same-ID recycling, seller relocations and missing values. Do not treat similarities as definitive proof of identical vehicles.

Existing American ASK/BID models, dependencies, navigation and Streamlit deployment remain unchanged.
