# Named advert types after Estimate

The asking-price receipt now has a **Your advert’s tribe** card. It groups the
submitted description and vehicle facts with similar historical Craigslist adverts.
This is an additional unsupervised description classification, independent of the
MiniLM + CatBoost AFT price. It does not filter cars or enter either price model.
Auction data have no descriptions: this feature belongs to the asking calculator.

## Mixed-feature geometry

The workflow follows the official [UMAP document embedding guide](https://umap-learn.readthedocs.io/en/latest/document_embedding.html)
and its Hellinger distance, extending it to three nonnegative feature blocks:

| Block | Weight when all available | Representation |
|---|---:|---|
| Description | 65% | English unigram/bigram TF-IDF, sublinear term frequency, min_df=5, max_df=0.95, at most 16,000 terms |
| Common categories | 20% | One-hot make, model, condition, cylinders, fuel, title, transmission, drive, body, color and state |
| Numeric vehicle facts | 15% | Model year and odometer, expressed as training-fitted quantile ranks |

Category vocabularies are learned **only from the fit sample**, retaining a level
only if its frequency is at least max(30, ceil(0.005 × fit rows)). Missing labels,
rare levels and unseen values contribute zero: there is no catch-all rare-category
indicator. Each recognized categorical field has equal mass within its block.

Numeric values use at most 257 empirical training quantiles. Each observed number
becomes a two-column distribution **[q, 1−q]**. This retains numeric ordering without
letting 100,000 miles drown out all language dimensions. Each observed numeric field
has equal mass within the numeric block. Missing/invalid numbers are ignored;
valid values beyond the fitted quantiles clip to endpoints. Year eligibility is
1900–2021 and mileage 0–2,000,000 miles, matching the calculator’s historical domain.
This is a proximity feature, not a time-series depreciation estimate.

Within each block, features are L1 normalized. Concatenate the weighted blocks and
normalize the resulting row to sum to one. UMAP uses:

    H(p,r) = sqrt(1 − sum_j sqrt(p_j r_j))

For complete records, the squared distance is exactly the weighted sum of the
three blockwise squared Hellinger distances. When a block is absent, its mass is
redistributed to available blocks; comparison with incomplete records also reflects
their different available mass. At least five description words and two recognized
TF-IDF features are required: numbers alone cannot produce an advert-type verdict.

Prices, predicted_text_price, residuals, dates, VINs, contacts, coordinates, and
seller identity are **not clustering features**. Preparation still uses the
existing price-eligibility range to define the historical cohort; this is not a
claim that the source population is unselected. Text normalization strips HTML,
URLs, email-like tokens and numeric identifiers/monetary figures; it is not a
guarantee that all personal names or addresses are removed.

## Discovery, naming and new-car assignment

UMAP maps the mixed vectors into **10 dimensions**, n_neighbors=30, min_dist=0,
seed=42. HDBSCAN uses Euclidean distance in that embedding, min_cluster_size=40,
min_samples=10, and prediction_data=True. No clustering is fitted on a 2D plot.

Names use mean within-cluster TF-IDF multiplied by smoothed positive log lift over
the corpus. Terms must occur in at least 8% of members; overlapping unigrams and
bigrams are suppressed. The strongest two phrases plus a unique cluster ID form
the name. Six distinctive keywords are shown in the GUI. A real embedded member
nearest the centroid is identified by hash in the report; its original sentence
is not published as a representative testimonial or fabricated summary.

At inference the **same fitted** vectorizer, category encoder and numeric quantiles
produce exact mixed Hellinger distances to the reference sample. UMAP's
precomputed-distance transform places the advert in the **unchanged fitted
embedding**, feeding [HDBSCAN approximate_predict](https://hdbscan.readthedocs.io/en/latest/prediction_tutorial.html).
This avoids storing Python-version-specific compiled NNDescent closures. Numeric
reference vectors, embedding and UMAP layout constants are preserved instead;
the new-point layout uses the same UMAP implementation and pinned version.
The partition is frozen: it does not relearn on visitors. Noise and match strength
below 0.30 are shown as **Unclassified advert**, rather than forcing a label.
That threshold and the block weights are transparent heuristics, not validated
seller-classification optima. Membership strength is **not a calibrated probability**.

UMAP can distort density and create artificial splits, as its
[clustering guide warns](https://umap-learn.readthedocs.io/en/latest/clustering.html).
Groups may reflect dealer boilerplate, manufacturer wording, region, vehicle type
or condition. They must **not be relabelled dealer/private without manual labels
and a separate supervised evaluation**. A vocabulary label is not a verified fact
about the submitted car.

## Reproduction and diagnostics

Source: Tom, [Used Cars Dataset](https://zenodo.org/records/10457828), CC BY 4.0.
Retain attribution for the derived vocabularies and aggregate summaries.

```bash
python -m pip install -r requirements-clustering.txt
python -m scripts.prepare_all /path/to/vehicles.zip --output training-cache/all-cohort.jsonl
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m scripts.train_advert_clusters training-cache/all-cohort.jsonl
python -m pytest tests/test_advert_clusters.py tests/test_advert_cluster_ui.py -q
```

Default fit: deterministic SHA256(seed:row_id) sample of up to 20,000 unique cleaned
descriptions within the existing chronological training phase. `--max-train 0`
requests all eligible unique training descriptions and can be substantially slower.
Numeric transforms, common-category selection, TF-IDF, UMAP, clusters and naming
are all fitted from this training sample. No original validation or test rows are
used to fit them. Separate deterministic samples of up to 1,000 validation and
1,000 test adverts diagnose frozen assignment coverage; any normalized description
seen anywhere in train is excluded from these diagnostics.

`models/advert-clusters/report.json` records sample sizes, noise, holdout coverage,
common-category vocabularies, names, weights and dependency versions. Coverage is
**not accuracy**: no ground-truth advert-type labels exist. There is no claim of
seller classification precision/recall, price-MAE improvement or seed stability.
These require additional experiments, not an interpretation of the coverage count.

### Installed run, 29 October 2026

The selected 20,000 training descriptions yielded 19,998 usable vectors, from
145,703 available unique normalized training descriptions. HDBSCAN found **86
groups** and marked **38.23%** of fitted vectors as noise.

| Frozen chronological sample | Adverts | Strong assignments | Coverage |
|---|---:|---:|---:|
| Validation | 1,000 | 429 | 42.9% |
| Test | 1,000 | 455 | 45.5% |

Median membership strength among assigned adverts was 0.865 on validation and
0.938 on test. These are coverage diagnostics, not accuracy or calibrated
confidence. Terms such as *Carvana*, *Autonation / Confirm Availability*, and
*Engine Functioning / Functioning Properly* illustrate how groups capture
retailer templates as well as car language. This does not establish who posted a
new advert. The exact names, sizes and other keywords are in the report.

## Serving and privacy

The trusted repository artifact is compressed joblib in checksum-verified parts,
with a SHA256 manifest and
matching pinned dependency versions. Never load user-uploaded pickle/joblib files.
The bundle contains numeric reference features/embeddings and aggregate vocabulary,
not a CSV of original descriptions or source listing IDs. Reference vectors and
vocabulary remain derived source data, not anonymization guarantees. New visitor
descriptions are processed locally; cluster results are cached only in the current
Streamlit session, with no new analytics events or external summarization service.

Serving imports the clustering stack only after a saved estimate exists. A missing
bundle, version mismatch or clustering failure leaves the dollar receipt available
and shows a short unavailable message. Initial UMAP/Numba compilation on a cold
server can take longer than subsequent requests; a trained model is never refitted
during an Estimate click. The existing ask and bid weights are unchanged.
