# Named advert types: categorical-token TF-IDF release

After **Estimate asking price**, the receipt shows an advert group, distinctive
keywords, membership strength and an explanation. These unsupervised groups are
independent of MiniLM + CatBoost AFT; neither asking nor auction prices are refitted.
They are not verified dealer/private labels or vehicle-quality verdicts.

## Description, categorical and numerical features

All 11 categorical model fields enter **the same TF-IDF representation as the
seller description**: manufacturer, model, condition, cylinders, fuel, title status,
transmission, drive, body type, paint color and state. There is **no OneHotEncoder
and no separate one-hot block**.

Each nonmissing value is an atomic field-qualified token, such as
`cat:model=qx70%203.7%20sport%20utility`. Field names prevent unrelated categories
from colliding; quoting preserves spaces, punctuation and digits. Missing/unknown
placeholders are omitted. Every observed categorical level in the fit sample,
**including a level occurring only once**, is retained in the vocabulary. Category
terms have `min_df=1` and no frequency cap. Truly unseen inference-time levels
cannot have fitted IDF values and are ignored by the frozen vocabulary.

Description terms remain English unigrams/bigrams, sublinear term frequency,
minimum document frequency 5, maximum frequency 95%, and up to 16,000 description
terms. This description-only vocabulary is unioned with **all categorical tokens**,
then one unified TF-IDF vectorizer fits IDFs over the resulting token documents.
Frequent categories therefore have lower IDF than rare ones. Rare categories can
influence neighborhoods substantially; rarity does not imply commercial relevance.

Year and mileage remain numerical. Up to 257 empirical **training** quantiles map
an observed number to percentile q, represented as `[q, 1-q]`. Missing/invalid
numbers contribute zero; eligible year is 1900–2021 and mileage 0–2,000,000 miles.
Values beyond fitted quantiles clip at endpoints. This is proximity scaling, not
a market-clock forecast or depreciation model.

Each block is L1 normalized. Combine **85% unified TF-IDF + 15% numerical**, then
L1 normalize the joint row. Weighted Hellinger distance is:

    H(p,r) = sqrt(1 - sum_j sqrt(p_j*r_j))

For two complete records, squared distance decomposes into the weighted sum of
blockwise squared Hellinger distances. Missing blocks redistribute available mass.
At least two recognized TF-IDF features are required; a description may be short
or absent when sufficient vehicle categories are supplied. A genuinely empty or
unrecognized input remains unclassified. No prices, text-price predictions,
residuals, VINs, contact details, coordinates or seller IDs enter the clustering.

## Validation tuning for broad native assignment coverage

The original goal was 95% native assignment coverage. **The user accepted 92.1%
validation coverage and stopped the search**. The released configuration is UMAP
5 neighbors, 2 dimensions, min_dist=0, and HDBSCAN min_cluster_size=80,
min_samples=5. No further settings are searched in the finalization run. This is
assignment coverage, not classification accuracy: there are no ground-truth
advert-type labels here.

### Accepted release results

The fixed configuration fits **20,000 training adverts** and finds **48 groups**.
The unified vocabulary includes **5,468 categorical tokens**, of which **3,526
occur only once**. Training coverage is 90.235%; the largest group contains
48.275% of the training sample.

| Frozen sample | Adverts | Native assignments | Coverage |
|---|---:|---:|---:|
| Validation | 1,000 | 921 | **92.1%** |
| Test, evaluated after selection | 1,000 | 922 | **92.2%** |

The original 95% goal was superseded by the user's acceptance of 92.1%. The
remaining density-noise adverts were not forcibly allocated to groups. These
figures are assignment coverage, not semantic accuracy.

The bounded search compares UMAP layouts `(neighbors, dimensions, min_dist)`:
`(5,2,0)`, `(10,5,0)`, `(30,5,0.1)`, `(60,10,0.1)`. Within each layout, HDBSCAN
compares `min_cluster_size` 15/40/80 and `min_samples` 1/2/5, with EOM extraction.
Smaller min_samples is less conservative. No epsilon merge is used because new
adverts must use a compatible frozen prediction path. Each layout fits on training
rows; validation adverts are projected into that fixed embedding once per layout.

Selection maximizes validation coverage among fits with **at least five groups
and no training group exceeding 70% of all training rows**. These guards prevent
a trivial one-bucket result but do not establish semantic quality. Stop early if
an admissible candidate reaches 97% validation coverage. Otherwise finish the
bounded search and report any unmet target honestly. **Evaluate the test sample
only after the candidate is selected; never tune parameters on test coverage.**

Native `hdbscan.approximate_predict` labels are counted as assigned even when
membership strength is low: the previous 0.30 rejection threshold is removed.
The GUI explicitly flags assignments below 0.30 as tentative. Native density noise
remains unclassified. **No nearest-centroid fallback, forced noise allocation or
synthetic “other” bucket is counted toward native coverage.**

Two-dimensional clustering layouts are included as an explicit experiment, not
merely a visualization. UMAP can alter density and introduce artificial splits.
A high coverage layout, especially a low-dimensional one, may produce broad or
unstable groups; no semantic-accuracy or stability claim is made. See the official
[UMAP document guide](https://umap-learn.readthedocs.io/en/latest/document_embedding.html),
[clustering caution](https://umap-learn.readthedocs.io/en/latest/clustering.html) and
[HDBSCAN parameter guide](https://hdbscan.readthedocs.io/en/latest/parameter_selection.html).

## Names and frozen serving

Names use centroid TF-IDF times positive smoothed log lift over the corpus, with
at least 8% within-group term support and overlap suppression. Field-qualified
categorical tokens are rendered as readable `field: value` keywords. The strongest
two phrases plus unique C-number form the name; six keywords appear in the GUI.
No external language model invents summaries or sends descriptions elsewhere.

Persist only numeric reference vectors, the fitted UMAP embedding and layout
constants. At inference, exact mixed Hellinger distances feed UMAP's precomputed
transform against that unchanged embedding, followed by HDBSCAN's frozen
prediction data. This portable representation avoids Python-specific compiled
NNDescent closures. SHA256 manifests verify compressed artifact parts; dependencies
are pinned and serving tests cover Python 3.12 and 3.13. Load only repository-trusted
joblib files, never visitor uploads. New results are cached within the visitor's
Streamlit session; no new analytics collection is introduced.

## Reproduce and inspect

```bash
python -m pip install -r requirements-clustering.txt
python -m scripts.prepare_all /path/to/vehicles.zip --output training-cache/all-cohort.jsonl
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m scripts.tune_advert_clusters training-cache/all-cohort.jsonl --finalize-known
python -m pytest tests/test_advert_clusters.py tests/test_advert_cluster_ui.py -q
```

The existing chronological 70/15/15 phases remain intact. Fit a deterministic
SHA256(seed:row_id) sample of up to 20,000 unique training descriptions. Holdout
samples contain up to 1,000 validation and 1,000 test descriptions, excluding any
normalized description seen in training. Empty descriptions are not removed by
an arbitrary word-count eligibility cutoff. The historical upstream cohort still
has its original price/date eligibility and VIN/text deduplication, so these
coverage figures are not estimates for every raw Craigslist row or future visitor.

`models/advert-clusters/report.json` records every parameter trial, cluster count,
largest-group share, training noise, validation/test coverage, target flags,
singleton categorical count and limitations. The parquet sample cache lives under
ignored training-cache and is not deployed. A 1,000-advert coverage estimate has
sampling uncertainty and cannot guarantee 95% of future traffic.

Source: Tom, [Used Cars Dataset](https://zenodo.org/records/10457828), **CC BY 4.0**.
The artifact's vocabulary and aggregate summaries are derived source data, not
anonymization guarantees. Common numeric/contact patterns are stripped from text,
but this does not guarantee removal of every personal name or address.
