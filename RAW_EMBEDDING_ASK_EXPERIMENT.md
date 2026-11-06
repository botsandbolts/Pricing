# Neural Lab experiment: Embeddings only, full neural price prediction

This study follows the user's explicit correction: use pretrained sentence
embeddings as the only description representation; do not convert them
into text-price estimates. The price comes from full DeepCross+AFT.

## Candidates

- Frozen all-MiniLM-L6-v2, 128 tokens, embedding-only control
- Frozen all-MiniLM-L12-v2, 128 tokens
- Frozen BGE-small-en-v1.5, 128 and 256 tokens
- Frozen GTE-small, 128 and 256 tokens
- Frozen all-DistilRoBERTa-v1, 128 and 256 tokens

Eight variants use a model revision pinned to the concrete Hugging Face
SHA recorded at run start. Token truncation is explicit. Inference
uses CPU models, not a remote text-inference API.

## Data and features

The identical 2021 original asking-price 211,127-car cohort from
neural-ask-frozen-features, run 42224932982. Original chronological
train/validation/test: 147,777 / 31,668 / 31,682. Model fitting
uses 133,000 training cars; the final 14,777 training cars serve as a
training-only early stopper. Original validation chooses the encoder
and token length. Test is scored only for the selected winner.

Non-text inputs are unchanged: seven numeric values and eleven
categorical fields with embeddings. Price mentions are removed
from text with scripts.common.clean.

The only text input is a frozen 384- or 768-dimensional L2-normalized
sentence embedding. No Ridge head, price scalar, predicted text price,
TF-IDF, SVD, or price-driven encoder training is used. A trainable
LayerNorm -> Linear(embedding_dim,96) -> GELU -> LayerNorm -> Dropout(0.08)
adapter feeds the 12%-dropout DeepCross AFT model.

DeepCross alone learns a two-output exact-event log-normal price location
and scale under AFT-style likelihood. There is no invented sale censoring.

## Robustness and leakage prevention

Frozen embeddings are computed without price labels. Every encoder
passes 24 sample batch-1 versus batch-16 embedding consistency checks.
Embeddings and canonical holdout identities have SHA256 fingerprints.
Original numeric/categorical arrays are reused directly.

Eight candidates emit original validation predictions only. Once each
row ID and price label matches the old benchmark, one encoder is
selected by lowest validation dollar MAE. Only that winner is evaluated
on untouched original test. Two other independent seeds are assessed
on validation, without selecting a lucky production seed.

Reference: former DeepCross+MiniLM price-scalar validation MAE 3,046.61
and original test MAE 3,261.54 USD. Full XGBoost aggregate test
3,377.73 USD. The former scalar model is a benchmark, not an input.

## Production and privacy

This benchmark does not automatically replace production or the
Experimental Models alpha. The winner must pass a future complete
Streamlit serving and checkpoint parity check before release.
Model files remain local; no new model database.
The benchmark does not write to Turso. End-user Neural Lab predictions
continue to use existing consent-gated analytics with model/version labels,
no descriptions, no embeddings.

Workflow: .github/workflows/raw-embedding-ask-challenge.yml
Final report: reports/raw-embedding-ask/comparison.json


## Ten-way parallel embeddings and practical compute budget

Each of the 8 encoder/context candidates is partitioned into exactly 10
disjoint, original-index-contiguous row chunks: 80 independent embedding
jobs total, approximately 21,112–21,113 listings each. At most 30 GitHub
Actions chunk jobs run in parallel. No individual chunk encodes all listings.

Every chunk validates 24 randomly chosen batch-1 versus batch-16 embeddings,
pins the Hugging Face model source commit SHA, redacts prices, and publishes
only a float16 embedding matrix and an audit manifest. The assembler
for each encoder refuses missing/duplicated/misaligned chunks, verifies
unchanged original row fingerprints, per-chunk file SHA256, encoder revision
consistency across all ten pieces, and reinserts chunks at their exact
original array offsets. Once assembled, the existing original DeepCross
AFT model trains on the full 133,000 fit rows. No change to train/holdouts.

Operational job timeouts are 70 minutes per chunk, 15 minutes for assembly,
70 minutes for each neural training arm, 65 minutes for final report and
robustness checks, and 10 minutes for code tests. These limits bound
individual jobs; GitHub runner queueing and repeated matrix waves can
still make the entire workflow longer than four wall-clock hours.
Failed/expired chunks fail closed rather than synthesizing predictions.
