# pricing

A used-car pricing GUI with make/model/year dropdowns, mileage, title and condition selectors, equipment checkboxes and up to 6,500 characters of free-text notes. It produces a historical US asking-price estimate in dollars and compares three real fitted models. No API key is needed. Inputs are processed locally in a browser Web Worker; nothing is sent to a valuation API.

## Run

```bash
python scripts/fetch_assets.py
python -m http.server 8501 --bind 0.0.0.0 --directory dist
```

Open http://localhost:8501. In Codespaces, port 8501 opens automatically. The first estimate downloads the 279 MB ONNX encoder from Hugging Face and loads the 11 MB runtime; later estimates reuse the loaded session. A modern browser with WebAssembly is required.

## Model

* Full cleaned text: unigram/bigram TF-IDF + Ridge.
* Meaning: frozen XLM-RoBERTa sentence-transformer paraphrase-multilingual-mpnet-base-v2, using its int8 ONNX export, mean pooling and unit-normalisation per chunk.
* Long context: up to **8,000 input characters** split into consecutive **126-token chunks**, with the XLM-R special tokens `<s>` / `</s>` added. Every chunk is read, including content after character 1,000. Chunk embeddings are averaged and normalised. This is chunk aggregation, not an 8,000-character attention window. There is no 128-token cutoff.
* Combined model: concatenated TF-IDF and 768-dimensional sentence embedding + Ridge, predicting `log1p(price)`. `expm1` produces the displayed dollar estimate. The combined model is the primary output; the other two estimates are shown for comparison.
* GUI facts are converted into English listing text. They are not separately trained tabular features or hand-coded price adjustments. Form-generated descriptions can differ from the original seller descriptions used in evaluation.

## Evaluation and uncertainty

See `dist/evaluation.json` and the app's Model results tab. The same 10,892 filtered/deduplicated historical listings are split chronologically 70/15/15. Vocabulary/IDF and Ridge weights use training only; alpha is selected by validation USD MAE. The exact int8 encoder used by the browser is used to generate training and evaluation embeddings. The earlier 128-token float encoder's results must not be presented as this model's results.

The indicative 80% band comes from the 80th percentile of absolute log residuals on validation listings. Its actual test coverage is reported. It is not a certified per-vehicle confidence interval or guaranteed sale range. Prices are rounded to $50 to avoid false precision.

The source is April–May **2021 US asking-price data**, restricted to $1,000–$150,000. This is not a live-market valuation, a German-market model, or a verified final-sale-price model. Newer vehicles and predictions outside that range are flagged. No inflation correction is applied. Regex money redaction and exact-text/VIN deduplication reduce leakage but do not eliminate near duplicates or every price expression.

## Reproduce training

Requires Python packages in `requirements-training.txt` and the public `vehicles.zip` snapshot. An optional second argument can reuse the prior benchmark's TF-IDF vectorizer and check held-out row identity. Run `python scripts/fetch_assets.py --training` to download the encoder and tokenizer files for training.

```bash
python scripts/train_long.py /path/to/vehicles.zip
node scripts/check.mjs
```

The script can verify held-out row identity against the baseline, trains all three long-context models, and exports browser-readable weights, the evaluation and parity fixtures. A Python/scikit-learn process is not needed to run the GUI.

Source dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Public snapshot: https://zenodo.org/records/10457828
Encoder: https://huggingface.co/Xenova/paraphrase-multilingual-mpnet-base-v2 (Apache-2.0; see upstream model card and license).
Runtime: ONNX Runtime Web 1.22.0 (MIT).

Trained model weights are included as JSON. The large public encoder/runtime binaries are fetched by setup rather than copied into the GitHub repository. The hosted site packages the tokenizer/runtime locally and downloads the pinned public XLM-RoBERTa ONNX weights from Hugging Face; no vehicle text is sent there.

## Verification

Tokenizer and TF-IDF predictions are checked against Python fixtures. The previous GUI QA covered custom model entry, year extrapolation guidance and the evaluation tab. The XLM-RoBERTa browser test downloaded the pinned 278,647,663-byte encoder and encoded a 1,344-character German description as three chunks into a unit-normalised 768-dimensional vector. Tokenizer IDs matched Python for English, German, Hebrew, French and Chinese fixtures. CPU and WebAssembly int8 kernels can give small numerical differences. Current tokenizer and end-to-end browser verification is recorded with the XLM-RoBERTa migration. Browser inputs are never sent to a remote inference server.

## XLM-RoBERTa architecture

Despite its `paraphrase-multilingual-mpnet-base-v2` model name, this multilingual sentence-transformer uses `XLMRobertaModel` as its encoder (12 layers, 768 hidden dimensions). The encoder is frozen; Ridge is retrained on its embeddings. It is not a MiniLM model and not an end-to-end fine-tuned XLM-R price regressor. Long notes are read in consecutive 126-token chunks, matching the sentence model’s original 128-token window including special tokens. There is no single-window truncation of user notes.

The larger model needs materially more browser memory and first-download bandwidth. Multilingual embeddings do not establish multilingual pricing accuracy; this training/evaluation dataset is historical US listings.

Encoder ONNX files under `dist/encoder/onnx/` are ignored by Git and are only needed for local retraining. Browser inference downloads the pinned public XLM-RoBERTa weights directly. Large tokenizer files are downloaded by `fetch_assets.py` during Codespaces setup.
