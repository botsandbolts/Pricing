# pricing

A description-only used-car price app with free-text notes, vehicle-characteristic dropdowns and a USD estimate. The pricing pipeline is:

1. `embaas/sentence-transformers-multilingual-e5-large` encodes each description as a 1,024-dimensional dense vector.
2. Isolation Forest scores **training vectors only**. Exact score ranking removes the most anomalous **50%** of training candidates.
3. An **Extra Trees regressor (200 trees)** learns `log1p(asking price)` from the retained half. The app reverses the log transform to display dollars.

There are no TF-IDF features, Ridge regression or separate make/model embeddings. Make and model remain ordinary form facts written into the description. No other predictors are fitted.

## Run

```bash
python scripts/fetch_assets.py
python -m http.server 8501 --bind 0.0.0.0 --directory dist
```

Open http://localhost:8501. Codespaces starts port 8501 automatically. On first inference the browser downloads roughly **562 MB** of quantized E5-large weights. A modern desktop browser with sufficient memory is recommended. Your vehicle text is processed locally in a Web Worker and is not sent to Hugging Face or an inference API. No API key is needed.

## Text model

The requested embaas checkpoint uses XLM-RoBERTa, 512-token windows, mean pooling and normalisation, outputting 1,024 dimensions. All 391 shared source tensors were verified bit-identical to the reference behind the public ONNX export. The reference has one additional deterministic position-index buffer. See `dist/model-provenance.json`.

The app supports up to 8,000 input characters, including form facts, with 6,500 characters available for notes. Each E5 window includes its `query: ` prefix and special tokens within the 512-token budget. All windows are encoded, averaged and normalised. This is chunk pooling, not an 8,000-character attention window. The encoder is frozen. CPU and browser quantized kernels can produce different numerical predictions.

## Isolation Forest: exactly 50%

The pilot has 1,200 chronological training candidates. Isolation Forest uses 300 trees and only description embeddings—no prices, validation data or test data. We sort scores deterministically and keep exactly the least anomalous 600 rows, removing 600. Score ranking makes the removal count exact even if scores tie.

All 300 validation and 300 test listings stay **unfiltered**. Extra Trees hyperparameters are selected by validation USD MAE. Test rows are used only for the final evaluation. The 80% indicative error band uses absolute validation log residuals; actual test coverage is reported.

An anomalous description is not evidence of a wrong listing. Removing half the training data may exclude valid rare vehicles, unusual conditions or writing styles, and can worsen accuracy. The app reports the measured error of this requested configuration.

## Pilot results and limits

See `dist/evaluation.json`: 600 retained training rows, 300 validation and 300 test rows. The canonical cohort contains 10,892 eligible/deduplicated listings; the pilot is sampled independently within its chronological partitions. It is not directly comparable with earlier full-cohort evaluations.

The source contains historical US asking prices from April–May **2021**, restricted to $1,000–$150,000. This is not a current-market valuation, verified transaction-price model or separately validated multilingual-market model. No inflation adjustment is applied. Original seller descriptions are evaluated; GUI-generated descriptions are not independently validated. Estimates are rounded to $50.

The browser uses exported regression trees. Numeric parity checks verify that the serialized forest matches sklearn on fixed embeddings. The Isolation Forest is a training filter; the browser does not reject a new form submission as an outlier.

## Reproduce

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training.txt
python scripts/download_models.py /path/to/model-cache
python scripts/verify_weights.py /path/to/model-cache
python scripts/prepare_dense.py /path/to/vehicles.zip
python scripts/train_forest.py --models /path/to/model-cache
node scripts/check.mjs
```

For the full canonical cohort, run `prepare_dense.py ... --full` before training. The filter will still remove exactly half of training candidates. Large-model CPU encoding is expensive. Embedding generation checkpoints can resume the same corpus/configuration. The source dataset and transformer binaries are not stored in the repository.

To refit trees from existing E5 vectors without re-encoding, run `python scripts/fit_forest.py`. Filtering scores and retained/removed row identifiers are recorded in the ignored `training-cache/isolation-filter.csv`. CPU reference test predictions are recorded there as well.

Source: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data
Snapshot: https://zenodo.org/records/10457828
Requested encoder: https://huggingface.co/embaas/sentence-transformers-multilingual-e5-large
Verified ONNX export: https://huggingface.co/Xenova/multilingual-e5-large
Transformer/tokenizer license: Apache-2.0. ONNX Runtime Web: MIT.
