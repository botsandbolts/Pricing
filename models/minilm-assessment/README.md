# Installed MiniLM + LightGBM

This is a working CPU inference bundle from the **completed chronological assessment** in
`botsandbolts/notebooka6bded45ca`, Kaggle saved version **362555957**.
It is not the separate, unfinished full-data deployment refit.

* Fine-tuned `sentence-transformers/all-MiniLM-L6-v2`: 148,290 training descriptions,
  two epochs per fit, three-fold OOF description-price stacking.
* Isolation Forest with `contamination="auto"` removed 38,681 training rows using only
  their OOF `predicted_text_price`; LightGBM fitted 109,609 retained training rows.
* Validation and test: 31,777 cars each, unfiltered and chronologically later.
* Text head: standardized log(1 + asking price), converted back to dollars.
* Selected LightGBM: 999 trees, standardized dollar target, Huber alpha 1.5.
* Recorded test MAE **$3,559.07**, RMSE **$7,637.47**; description-only MAE **$3,688.31**.
* Labels are historical April–May 2021 **asking prices**, not confirmed sales or current values.

The bundle includes the real saved transformer weights, regression head, tokenizer,
selected assessment LightGBM, evaluation and provenance. The category vocabulary
is recovered directly from that LightGBM file's `pandas_categorical` metadata in the
original `pricing.tabular.CATEGORICAL` order. No category vocabulary is guessed or refitted.
The original assessment omitted a saved calibration radius; this installation therefore
returns point estimates and explicitly unavailable intervals rather than inventing a band.

The 90,864,192-byte safetensors payload is losslessly gzip-compressed and split into
parts of at most 8 MiB for repository transport. `weights.json` checks individual parts,
compressed contents, decompressed weights and configuration. At first load it reconstructs
a normal local Hugging Face directory under `~/.cache/pricing` (override with
`PRICING_CACHE_DIR`). No Hugging Face download, GPU or Kaggle connection is needed for inference.
`manifest.json` records the exact file checksums and the incomplete full-refit status.

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Select **MiniLM + Isolation Forest + LightGBM**, open **Value a car**, enter a seller
description and estimate. All six explanation graphs remain available. Graphs hold the
submitted text-price signal fixed across scenario cars; they do not explain individual words.
MiniLM + Isolation Forest + LightGBM is now the default studio choice; the regularized ensemble remains selectable for comparison.
