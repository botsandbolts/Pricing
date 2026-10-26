# Larger regularized LightGBM ensemble

This upgrade targets the final **Pricing asking-price model**, while retaining the age/CPI-adjusted CarBids signal as an input. It does not convert asking labels into verified retail transactions and does not retrain a sentence transformer.

## Training design

Eight fixed candidate configurations compare log1p and dollar targets, Huber and absolute-error losses, 63/127/255 leaves, depth limits, stronger leaf-size constraints and regularization. Each candidate trains up to 3,500 trees with early stopping based on validation dollar MAE. Row bagging is actually enabled (`bagging_fraction=0.85`, `bagging_freq=1`); feature fraction is 0.85. L1/L2 penalties, categorical smoothing and category L2 penalties discourage fragile splits. No Extra Trees or Ridge model is used.

Additional features include log mileage, mileage per model-year age, missingness indicators, exact make/model and make/body combinations, and training-fitted category frequencies. CarBids age, mileage, market-time and partially pooled relative-value features remain included. Category vocabularies and frequencies are learned only from the training partition during assessment. No observed asking price, sale outcome or result text is added as a predictor.

The best three configurations by validation MAE enter a nonnegative, sum-to-one mixture search on a 0.05 weight grid. Single-model solutions are allowed. Selected configurations are each repeated with **three predeclared seeds (42, 137, 777)**, using their selected tree counts; seed predictions are averaged in dollars. This is a LightGBM ensemble, not a claim that more trees always improve accuracy.

Assessment uses the original chronological partitions: 148,290 train, 31,777 validation, 31,777 test. Test outcomes do not choose hyperparameters, tree counts or mixture weights. Promotion to the default asks for better validation/test MAE and no worse test RMSE than the original tabular baseline. **This historical test benchmark was inspected in earlier project experiments. It is not a fresh external replication.** The data cover a short listing period, so market-drift robustness is unproven.

After assessment, fixed member configurations/counts/weights are refitted on all 211,844 eligible records. The original historical baseline and earlier structural-transfer experiment remain available. The shared GUI uses the larger ensemble only when its promotion flag is true. Model IDs distinguish the active ensemble from the old baseline, and the GUI cache notices newly pulled manifests.

## Explanations and uncertainty

Different members may predict on different target scales. Their final dollar predictions are blended. Native TreeSHAP is displayed **per member on that member's own target scale**; log-price and dollar contributions are not incorrectly summed into a single causal explanation. Members' disagreement is not a calibrated confidence interval. Historical validation-residual bands do not establish future-market coverage.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m marketbridge.robust --cohort training-cache/full-cohort.jsonl --structural models/structural-transfer --output models/robust-asking
```

This frozen comparison is for the original full cohort whose SHA is recorded in `models/transfer-asking/manifest.json`. Rebuild the baseline before comparing a different dataset. `train_robust_kaggle.ipynb` checks this cohort identity and embeds the frozen upstream model/source. It needs CPU and Internet, with **no neural epochs or GPU requirement**. Completed candidate fits are cached; interrupted seed averaging or final refits restart that stage. Save Kaggle outputs and resume with identical source/data. The notebook has an eight-hour session budget.

Search caches are intentionally excluded from Git. The deployed members, fitted feature schema, manifest, complete candidate grid and evaluation are committed to both integrated repositories. To reproduce training exactly, use the frozen package and recorded data version rather than silently refreshing the CPI snapshot.

## Capacity versus reliability

More leaves and trees can model richer interactions but can also overfit, increase memory use and slow inference. Regularization, validation stopping and averaging are safeguards, not guarantees. A broader future holdout, new sale labels, improved vehicle-generation information and better input quality may contribute more than additional model capacity.

## Trained result and active deployment

The winning **nine-member ensemble** has **18,324 trees**. Its configurations receive weights 0.50 / 0.30 / 0.20, each split equally over three seeds. The selected maximum leaf counts are 255 / 127 / 127. All selected members use log1p-target Huber loss; the dollar and absolute-error candidates remain recorded in the search report.

| Historical benchmark | Original baseline | Winning ensemble |
|---|---:|---:|
| Test MAE | $3,328.66 | $3,288.05 |
| Test RMSE | $6,839.35 | $6,664.99 |
| Validation MAE | $3,140.08 | $3,084.65 |

Both promotion criteria were met. Pricing's model selector now defaults to **Best trained ensemble**. Its normal Value a car form and integrated comparison use the promoted bundle. The original tabular baseline and untrained MiniLM branch remain separate options. CarBids' shared comparison uses the same winning asking ensemble; its historical auction model is unchanged. All figures above are asking-price benchmarks before all-row deployment refit, with the reused-test limitation described above.

The compressed deployed bundle is approximately **145.6 MiB**. This is intentionally much larger than the baseline, so clone/pull, cold loading and inference take longer. No GPU is needed to serve it. Earlier per-car result downloads are versioned snapshots from before this upgrade; regenerate batches to obtain the active ensemble's predictions.

## Lossless model storage

The GitHub connector limits requests to 16 MiB. The gzip weights are therefore split into verified binary parts of at most 8 MiB, with SHA-256 checksums in each member's manifest entry. The loader rejoins the exact bytes before decompression. This changes neither trees nor numeric predictions; it is not pruning or quantization. To pack a newly trained whole-file bundle, run `python -m marketbridge.model_storage models/robust-asking`. The original training module bytes for the recorded training fingerprint are archived in `reports/robust-training-source.txt`; the later multipart loader is an inference/storage extension.
