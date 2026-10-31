# Five-fold CatBoost CPU experiment

The command below trains **five group-held-out models per market**, plus a calibration-only refit for the chronological validation and test cohorts. It writes one log-dollar residual per vehicle, held-out dollar MAE and log RMSE. Native CatBoost categorical handling is used; no one-hot or target encoding is fitted.

```
python -m scripts.prepare_full training-cache/ask.zip --output training-cache/full-cohort.jsonl --workers 4
python -m carbids.data --input training-cache/bid.zip --output training-cache/bid-cohort.jsonl
python -m scripts.train_catboost_fivefold --iterations 1000
```

The original source archives must be placed at the indicated paths and checked against the source hashes in `models/forest-original/{ask,bid}/provenance.json` before training. The asking target is the observed **advertised** price: exact-dollar CatBoost `SurvivalAft:dist=Normal;scale=1.0` labels `[P,P]`, with log-price modeled internally. It is not a confirmed sale or a fabricated censoring interval. The auction target uses `Quantile:alpha=0.5` on log CPI-reference dollars. No residual trimming is applied. Both losses run on CPU because SurvivalAft does not support CatBoost GPU training.

`training-cache/catboost-fivefold/{ask,bid}` receives `fold-1.cbm` through `fold-5.cbm`, `calibration-refit.cbm`, `predictions.csv.gz` and `evaluation.json`. These are **research artifacts**, not an automatic replacement for the live MiniLM models. Inspect untouched test MAE, subgroup error and interval coverage before promoting anything. No scores are reported until the command finishes on real original cohorts.
