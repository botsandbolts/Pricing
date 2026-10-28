# MiniLM + CatBoost AFT default

The asking model includes the dynamically computed `predicted_text_price` in
its actual fitted feature schema. MiniLM reads only the seller description;
make/model remain categorical vehicle facts. CatBoost is newly refitted on
158,331 retained Craigslist cars, using separate full-cohort OOF features.

The auction weights are exactly the previously deployed CatBoost-AFT weights,
trained on 412,626 retained auction records. The Kaggle/tunguz auction source
has no free-text descriptions. This branch is explicitly tabular.

| Asking assessment | Cars | MAE USD | RMSE USD |
|---|---:|---:|---:|
| Filtered validation | 23749 | 1842.63 | 2933.67 |
| Filtered test | 23760 | 2001.21 | 3133.51 |
| Original unfiltered test | 31682 | 3308.86 | 6976.90 |

Filtered scores are optimistic because the previous requested residual masks
used validation/test outcomes. The original test is a previously examined
benchmark, not a fresh external test. These assessment scores are not an
independent test of the separate all-retained-data deployment refit. This
release does not claim to outperform every historical model in the repository.

See [architecture and reproduction](../../MINILM_CATBOOST.md). Checksum-verified
parts store the final CatBoost models. MiniLM weights are shared with the
existing `models/minilm-cleanlab/text-model` bundle and checked by provenance.
