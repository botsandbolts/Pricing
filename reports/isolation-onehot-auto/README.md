# Full-data one-hot Isolation Forest results

Default automatic threshold: score_samples < -0.5. No cars were removed; deployed pricing weights are unchanged.

| Dataset | Cars | Encoded columns | Flagged | Minimum score | Median score | Maximum score | OutlierTree flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| ask | 211,127 | 24,966 | 0 | -0.369324 | -0.324974 | -0.304176 | 31 |
| bid | 550,181 | 1,034 | 0 | -0.373744 | -0.326729 | -0.304374 | 74 |

Zero flags means no score crossed the automatic threshold. It is not proof of clean data. Sparse one-hot encoding, high cardinality and small per-tree subsamples can weaken useful anomaly signals; actual detection quality needs reviewed labels or a controlled validation experiment. There are no verified anomaly labels here, so precision, recall and accuracy are unknown.

OutlierTree at pct_outliers=0.099 flagged 31 asking and 74 auction records; agreement intersections with Isolation Forest are both zero. No target percentage was imposed. Full-data descriptive results include validation/test and must not filter held-out evaluation.

See [methodology](../../ISOLATION_ONEHOT.md), per-market reports for score quantiles and most anomalous examples, and provenance.json for source/version/checksum metadata. Full score exports are generated locally by the reproducible script and excluded from git.
