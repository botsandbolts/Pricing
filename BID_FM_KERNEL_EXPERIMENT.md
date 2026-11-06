# Wholesale auction model challenge — factorization machines and fast SVM-like regression

## Executive briefing

**Question:** Can a classical nonlinear method outperform XGBoost on historical wholesale auction sale prices without the cost and complexity of a full neural FT-Transformer?

The experiment tests (1) regularized second-order factorization machines,
(2) *actual field-aware factorization machines* (each category embedding depends
on its partner field), and (3) random Fourier features with epsilon-insensitive
or Huber SGD regression, a scalable approximation to RBF-style SVR. All use the
original five numerical and ten categorical **pre-sale** features.

These are **observed 2014–2015 wholesale auction sales normalized to the 2015
dollar basis**, not submitted-but-unaccepted dealer bids, asking prices, or
current-year auction sales. No seller descriptions were available to this
cohort.

### Why these models?

XGBoost learns nonlinear splits and has an impressive benchmark around
USD 1,801 original-test MAE (full-data) and USD 1,920 for the frozen
Lemon-cleaned training policy. These are different training regimes.
FM models transfer information across sparse combinations of vehicle make,
model, drivetrain, location, and transmission instead of memorizing every
specific interaction. Field-aware FM allows each field to behave differently
depending on its interaction partner. RFF approximates smooth similarities
without the prohibitively large full RBF kernel matrix.

Lower test MAE is the goal; reduced compute, reproducibility and honest
generalization are additional product criteria. Prior numerical values
are benchmarks only and do not imply improvement.

## Frozen data / identities

Use original run 42249111756 artifact named neural-bid-frozen-features,
with exact source fingerprints and VIN-disjoint chronological partitions:

- Full original training: **384,198** unique auctions
- Frozen 25%-Lemon-trimmed training: **288,148** records
- Original untrimmed validation: **82,380**
- Original untouched untrimmed test: **83,603**

All candidate models keep the full original validation and test records.
For FM and RFF, training uses the earlier 90% of the selected training
cohort; the final 10% is an internal chronological early stopper.
Only the external validation partition selects architecture/rank, cohort,
and blend weight. The original test is evaluated for **one validation winner
only**, after every candidate validation file is audited by row ID, split,
price label and frozen SHA256.

The original numeric coordinates are model year, auction vehicle age,
log odometer, log odometer per vehicle age and pre-sale condition grade.
The ten native dictionary-encoded categoricals include manufacturer, model,
body type, transmission, state, fuel, drivetrain, title status, cylinders,
and paint. These source encodings were fitted on the original train split,
not target labels or any future holdout.

## Prespecified 10-arm benchmark

| Class | Parameters | Fit cohort |
|---|---|---|
| Second-order FM | Rank 16, rank 32 | Full and Lemon |
| True field-aware FM | Rank 8 | Full and Lemon |
| Random Fourier + Huber SGD | 512 RBF features | Full and Lemon |
| Random Fourier + epsilon-insensitive SGD | 512 | Full |
| Random Fourier + epsilon-insensitive SGD | 1,024 | Lemon |

### Mathematical FM

For the one-hot active categorical entries and continuous numeric
coordinates x, second-order FM models

y_hat = b + sum_i w_i x_i + sum_{i<j} (v_i dot v_j) x_i x_j.

We also include *univariate numeric hinge splines* for nonlinear ageing,
mileage and vehicle-condition main effects. The latent interactions still
use the five original numerical fields and ten categorical fields. FM
training uses Huber loss on log observed auction dollars, with positive
price retrieval via exponentiation and bounded clipping.

For true field-aware FM, a category's latent vector depends on the field
of the feature with which it interacts. The pair score is

(v_i,field(j) dot v_j,field(i)) x_i x_j,

rather than the usual (v_i dot v_j) x_i x_j. Here there are exactly
15 fields and 105 feature-field pairs, with an 8-dimensional latent space.
Categorical embedding table dimensionality is 15*8 per categorical field.
This is **not merely field weighting**; both partner-specific embeddings
are fitted. No model ever sees individual auction IDs as predictors.

### Approximate kernel regression

Numeric features are standardized on fitting rows only; unknown categories
are handled by train-fitted sparse one-hot encoding. A seeded random
Fourier feature map approximates an RBF kernel on their mixed feature
space (gamma=0.5). SGDRegressor optimizes either a Huber or
epsilon-insensitive objective on *centered log prices*, with an
explicit internal chronological stopper. This behaves similarly to a
large-scale SVR approximation without constructing the full N*N Gram matrix.
It is NOT an exact libsvm/SVR solver. The experiment also publishes the
training time and feature count, so the trade-off is visible.

## Ensemble and honest residual policy

The benchmark tests

P_blend = (1-alpha) * P_XGBoost-Lemon + alpha * P_selected_FM_or_kernel

for prespecified alpha in {0,0.1,0.2,0.35,0.5,0.75,1}, chosen only on
validation. **This is convex blending, not a separately fitted residual
regressor.** The true residual version would require original training-row
out-of-fold predictions from XGBoost, made without those rows' labels.
In-sample residuals are not accepted as a substitute.

Results show an independent standalone winner and the validated blend of
*that same* winner. An exploratory best-any-model validation blend may
also appear as validation-only diagnostic, but is never silently
test-evaluated if its model differs from the standalone winner.

## Audit & operational ceiling

- Ten independent GitHub Actions training jobs with max 10 parallel.
- Check job max 20 minutes; individual training job max 95 minutes;
  report max 50 minutes.
- The check stage creates a four-hour study deadline, handed to every job.
  Jobs stop without training when deadline budget is exhausted.
- Each script also has its own training-time budget, independent of GitHub
  timeout. GitHub queueing may delay a run, but cannot justify silently
  training longer than the declared budget.
- Missing jobs, changed holdout row IDs, altered price labels, or incomplete
  predictions cause the report to fail closed.
- The original XGBoost Lemon model is compared on exact row pairs.
  The published full-data XGBoost is clearly marked **aggregate and unpaired**.
- Selection and reporting do not modify or deploy production bid models.

### Output when completed

reports/bid-fm-kernel/comparison.json
reports/bid-fm-kernel/validation-leaderboard.csv
reports/bid-fm-kernel/validation-blends.csv
reports/bid-fm-kernel/winner-original-test.csv.gz

Workflow: .github/workflows/bid-fm-kernel-benchmark.yml.
