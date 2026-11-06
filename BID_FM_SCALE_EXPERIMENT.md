# Expanded wholesale FM model comparison — full training versus Lemon

This is a separate follow-up to the completed 10-arm FM, field-aware FM, and
RBF approximation benchmark. The goal is **scaling factorization model
capacity on the original auction bids**, not making claims from a subset.

Original 2014–2015 VIN-deduplicated and time-ordered observed auctions:
384,198 full original training examples, 288,148 original Lemon-kept training
examples, original 82,380 validation rows and 83,603 untouched test rows.
The source is the authenticated frozen artifact from GitHub run 42249111756.

The **same original holdouts** are used for ALL models. The original training
cohort's final 10% is used as an internal stopping set and does not leak test
prices or original validation prices into model weights. Cohort choice (full
versus Lemon) is a model-selection hyperparameter and is selected using
original validation dollar MAE only.

## Twenty-six registered factorization arms

| FM architecture | Latent ranks | Train cohorts |
|---|---|---|
| Standard 2nd-order FM | 16, 32, 64, 128 | full, Lemon |
| True Field-Aware FM | 8, 16, 32 | full, Lemon |
| HOFM order-3 with ANOVA interaction sums | 16, 32 | full, Lemon |
| Field-pair-gated FM | 16, 32 | full, Lemon |
| DeepFM (FM plus small jointly fitted MLP) | 16, 32 | full, Lemon |

All experiments use identical, frozen, original five numeric and ten
categorical pre-sale auction fields, with no free text, model-year price
lookups, labels-as-features, or hierarchy.

Standard FM is the low-rank equivalent to a second-degree polynomial
interaction model. True FFM learns **different category embeddings for
different interaction-partner fields**, not simply field-wide weights.
HOFM3 adds an explicit third-order interaction through the elementary
symmetric polynomial formula, computed in O(n_features * rank). The
field-pair-gated model starts from ordinary FM and fits separate
upper-triangular field-gates on pair dot products. DeepFM adds a small
regularized MLP fed the exact same FM latent vectors, initialized to zero
initial output so the FM backbone is preserved.

All models use a robust Huber loss on log observed auction prices and report
original-dollar MAE after exponentiation. Training is CPU-only and
time-limited; the workflow has a 4-hour study deadline and a maximum
90-minute per-candidate GitHub job, with no hidden recurring retraining.
If any required candidate fails or misses the budget, the full report fails
closed. No production promotion happens automatically.

## Model selection and test isolation

- 26 independent models publish validation price predictions only.
- Check all 26 manifests, original auction row fingerprints, train sizes,
  identical validation observed prices, and unique row IDs.
- Original validation chooses rank, model family, and full/Lemon policy.
- Original test is scored for **one validation-selected checkpoint only**.
- A fixed XGBoost/Lemon blend coefficient is also chosen on validation
  for that same selected checkpoint, and reported on original test.
- Original full XGBoost benchmark is explicitly labelled aggregate/unpaired;
  original Lemon XGBoost is row-paired on exactly the same original auctions.
- XGBoost train-only residuals are not fabricated using in-sample errors.

The public report will compare validation MAE by architecture, latent rank,
and training cohort, emphasizing whether the extra full-data training samples
beat the 25%-Lemon cleaning policy. It will also report model complexity,
training time, independent test scores and bootstrap paired confidence
intervals.

Sources:
- scripts/bid_fm_scale_train.py
- scripts/bid_fm_scale_report.py
- .github/workflows/bid-fm-scale.yml

Outputs after success:
- reports/bid-fm-scale/comparison.json
- reports/bid-fm-scale/validation-leaderboard.csv
- reports/bid-fm-scale/full-vs-lemon-leaders.csv
- reports/bid-fm-scale/validation-xgboost-blends.csv
- reports/bid-fm-scale/winner-original-test.csv.gz
