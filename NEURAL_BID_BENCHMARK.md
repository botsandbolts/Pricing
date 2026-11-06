# Neural auction / bid pricing: independent 18-arm historical wholesale benchmark

The goal is to reproduce the completed asking-price sweep, **not** to predict individual
live auction bids: the source provides VIN-deduplicated reported 2014–2015 wholesale
sale prices, adjusted to the same historical 2015 dollar basis as existing bid XGBoost.

## Completely preserved comparisons

Archived original data come from the same `matched-bayesian-redesign-data` evidence used
in prior bid modeling. The **full** eligible cohort contains 550,181 sales:
384,198 training, 82,380 original validation and 83,603 original test. The frozen Lemon
Squeezer cohort retains 288,148 training observations and **all** the original holdouts.
Exact `row_id`, `vehicle_group`, split and observed normalized price are audited.

We test: 3 networks (ResMLP, DeepCross, FT-Transformer) × 3 objectives (Huber,
median/pinball, exact log-normal AFT) × 2 *training-only* cohorts (full, Lemon).
Training-only final 10% provides early stopping; the original validation chooses the
best model; original test is not used for model selection. No hierarchy is involved.

## Sources and features

The archived full cohort and cleaned cohort are used directly; no raw Kaggle archive
is redownloaded. The source's pre-sale categorical make/model/body/state/color/transmission
and numeric model year, sale-date vehicle age, odometer and auction condition grade are
encoded **from training rows only**. The 2015 wholesale data have **no verified pre-sale
seller text**. A constant-zero one-dimensional model compatibility field carries no text
or price signal; the model's mathematical feature tokenizer receives only constants.

The auction price target is the **original full-bid XGBoost's normalized wholesale dollar
label**, not a new adjustment, a prospective quote, or a 2026 current value. Asking-price
AFT had exact asking prices; here AFT operates on **exact reported auction sales**, again
with **no invented censoring**. Models and pretrained category IDs remain experimental.

## Benchmarks and reproducibility

After all 18 jobs succeed, the report checks every untouched test/validation identity,
price and MAE against archived Lemon XGBoost quantile-0.5 predictions (paired), and
publishes published original full-data bid XGBoost as a descriptive aggregate baseline.
The report includes paired 300-resample bootstrap MAE differences; correlated vehicle
outcomes may make the listing-level uncertainty too narrow.

- Workflow: `.github/workflows/neural-bid-sweep.yml` (CPU GitHub Actions, 9 parallel jobs)
- Cohort preparation: `scripts/neural_bid_prepare.py`
- Training: `scripts/neural_bid_train.py`, reusing exactly the original shared architecture and regularization code
- Audited comparison: `scripts/neural_bid_report.py`
- Streamlit Model Lab: `pricing/neural_bid_ui.py`
- Published leaderboard: `reports/neural-bid-benchmark/comparison.json`
- Reusable Kaggle binary outputs: `canonical-bid.parquet` (ZSTD) and `features.npz`

The existing bid production model remains unchanged. No neural winner is installed as a
live bid estimator without separate deployment and external validation.