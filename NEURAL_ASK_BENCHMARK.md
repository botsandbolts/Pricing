# Neural asking-price experiment

This is an experimental, ask-only research benchmark, not a production price
estimator. The current XGBoost and Bayesian/hierarchical layers remain intact.

## Research grid

- Three architectures: 3-block ResMLP, low-rank Deep & Cross, two-layer
  FT-Transformer with learnable categorical, numerical and text tokens.
- Three losses: Huber on log-price, quantile 0.5 (median) on log-price, and
  heteroscedastic log-normal AFT on exact observed positive asking prices.
- Two training cohorts: 147,777 original full training observations (including
  valid unusual listings) and the archived 110,832 retained Lemon Squeezer
  training observations. The removed 25% is a frozen TRAIN-only subset.
- The SAME 31,668 original chronological validation and 31,682 original test
  listings are evaluated by all 18 trained networks without any test filtering.
- Networks select their checkpoint using only the latest 10% of their respective
  training cohort. The cross-arm winner is chosen using original validation
  dollar MAE. Test outcomes are reported, never used for winner selection.

## Features and regularization

Data come from the verified original Craigslist Used Cars 2021 archive.
The original deduplication and price eligibility steps still apply; the full
cohort includes all canonical eligible training rows, not every invalid raw
advertisement. Numeric features include year, mileage, vehicle age and
description lengths. Train-only category vocabularies provide learnable
embeddings. TF-IDF with unigram/bigram 24-dimensional TruncatedSVD is fit on
a random sample of 30,000 training descriptions, without prices. No supervised
text price prediction, target encodings, XGBoost teacher labels, or hierarchy
enter the neural input.

Regularization includes AdamW weight decay 0.025, dropout, layer normalization,
gradient clipping at norm 1, cosine LR decay, limited jitter and training-only
early stopping.

AFT is a proper log-normal log-likelihood with ALL observations exact.
An asking amount is not evidence of interval or right censoring. No censored
bounds are fabricated.

## Interpretation of comparisons

The reporting code verifies that every neural prediction has the exact same
row ID, split and observed price as archived XGBoost Lemon AFT predictions,
and publishes paired validation/test dollar errors and paired bootstrap
confidence intervals for MAE differences. The historical full-data XGBoost
AFT report is included as an aggregate reference: its test/validation counts
match but a per-row audit is not available, so this comparison is not paired.

The XGBoost baselines use different text signals (MiniLM-price or TF-IDF/UMAP).
Any accuracy difference reflects the WHOLE model-and-feature pipeline,
not an isolated causal benefit of neural architecture. A future feature-matched
ablation would be needed to resolve that question.

## Reproducibility and publishing

- Preparation: scripts/neural_ask_prepare.py
- Training: scripts/neural_ask_train.py
- Audited reporting: scripts/neural_ask_report.py
- GitHub Actions: .github/workflows/neural-ask-sweep.yml
- Interactive read-only panel: Model lab > Neural pricing proving ground
- Published leaderboard after ALL 18 jobs pass: reports/neural-ask-benchmark/comparison.json

Models and per-row prediction artifacts are retained in GitHub Actions for
90 days. Only the small verified results are committed to the repository.
No neural model will be promoted to the live calculator automatically.
