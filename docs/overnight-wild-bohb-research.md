# Overnight WILD BOHB — Ask and Bid CVaR95 risk research

**UTC hard cutoff:** 2026-11-08 05:00:00Z = **Sunday 08 Nov 2026 at 06:00 Europe/Berlin (CET)**.

**Status:** automated research. Experimental model outputs are never silently deployed.

## Objectives and validation

The only final model-selection objective is **CVaR95 of absolute USD valuation errors**:
the mean absolute price prediction error among the worst 5% of examples.
MAE and RMSE are secondary. Each completed checkpoint must include its complete
original-validation predictions, and an independent audit recomputes all three
metrics from the frozen, cohort-fingerprinted data. TEST is not read during
optimization. Historical source-year USD scoring is *not* proof of November
2026 adjusted-date appraisal accuracy.

The incumbents are the validated Ask ResMLP ($18,069.28 CVaR95, $2,836.44 MAE,
$5,718.30 RMSE) and Bid DeepFM + Lemon ($8,720.66 CVaR95, $1,684.12 MAE,
$2,975.84 RMSE), **on the original validation cohorts**. The 216-epoch-budget
candidates alone enter the direct full-fidelity challenger leaderboard; lower
fidelities are retained as research evidence.

## Search breadth

- Ask: **32** search dimensions including frozen BGE projection width and
  layers, projection dropout, DCNv2 cross layers and scale, low-rank size,
  mixture-of-expert count, nonlinear residual block depth/width, activation,
  output/embedding dropout, feature noise, batch, gradient clipping and
  patience.
- Bid: **26** dimensions including DeepFM latent rank, tower widths/depth,
  dropout, activation, pairwise gates and fixed **pre-registered** neural /
  original Lemon XGBoost blending alpha.
- Both: log learning rate/weight decay, Adam/AdamW/RAdam/NAdam/AMSGrad,
  beta1/beta2, constant/cosine/warm-restart/step learning-rate schedules,
  per-configuration risk-loss family, tail strength, tail quantile, log-Huber
  threshold, asymmetry, dollar-error scale, softened CVaR temperature,
  focal exponent and AFT sigma floor.
- Full training budget schedule: **6 → 18 → 54 → 162 → 216 epochs**.
  Native training retains its independent TRAIN-only early stopping: an assigned
  216-epoch budget is the *maximum*, not a claim that 216 epochs were executed.
  Promotions are deterministic *restarts from the same seed*, **not** resumed
  optimizer checkpoints. Warm-restart/cosine schedules use fixed periods across
  budgets to maintain comparability of early learning.

## Experimental training losses

These are **surrogates**. Every trial is still evaluated with exact historical
holdout CVaR95, MAE and RMSE in USD. All loss computations use TRAIN-only data;
all dollar errors are normalized for gradient stability.

1. Exact-observation log-normal AFT NLL (Ask); Bid retains robust log-Huber proxy
2. Log-price Huber / Smooth L1
3. Log-price pseudo-Huber / Charbonnier
4. Log-cosh
5. Log-price MAE
6. Log-price MSE
7. Asymmetric expectile in log space
8. Dollar Huber
9. AFT plus dollar-Huber hybrid
10. Dollar top-k CVaR training penalty
11. Softplus-smoothed dollar CVaR tail penalty
12. Dollar focal penalty
13. Asymmetric dollar-error penalty
14. Log-dollar hybrid

Tail-relevant formulations may reduce catastrophic misses but can also become
noisy, unstable or overfit the small extreme-error sample. The independent
validation prevents claiming an improvement based on TRAIN-only loss alone.

## Scheduler / robustness controls

The GitHub Actions study allows **up to 30 eight-worker batches**, four Ask and
four Bid concurrently. It checks time before every batch and stops dispatching
near the 06:00 CET deadline, with remaining-time bounds on each training worker.
Unstable isolated hyperparameters are registered as failed attempts, not fake
zero-error winners. Missing/unmatched data, original-cohort drift, or audit
parity failures remain fatal. Matrix failures must never trigger production
promotion, and all candidates are explicitly labeled research-only.

Outputs: `overnight-history-<stage>` artifacts and final
`overnight-BOHB-final-comparison` audit artifact. The last completed stage
remains available even when the deadline skips later configured batches.

## Literature / implementation references

- Falkner, Klein & Hutter, **BOHB: Robust and Efficient Hyperparameter Optimization
  at Scale**, ICML 2018:
  https://proceedings.mlr.press/v80/falkner18a.html
- HpBandSter BOHB docs (kernel density modelling, training budgets):
  https://automl.github.io/HpBandSter/build/html/optimizers/bohb.html
- PyTorch built-in optimizers and LR schedulers:
  https://docs.pytorch.org/docs/stable/optim.html
- PyTorch SmoothL1 robust regression objective:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
- Foret et al., **Sharpness-Aware Minimization**: an inspiration for future
  gradient-geometry experiments, *not implemented* in this study:
  https://research.google/pubs/sharpness-aware-minimization-for-efficiently-improving-generalization/

**Not implemented intentionally:** no synthetic censored labels, no leakage
from original TEST, no pricing-date extrapolation changes, no automatic
Streamlit deployment, and no claim that the largest epoch budget was fully
consumed when early stopping occurs.
