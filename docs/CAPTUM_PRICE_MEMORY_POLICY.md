# Captum on the public pricing page · low-RAM policy

Car Price Studio keeps the **verified CVaR95 Ask ResMLP #15** and **Bid tuned
DeepFM16 #18 + original Lemon XGBoost** models. Integrated Gradients
**does not train, convert, reload, pickle or replace any pricing checkpoint**.

## What we learned from Captum's own documentation

- [Integrated Gradients API](https://captum.ai/api/integrated_gradients.html):
  `internal_batch_size=1` processes one expanded interpolation sample per
  forward/backward step, instead of holding `n_steps × batch_size` autograd
  graphs at once. This significantly limits **peak attribution** memory, at
  the cost of additional serial execution.
- The default Captum `n_steps=50` is reduced to **8** on the public page.
  Users may request 16 or 24 steps, but no more. A higher value can improve
  integration accuracy, not model prediction accuracy.
- `method='gausslegendre'`: Captum's recommended high-quality quadrature
  scheme. It integrates along a *single numerical-feature reference path*.
- `multiply_by_inputs=True`: use conventional Integrated Gradients with
  input-versus-baseline differences, in the model's **USD** output space.
- `return_convergence_delta=True`: two additional small forward passes
  to audit approximation completeness. We explicitly warn when the error
  exceeds the budget; we do not pretend a poor approximation is exact.
- Do **not** use NoiseTunnel/SmoothGrad, multiple reference cars, LayerIG
  over BGE token layers, text-phrase perturbation batches, or large SHAP
  reference matrices. Each increases the amount of computation or active
  network memory without being necessary for this specific UI.
- Only one visitor's explanation is active at a time. The normal
  prediction path is never blocked by the attribution lock.

## Important scope distinctions

**Ask:** The original BGE is executed once during a user's ordinary
valuation and its final **384-dimensional frozen embedding** is retained
inside the visitor's session receipt. Captum attributes only the **seven
normalized numeric inputs** of the *actual deployed* ResMLP (or verified
DeepCross fallback), holding the categorical inputs and BGE embedding fixed.
The two text-statistic features are not word/token attributions.

**Bid:** Captum attributes only the five normalized numerical inputs of the
**actual weighted DeepFM branch**, holding the auction categories fixed.
The Lemon XGBoost tree branch is **not explained by these gradients** and
its weighted output is never mislabeled as DeepFM attribution.

The baseline is a vector of zeroes **after the training-set numerical
standardization** (approximately training medians), with original categorical
codes and BGE text embedding unchanged. It may be an economically impossible
vehicle. Attribution of correlated features (e.g. vehicle year and age,
mileage and miles per year) is sensitive to this baseline and cannot be
interpreted as causal resale-price changes.

The dollar contributions add up approximately to **baseline neural price
to original neural price**, not to an absolute decomposition of the car's
total price or a retailer's profit. Reject the result if the exact native
model's price fails to reproduce the saved price, or if gradients are
nonfinite. Always show the absolute convergence error and warn when its
approximation tolerance is exceeded.

## No automatic GPU, model swapping or new heavyweight packages

The only new runtime requirement is **`captum==0.9.0`**. It is lazily
imported inside the explicit user-triggered attribution function. The
BGE encoder, PyTorch pricing head and Lemon XGBoost continue through their
original globally cached verified loaders. No process spawning, loading and
unloading model weights or quantization is used.

The public first-appraisal path does not call Captum. Historical valuation
charts and multi-mileage predictions were made opt-in, so the previous
Streamlit expander behavior cannot accidentally run those computations
during ordinary page rerenders.

## QA and limitations

- `tests/test_captum_price_lite.py` runs **real Captum** against small,
  deterministic PyTorch heads and checks numerical price parity, completeness,
  batch limit, frozen BGE reuse, true Bid neural-only scope, erroneous model
  receipts and concurrency limits.
- `.github/workflows/captum-price-qa.yml` runs those tests plus homepage
  launch contracts and five desktop/mobile browser smoke cases; screenshots
  and JUnit results are retained.
- Synthetic attribution tests are not evidence of historical or modern
  car-pricing accuracy. The research-model release workflow independently
  audited historical test predictions before promotion.
- Streamlit Community Cloud memory limits, browser/Android behavior,
  newly selected dates and end-user concurrency must still be observed
  in live deployment; CI cannot prove the hosted app is OOM-free.

Source: [Captum Integrated Gradients API](https://captum.ai/api/integrated_gradients.html),
[Captum OOM FAQ](https://github.com/meta-pytorch/captum/blob/master/docs/faq.md),
[Captum 0.9.0 release notes](https://github.com/meta-pytorch/captum/releases/tag/v0.9.0).
