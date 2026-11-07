# Car Price Studio · Captum without the memory monster

**Public route:** `🚗 Price my car` → Ask/Bid receipt → **🔬 Why THAT price?**

**Production implementation:** [`pricing/captum_price_lite.py`](../pricing/captum_price_lite.py).

This is an on-demand explanation of the **existing parity-verified models**, not a new estimator, model conversion, retraining job, standalone explanation service or price recommendation from an LLM.

## Why Integrated Gradients?

Captum's original [Integrated Gradients API](https://captum.ai/api/integrated_gradients.html) supports exact settings for controlling the number of forward/backward samples in memory. Its [OOM FAQ](https://github.com/meta-pytorch/captum/blob/master/docs/faq.md) specifically recommends a smaller `n_steps` or `internal_batch_size`. Smaller steps can increase numerical integration error, which is why the UI also shows a convergence residual.

| Parameter | Chosen behavior | Reason |
|---|---|---|
| `internal_batch_size` | **1, fixed** | Evaluate each interpolation point separately, rather than constructing a large `n_steps * batch_size` activation graph |
| `n_steps` | **8 / 16 / 24** | A bounded quick-to-detailed choice instead of Captum's 50-step default |
| `method` | `gausslegendre` | Captum's quadrature approximation; efficient at low step budgets |
| `return_convergence_delta` | `True` | Explicit completeness check; exposes numerical integration residual |
| `multiply_by_inputs` | `True` | Return integrated changes relative to a baseline rather than raw gradient sensitivity |
| Forward output | **Historical dollars** | User-friendly amounts; no artificial dollar conversion from log-space attributions |
| Inputs | **7 Ask numeric / 5 Bid numeric** | Avoid gradient paths through a 384D transformer and many categorical embeddings |
| Baseline | **0 on standardized numerics** | Per-feature median of the train-only normalizer, with current category and BGE vector fixed |
| Execution | **Only on button click** | No extra gradient work on page load or first price request |
| Concurrency | **One active request** | Protect other Streamlit sessions from overlapping attribution memory spikes |
| Backend | **Existing CPU PyTorch checkpoint** | No quantization, surrogate, ONNX transformation, subprocess or model reloading |

## What is explained, precisely?

### Seller Ask — native BGE + neural price head

The full original BGE-small encoder has already generated its 384-dimensional text embedding during the user's estimate. The small embedding is saved in Streamlit session state alongside the existing model receipt. Captum explains only the **seven original normalized numerical inputs** to the complete live ResMLP/DeepCross pricing head. **The original BGE embedding and all 11 categorical IDs remain fixed.**

An attribution bar is therefore **not** a price contribution of a particular advert word, trim, make, title or brand. The synthetic numeric-median reference is not a real transaction or necessarily a physically plausible car.

### Wholesale Bid — DeepFM part of the verified blend

The deployed 2026-11-07 CVaR95 winner uses a validation-selected **90% DeepFM / 10% Lemon XGBoost** blend. Captum explains **the weighted DeepFM branch only**, over its five numerical auction features. The original XGBoost tree, categorical embedding IDs and auction reference date remain fixed. The full Bid can differ because the XGBoost 10% contribution is not part of these gradients.

### Validation safeguards

- Every explanation must reproduce the original *saved* Ask or weighted neural Bid amount with the unchanged serving network.
- Captum returns one approximation residual from the **completeness property**. If it is too large, the UI warns and does not display speculative individual feature contributions.
- Model output and gradients must be finite, and input dimensions must match the training cohort.
- A feature's local model attribution is **not a causal resale effect**, profit margin, repair cost, reliability estimate or individual prediction interval.
- The historical Ask/Bid labels and selected-date market-index adjustments are different populations and operations. Their scientific limitations remain explicit.

## Packages and cold-start footprint

The only new dependency is **`captum==0.9.0`**, a pure-Python wheel in the existing PyTorch environment. It does not install Captum Insights, extra transformer weights, or another BGE instance. Captum 0.9 adds NumPy 2 support; [see official release](https://github.com/meta-pytorch/captum/releases/tag/v0.9.0).

The first public page view imports only lightweight helper functions. `from captum.attr import IntegratedGradients` happens **after a person presses Explain**, and the original pricing head is retrieved from the same `st.cache_resource` release loader that produced the price.

## Quality checks

The GitHub workflow [`captum-price-qa.yml`](../.github/workflows/captum-price-qa.yml) runs:

1. Actual Captum 8- and 16-step gradients on representative PyTorch pricing-head architectures, with exact dollar reconciliation.
2. Release-gate tests for Ask and Bid, original model class and immutable checkpoint contracts.
3. Assertions that the form reuses the saved BGE representation and exact historical Bid input record, and that the large history/mileage charts remain click-triggered.
4. A Playwright browser smoke test for desktop, tablet, Android-sized, WebKit iPhone-sized and Firefox screens.

Tests of small synthetic networks check Captum's *implementation and arithmetic*, **not real-car prediction accuracy**. Historical 31,682-row Ask and 83,603-row Bid score audits are recorded separately in the released model manifests. A browser smoke test is likewise not a measurement of real production memory exhaustion.
