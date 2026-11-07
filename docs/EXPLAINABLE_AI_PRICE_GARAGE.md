# Why this price? · Lightweight model-faithful explanations

**Live public implementation:** [`pricing/price_explain_lite.py`](../pricing/price_explain_lite.py) attached to the main seller Ask + wholesale Bid calculator.

## Stable release decision

On 2026-11-07 the pricing UI was restored from GitHub revision
[`490a6906`](https://github.com/botsandbolts/Pricing/commit/490a6906e45f301a245aaaa951f480fef20b06a9),
the version active shortly before the LinkedIn announcement. The native pricing
models, release gates, in-process loaders and caching remain unchanged:

- **Ask**: pinned BGE encoder + complete original DeepCross/AFT PyTorch head
  (or validation-selected ResMLP only if its release manifests and parity receipts
  are installed and verified).
- **Bid**: original DeepFM16 + Lemon XGBoost (or the verified challenger only
  when its release-gate artifacts exist).
- Models are obtained from the original `st.cache_resource` pathways,
  shared across sessions and reruns. **No subprocesses, per-request model
  loading/unloading, encoder copies, hidden fallback models, or quantized
  conversion files**.

In the original 2026-11-07 public state, the new CVaR95 challenger checkpoint
bundles were still unavailable. The UI displays their presence correctly and
falls back to the existing holdout-verified native models. It does not fabricate
CVaR95 test results or claim the unavailable winners were deployed.

## What the panel calculates

- **Quick business-language explanation**, without inference: Ask is an
  advertised listing price, Bid a historical wholesale sale target. They
  concern different markets; their difference is not a dealer's profit.
- **The real Bid ensemble**: show DeepFM's and Lemon XGBoost's prices and
  checked blend weights. Refuse the breakdown unless weighted constituent
  prices reconcile to the saved native model output.
- **Exact price clock**: `historical model price + vehicle ageing + aggregate
  market-index effect = selected-date estimate`. Verify the original anchor,
  decomposition and final price; no hidden surrogate or stochastic XAI.
- **One on-demand what-if**: mileage, title status or Bid auction grade. Reuse
  the same globally cached model, and the **already-computed BGE embedding**
  retained in session state for the Ask inference. Exactly one additional
  native price evaluation per click. Explicitly label this *model sensitivity*,
  not causality or a vehicle repair valuation.
- **Optional one-row native XGBoost TreeSHAP**: directly call the installed
  `Booster.predict(..., pred_contribs=True)` once on the same auction feature
  row as the saved original tree component. Verify that the log contributions
  reconstruct the tree margin and that the tree's dollar output matches the
  saved Lemon price. Show the top contributors in **log-price**, not fake
  additive dollar contributions. This explains only the tree's share of Bid,
  not DeepFM or the entire ensemble.

Explanations are **on demand**. The first appraisal adds no explanation
inference; a closed tab does not compute XAI. The previous automatically
executed five-point mileage sweeps and long diagnostic charts are not run on
the public main page.

## What this deliberately does NOT include

No Captum or gradients, no additional SHAP package, no 64- or 128-coalition
model sweeps, no second BGE instance, no background model jobs, no heavyweight
Plotly gallery, no loading/unloading processes, and no additional tracking.
The earlier `neural_price_xai.py` research experiments remain source
material but are **not imported by the public page**.

### Important scientific caveats

- Native XGBoost feature contributions are additive in the **raw untransformed
  margin (log-price)**, not automatically additive in USD.
- A single changed feature can interact with other attributes; local
  sensitivity is not a causal estimate or expected transaction gain.
- Historical asking listings and 2014–2015 wholesale auctions are distinct
  label populations; date transport to 2026 is a scenario, not a current
  transaction measurement.
- Signed explained factors never establish the exact economic value of
  changing a title, odometer reading, repair, condition or listing wording.
- CVaR95 means average *absolute* USD error among the worst 5% of the
  evaluated historical predictions, not a guaranteed individual error bound.

## Technical sources

1. [Streamlit: model caching](https://docs.streamlit.io/develop/concepts/architecture/caching):
   `st.cache_resource` shares a resource instance without copying it;
   cached objects should be used safely across threads.
2. [XGBoost Python API](https://xgboost.readthedocs.io/en/latest/python/python_api.html):
   `pred_contribs=True` returns one row of native TreeSHAP feature
   contributions, which add to the model's untransformed margin.
3. [CatBoost's native ShapValues](https://catboost.ai/docs/en/features/feature-importances-calculation):
   another dependency-free attribution route if a native CatBoost model is
   chosen as the deployed estimator later.

## QA

`python -m pytest -q tests/test_price_explain_lite.py tests/test_main_pricing_campaign.py tests/test_experimental_two_models.py tests/test_cvar95_experimental_ui.py`

The `price-explainability-qa.yml` workflow also checks the actual Streamlit
home page at desktop/tablet/Android/iPhone dimensions and preserves screenshots.
A passing offline QA receipt proves only this tested scope; it is **not**
a direct measurement of live cloud RAM or modern vehicle-market accuracy.
