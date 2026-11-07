# Why this price? · Native neural explainability

Car Price Studio uses **two economic targets**, not two labels for one target:

- **Seller Ask**: a historical advertised asking price (not a completed retail sale). The source encoder is the pinned BGE-small 256-token, 384-dimensional sentence embedder followed by the complete native PyTorch asking network (DeepCross/AFT or the parity-verified ResMLP challenger).
- **Wholesale Bid**: a completed historical 2014–2015 wholesale auction-sale target in 2015-index-normalized USD, **not a dealer's live bid or an offer**. The source engine is DeepFM16 + the original Lemon XGBoost on VIN-disjoint auction cohorts. The neural/tree blend weight comes from the released model's validation selection.

The original neural checkpoints, preprocessing, frozen BGE encoder and Lemon XGBoost are loaded through their verified serving APIs; XAI neither retrains nor serializes alternative model weights.

## Main-page UX

After a user requests an appraisal, **🔍 Why this price?** offers:

1. **Simple**: positive and negative dollar drivers compared against a disclosed *hypothetical reference configuration*, with honest noncausal language.
2. **Dollar drivers**: an accessible chart and screen-reader-friendly table; exactly six predeclared feature groups per market.
3. **Price clock**: historical model anchor + dollar depreciation effect + dollar market-index effect = the *identical* selected-date estimate already displayed.
4. **What if?**: mileage, title status, condition and Ask advert text. Reruns the original model, not an approximation or word-level causality assertion.
5. **Advanced AI**: Captum Integrated Gradients, native XGBoost TreeSHAP, BGE phrase-removal sensitivity, each separately on demand.
6. **Research receipt**: redacted JSON download, no listing text, contact details, VINs or vehicle facts.

The expensive parts are **never run on initial page load** or merely when a price is requested. Explain is one separate click; Captum/TreeSHAP/phrase-removal each need another explicit click. We use a single BGE encoder instance shared with Ask pricing. The group calculation reuses two prepared text embeddings across all 64 coalitions, rather than making 64 transformer calls. Stored results have a signature of the submitted price, input and model identity; changed data invalidate the previous explanation.

## Additive dollar explanation

The six groups for Ask are model+year, mileage, body+drivetrain, condition+title, state+paint, and advert description. Bid's sixth group is pre-sale auction grade instead of text.

Let \\(x\\) be the submitted vehicle, \\(x_0\\) the chosen reference configuration, and \\(v(S)\\) be the *actual saved native model* prediction after replacing only the groups in \\(S\\). Exact grouped interventional Shapley for group \\(i\\):

\\[
\\phi_i = \\sum_{S\\subseteq N\\setminus\\{i\\}} \\frac{v(S\\cup\\{i\\})-v(S)}{n\\binom{n-1}{|S|}}
\\]

For \\(n=6\\) all \\(2^6=64\\) coalitions are evaluated. The application **refuses the explanation** unless:

- The full-coalition native price equals the saved prediction, within documented cents-level/numerical tolerance.
- The reference price plus all six contribution dollars equals the full-coalition prediction.
- Every coalition price is finite and positive.

The reference is **not a dataset average, a conditional feature distribution, the model's global expectation, or a causally legitimate automotive counterfactual**. The user can compare either with the same make/model/year/body with standardized details, or a fixed 2010 Toyota Corolla. Arbitrary hybrid vehicles may not exist. Signed contributions explain the difference *from the chosen reference*; they are not the price or cost of replacing a part, an expected discount or a dealer margin.

## Captum Integrated Gradients

Captum is imported only when the user selects Advanced AI and clicks Calculate. It interpolates **normalized numerical inputs** (and frozen BGE embeddings for Ask) along a baseline-to-vehicle path, computes the true PyTorch network's output in **historical dollar space**, and measures approximation convergence/completeness. Categorical embedding codes are **held fixed**, so IG does not purport to explain model-year/make/trim categorical differences. For Bid, IG explains only the neural branch; the combined ensemble requires a separate tree analysis.

Refs: <https://captum.ai/api/integrated_gradients.html>

## Native XGBoost TreeSHAP

We invoke the **installed XGBoost Booster** with `pred_contribs=True` on the precise original feature matrix used to predict the Lemon branch. Its contributions are *additive in log dollars*. The UI displays those log contributions, plus an explicitly labeled exponential-path rescaling that reconciles **exactly** to the XGBoost **dollar** price:

\\[
d_i = \\phi_i \\frac{\\exp(z)-\\exp(b)}{z-b}, \\quad z=b+\\sum_i\\phi_i
\\]

This is **not dollar-space TreeSHAP**, and does not explain DeepFM. The total tree price must equal the actual Lemon component in the active model receipt, otherwise refuse the chart.

## BGE advert phrase removal

Split the user's text into at most 12 bounded sentence/word groups, remove each one in turn, and **re-embed the modified entire advert with the original pinned BGE** before rerunning the full neural price model. These are **one-at-a-time model sensitivities** (not additive SHAP and not causal evidence of negotiating improvements). The user's original advert remains local to the pricing request and never enters the download receipt.

## The selected-date effect

The app's historical model output (Ask May 2021 or Bid 2015) is transported using a *separately learned vehicle-age depreciation curve* and an aggregate used-car retail market index. We show an exact arithmetical reconciliation:

Historical price + age change + market change = displayed date-adjusted scenario.

This does **not** establish modern individual-car accuracy or wholesale market-index coverage. Recent generations, auction grades, title issues, textual ambiguity, dataset bias, and out-of-distribution listings may produce poor predictions.

## Quality gates

- `python -m pytest -q tests/test_neural_price_xai.py` tests exact Shapley including feature interactions, actual-vs-saved price parity, double-counted groups, source-cohort reference assumptions, BGE preparation count, Bid auction inputs, date bridge and real phrase-reembedding.
- Captum tests use tiny synthetic PyTorch networks to audit dollar completeness; they do not fake source-model accuracy.
- `python -m pytest -q tests/test_main_pricing_campaign.py tests/test_experimental_two_models.py tests/test_cvar95_experimental_ui.py` protects homepage routing, date controls, retired BART/scan flows and verified serving.
- A separate Playwright smoke test probes Chromium desktop/tablet/Android, Firefox desktop and WebKit iPhone, with screenshots and a JSON report.
- Actual original-trained checkpoints have their **own** no-fabrication native parity and release workflows. Passing synthetic attribution unit tests does not prove historical price quality or deployed checkpoint parity.

This capability is **open source**, deliberately CPU bounded, consent-aware and privacy-safe by design. No external explanation API or new data broker is involved.
