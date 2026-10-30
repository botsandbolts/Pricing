# Which words moved my estimate?

The main calculator now offers an optional phrase attribution panel immediately
after the price receipt and before the existing scientific explanation suite.
It runs only after the visitor requests it. The default asking model remains
MiniLM + CatBoost AFT; the auction model and all prices/training are unchanged.

## What is explained

The score function is the **complete deployed asking-price prediction in USD**:
description → existing MiniLM regressor → scalar `predicted_text_price` → selected
tabular pricing model → its normal served inverse transform and clipping.
Every coalition calls `pipeline.predict(..., explain=False)` with a copy of the
same submitted structured fields. CatBoost, original XGBoost AFT and MiniLM +
LightGBM use their existing prediction paths. No TF-IDF or surrogate is trained.
Tabular-only models show an explanation of why this panel is unavailable.
The wholesale auction dataset has no descriptions and is not explained here.

The baseline is **the same vehicle with an empty description**, including the
actual nonzero text-regression output for empty text. It is neither an average
car nor a zero text-feature ablation. Result labels distinguish the baseline,
submitted asking estimate, and their net description effect.

## Attribution contract

1. Apply `scripts.common.clean`, then retain the first 8,000 characters, matching
   `pricing.text.windows`. Highlight that cleaned text, not hidden raw input.
2. Prefer sentence/clause boundaries, merging shortest neighbouring clauses
   when over budget and splitting clauses longer than twelve whitespace tokens
   when budget allows. Unpunctuated text uses balanced word chunks. Preserve
   complete order and reconstruction. These are not learned semantic phrases.
   Small unpunctuated adverts can have one-word groups.
3. For a coalition, delete absent groups and separate retained chunks by spaces.
   Rerun the entire original pipeline. Cache identical coalition texts locally
   within this one analysis.
4. For ≤4 groups average every permutation (exact group Shapley values for this
   masking game). Otherwise average random permutations and their reverses with
   seed 42: Quick has ≤8 groups ×4 orders (≤30 unique evaluations); Closer has
   ≤12 groups ×8 orders (≤90). No extra SHAP dependency is needed to implement
   the standard permutation Shapley estimator.
5. Validate finite nonnegative outputs and dollar reconstruction. Independently
   compare the full-text prediction with the submitted receipt (one-cent
   absolute or 1e-7 relative tolerance). A mismatch discards the analysis.

Each path telescopes, so base + contributions equals the displayed prediction
up to numerical tolerance. This is an arithmetic guarantee, **not** a guarantee
of accurate individual allocations with a small permutation budget. The receipt
shows standard errors across antithetic-pair means for sampled paths, or zero
sampling error for exhaustive paths. These are not price confidence intervals.

## UX, performance and privacy

Highlights use signed-dollar tooltips and an accompanying sorted bar chart;
colour is never the only encoding. The chart shows the six largest effects and
combines the rest without losing their sum. Visitors can expand all phrases,
read the untruncated table, and inspect the method, runtime and diagnostics.
The interface stacks naturally on mobile through Streamlit's responsive columns.
Warm-up and inference latency depend on description length, model and CPU;
no unmeasured latency guarantee is claimed. A progress bar shows coalition work.

The explanation result lives only in the existing visitor's Streamlit session;
it is reused on reruns only when model instance/manifest, submitted car, cleaned
text, price and detail setting match. It is not placed in a global cache, logged
to analytics, sent to a summarization service, or written into a new database.
Only one result is retained per session. Normal price prediction and BART story
generation are not repeated by this panel unless its button is pressed.

## Interpretation limits

These are local, shared-credit reactions of a historical asking-price model.
Changing baseline, grouping or masking changes the explanation. Word deletion
can create unnatural descriptions, break negation, and alter tokenizer windows.
Adverts can repeat make/model/mileage, so effects are not condition-only premiums.
“Positive phrase effect” does not mean a causal resale gain, a verified repair,
a universal word-price rule, dealer profit, or a current completed-sale value.
Native tree SHAP graphs elsewhere continue to explain their own scalar/tabular
features and fitted output scales, not this separate text masking game.

Background: [SHAP text masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Text.html)
and [permutation explainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).
