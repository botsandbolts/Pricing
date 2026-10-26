# Visual explanation studio

After estimating a car in **Value a car**, choose **Build all explanation graphs**.
The studio works with the trained unfiltered ensemble, the filtered ensemble and
the tabular baseline. Calculation is on demand; changing display controls reuses
the analysis. Changing the car, model, background, member or sampling settings
requires a new analysis, so an old explanation cannot silently describe a new car.

## The six graphs

| Graph | Function explained | Units |
|---|---|---|
| Waterfall | Complete served model, including USD averaging across ensemble members | Dollars |
| Beeswarm | One selected LightGBM member across reference cars | That member's unstandardized target scale |
| Dependence scatter | The same member's native feature contributions | That member's target scale |
| Decision plot | Cumulative native contributions for the submitted car and up to four nearby reference cars | That member's target scale |
| Mileage curve | Complete model evaluated at changed mileage | Predicted dollars |
| Year × mileage heatmap | Complete model evaluated on a two-feature scenario grid | Predicted dollars |

The last two are model sensitivity plots. The heatmap is **not** a SHAP interaction
matrix. Neither curve nor heatmap establishes causal depreciation. Derived age,
mileage, composite-category and CarBids features are recomputed from each raw
scenario. The asking reference date stays May 1, 2021; model generations and future
prices are not identified by changing model year in the form.

## Dollar attribution for the actual ensemble

Each ensemble member predicts on its own standardized training target, applies
its inverse transformation and clipping, then contributes to a weighted USD
average. Exponentiating native log-SHAP values or averaging them does not explain
that complete dollar function.

The waterfall instead uses grouped interventional permutation Shapley estimates
on the **actual served USD function**. Eight raw-input groups partition the 13
structured fields. Make/model/body are masked together to avoid mixing a make
with another reference car's model; engine/drivetrain fields are grouped too.
Other correlated groups can still create unrealistic hybrids. These are model
credit allocations relative to a chosen background, not causal price components.

Eight fixed reference cars are selected evenly from the current gallery (or all
cars if fewer than eight). Quick/Balanced/Thorough use 4/8/16 independent random
permutation pairs, each paired with its reversed order: 8/16/32 feature orders.
Seed 42 fixes the draws. All coalition rows are predicted in a batch. No price
labels enter these calculations.

Each path telescopes from its reference prediction to the submitted prediction,
so the contributions reconstruct the price even though individual allocations
remain Monte Carlo estimates. Reconstruction is checked numerically. The receipt
reports standard errors across independent antithetic-pair means, conditional on
the fixed background. Those are **not** price prediction intervals or uncertainty
about the representativeness of the background. Reference choice changes the
explanation question and may materially change allocations.

## Native member explanations

LightGBM `pred_contrib=True` provides tree-path-dependent TreeSHAP values.
Contributions are multiplied by the member's target standard deviation; its mean
is added to the base. This reconstructs the member's unstandardized pre-clipping
score. For log-target members, axes explicitly say `log(1 + price)`. They never
pretend to be dollar discounts or the complete ensemble.

Beeswarm features are ordered by mean absolute SHAP within the gallery. Numeric
colours use per-feature 5th/95th-percentile scaling; categories are grey because
their integer encodings are not magnitudes. Dots sharing nearby contribution bins
are stacked symmetrically. The decision plot preserves all contributions by
combining undisplayed columns into Other features. Its feature order is a display
order, not an execution trace through trees. Nearby reference cars use year and
log-mileage distances, not target labels or verified comparable sales.

For a future completed MiniLM bundle, charts hold the submitted scalar
`predicted_text_price` fixed across structured-reference and sensitivity cars.
The dollar graph is then conditional on that description. Native charts can
allocate credit to the scalar, but none of these graphs claims word-level
attribution or a complete explanation of the transformer.

## Reference data and privacy

`models/explanations/reference-cars.json` decodes the existing 30 parity fixtures
in `dist/baseline-fixtures.json` using the exact shipped baseline schema. Re-encoding
reconstructs the original fixture matrix. These are evenly spaced historical
eligible-cohort rows, **not a random population survey or a new evaluation set**.
Only structured predictors are included; there are no labels, descriptions, VINs
or contact fields. The GUI states the small-gallery limitation.

An optional CSV can supply up to 1,000 reference rows / 2 MB. A strict allowlist
keeps vehicle predictors and ignores price labels and extra fields. At most 128
evenly spaced uploaded rows are used for native charts. The USD waterfall uses up
to eight of those rows. This is user-selected reference data, not automatically
representative data. Uploaded rows are kept only in the app's current session;
they are not written to repository files.

The explanation receipt exports the USD group allocations, Monte Carlo errors,
method settings, reconstruction error, selected member and reference provenance.
It is not a valuation certificate. Humorous lay explanations sit alongside the
technical contracts throughout the GUI.

## Validation

The new tests check nonlinear dollar reconstruction, exact additive-model
allocations, coupled make/model masking, reference fixture round trips, the real
nine-member prediction function, native reconstruction, recalculated derived
features, graph scales and decision-path endpoints. Streamlit AppTest exercises
all six plots for ensemble and baseline, display-control reruns and stale-chart
protection. Existing structural and ensemble checks remain in CI.
