# Car Price Studio — Europe alongside the United States

The app remains English-only. No IP- or browser-language redirection.
The US market retains its independent dollar Ask + historical Bid engines.

## Europe: verified rank-1 Gaussian AFT asking-price model

The rank-1 winner is a 600-round XGBoost survival:aft Gaussian ensemble from
https://github.com/botsandbolts/Pricing/actions/runs/42675550131.
The European source records advertised asking prices in euros, not completed
sales or wholesale bids. The untampered source contains 2,003,645 rows:
2,003,371 eligible prices, split 1,602,247 train, 200,039 validation and
201,085 sealed test. No deduplication/thinning was performed.

Winner (sealed test): MAE €3,156.17; RMSE €10,151.15; CVaR95 €22,684.94;
median APE 7.24%. Models were selected by validation CVaR95, not test.

## Release gating and browser QA

A trained model alone does NOT replicate inference. Every categorical mapping
must be reconstructed from the same training-only split. The packaging
workflow downloads the original Parquet data, checks the original SHA for
each split, reconstructs its exact train-only vocabulary, tests feature vectors
against the frozen training artifact and tests XGBoost inference parity.
Any failure blocks packaging.

The separate release workflow runs true Firefox, desktop and Android-width
viewports, on the real Streamlit app. It requires a working model-based
valuation, a functional market selector and absence of Streamlit exceptions.
Do not call a deployment complete until all release tests pass.

## Market-specific features

Europe has an independent asking-price form (km, registration year, EUR,
fuel, power, transmission, country), explanatory model contributions,
scientific details, scenario graphs, comparable aggregated source listings,
evaluation, model comparison and transparent anomaly reports. The limited
merchant and garage views do not invent unobserved dealer identities.

Price scenarios are NOT observed depreciation. Grouped listings are NOT
confirmed identical vehicles. A price discrepancy is NOT evidence of fraud.
No supported individual calibrated prediction interval is available.

## Data license and attribution

Source: KTHESAT/european-used-car-market-2m-listings, released as
CC BY-NC-SA 4.0. See
https://huggingface.co/datasets/KTHESAT/european-used-car-market-2m-listings.
The independent Car Price Studio software is MIT-licensed, but that license
does NOT supersede the data license. Only aggregate non-identifying
market summaries, not raw individual listings, are packaged here.

The interface is English as requested. Existing US pricing paths remain
unmodified. Publication to main is gated on clean packaging and Firefox tests.
