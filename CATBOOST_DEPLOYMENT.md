# CatBoost AFT defaults and dataset provenance

The user requested replacement of the newer XGBoost defaults with CatBoost using AFT loss. Both completed bundles under `models/catboost-aft` are required before the GUI activates them. Historical models remain available. The main calculator, auction companion and integrated two-market screen share the same fitted CatBoost bundles.

Ask: **Craigslist Used Cars**, 2021 advertised listing prices, archived as Tom's Used Cars Dataset at https://zenodo.org/records/10457828 (original Kaggle source https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data).

Bid: **Used Car Auction Prices**, Kaggle / tunguz, historical 2014–2015 reported auction selling prices, https://www.kaggle.com/datasets/tunguz/used-car-auction-prices. CarBids is this project's name, not proof these records came from the Cars & Bids platform. Wholesale sale prices and retail asking prices are different targets, and these datasets do not supply matched asking/final retail sale labels.

## Training and selection

CatBoost `SurvivalAft:dist=Normal;scale=1.0`, up to 1,500 trees, depth 8, learning rate .05, L2 leaf regularization 10, feature subsampling .8, Bayesian bootstrap, seed 42 and four CPU threads. Native categorical features use the persisted retained-training rare-trim mapping. Stronger regularization is a recipe choice, not a guarantee of better accuracy.

Exact positive dollar bounds `[price, price]` are supplied; CatBoost internally logs them. Raw predictions are log-price locations; exponentiation gives a conditional-median dollar estimate, not an arithmetic mean. No interval censoring is invented. Auctions retain the bundled CPI reference/transport logic.

The recovered fixed residual membership is reused exactly: remove ceil(25%) by absolute log residual in each original train/validation/test partition, union OutlierTree 0.099 flags. Training ranking used three-fold OOF CatBoost, while original validation/test ranking used train-fit assessment predictions. Retained counts: ask 158,331/211,127; bid 412,626/550,181. Original sources are not overwritten. Source SHA256 and all retained row identities are verified before fitting.

**Filtering validation/test using their labels makes filtered scores optimistic.** Original-test diagnostics are reported as well, but this is not a new untouched external benchmark. Assessment models fit retained training and use validation stopping; separate deployment models refit every retained phase at the selected tree count. The new defaults are user-requested, not automatically declared the best model. The refit is tabular-only; previous MiniLM options remain selectable.

## Explanations

Native CatBoost TreeSHAP reconstructs the fitted log-price margin. The GUI's dollar waterfall uses permutation Shapley estimates of the actual served price function; scenario plots recompute all derived features. Global CatBoost importance is PredictionValuesChange, explicitly distinguished from split gain and local SHAP. Categories are labels, not ordinal numbers. These plots explain model behaviour rather than causal effects or guaranteed transaction values.

## Reproduce

```bash
python -m scripts.train_trimmed_catboost_aft --side ask --ask training-cache/full-cohort.jsonl --bid training-cache/auction-cohort.jsonl
python -m scripts.train_trimmed_catboost_aft --side bid --ask training-cache/full-cohort.jsonl --bid training-cache/auction-cohort.jsonl
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Both memberships and underlying cohorts must match their provenance. Completed assessments resume before the final deployment refit; completed bundles are not silently retrained. `catboost==1.2.10` is part of the serving dependencies. CPU inference needs no Kaggle GPU.
