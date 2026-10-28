# CatBoost normal AFT: deployed defaults

The requested models use the same fixed residual-trimmed membership as the preceding XGBoost experiment. Both complete native CatBoost bundles are required before the GUI prefers them.

| Model | Dataset | Retained candidates | Trees | Filtered test MAE | Original test MAE |
|---|---|---:|---:|---:|---:|
| ask | Craigslist Used Cars | 158,331 | 1,500 | $2,233.98 | $3,761.59 |
| bid | Used Car Auction Prices | 412,626 | 1,500 | $1,275.37 | $2,090.20 |

**Filtered validation/test errors are optimistic selected-cohort results.** Original-test diagnostics are also shown; this is not a new untouched external benchmark. These defaults were set by user request, not declared the best accuracy result. This refit is tabular-only; previous MiniLM options remain selectable.

Ask means advertised Craigslist listing prices, not completed sales. Bid uses Kaggle / tunguz Used Car Auction Prices sellingprice records and the existing wholesale age/CPI transport; it does not establish current retail-sale calibration or paired bid/ask labels. Dataset names and links appear prominently in the GUI.

CatBoost SurvivalAft uses normal log errors, exact positive dollar bounds and scale 1.0. Parameters allow 1,500 depth-8 trees, L2=10 and feature subsampling .8. Assessment fits retained training and selects stopping on retained validation. Deployment weights separately refit all retained phases. No GPU is required for serving.

The user-requested top 25% absolute log-residual removal was applied independently to the original train/validation/test phases, union OutlierTree flags. Original cohorts were preserved. Training ranking used OOF predictions. Native CatBoost SHAP reconstructs log price, and global importance is PredictionValuesChange. See [methodology](../../CATBOOST_DEPLOYMENT.md).

[Download the complete trained bundles and assessment artifacts](https://chatgpt.com/library?file_id=libfile_11cc7a9b08d48191a951d52b287f462d&account_id=personal). Exact original filter memberships remain in the linked source archives recorded in provenance.json.
