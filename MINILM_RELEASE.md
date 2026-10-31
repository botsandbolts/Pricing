# Mini Lemon Squeezer — MiniLM replacement

The migration workflow builds a new release; the app reads its feature contract from `models/xgboost-hierarchy/ask/features.json`. Until that verified bundle is installed, the existing release keeps its actual TF-IDF label. Documentation is not a substitute for trained weights.

## What changes

- Replace all current asking-model TF-IDF columns with 384 normalized MiniLM embedding dimensions. No UMAP or TF-IDF vocabulary enters the new cleaner, regular asking AFT model or matching Bayesian residual fit.
- Recompute five vehicle-group-held-out XGBoost residuals, remove the largest 25% of training absolute log errors, then refit and recalibrate. Validation and test are not selected by the trimming rule. Wholesale has no descriptions, so its median-quantile model remains structured-only and is retrained separately.
- The pinned frozen `all-MiniLM-L6-v2` encoder is int8 ONNX, CPU, 128 tokens, with consistent text cleanup. It does not learn any prices. XGBoost learns the mapping from embeddings and vehicle facts to asking prices. This is NOT a newly fine-tuned supervised MiniLM backbone.
- Current dollar SHAP, native log SHAP, sensitivity plots, value timeline, sharing and BART use the release-declared model. Phrase analysis reruns the complete MiniLM → XGBoost → optional Bayesian → date-adjustment function; its full-description endpoint must match the displayed asking price.
- BART receives the vehicle advert, unverified seller description, the valuations and their date, plus short method/evaluation context. Exact numerical results remain deterministic in the cards and source story; the neural summary cannot certify them.

## Why the supplied Kaggle prices are not simply reused as five-fold features

`botsandbolts/pricing` supplies the original 211,844-row cohort, three completed supervised OOF files and optimizer checkpoints. Its saved training status is pending. The cohort hash and original fold archive hashes are checked. Those three OOF assignments are not nested within a new five-fold XGBoost split: directly reusing them can pass held-fold labels indirectly through other training features. This release instead computes label-free embeddings of the original descriptions, then fits each XGBoost fold strictly within its training cars. The original supervised training evidence remains preserved.

## Evaluation and deployment

`MiniLM-only Lemon Squeezer and Bayesian release` builds original five-fold predictions and cleaned five-fold predictions, refits regular models, fits matching Gaussian hierarchical residual posteriors, measures original eligible validation/test MAE and coverage, and checks real serving inference under memory and package-size budgets. It publishes the bundle only after those checks pass. Scores shown in the GUI come from that bundle, not earlier TF-IDF scores. No improvement or current-market accuracy is assumed.
