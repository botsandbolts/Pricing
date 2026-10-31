# Serving contract

The active feature contract is `ask/features.json`; the release identity is `release.json`. Regular asking and auction weights remain separate from Bayesian posteriors. Read each market's `bayesian/bayesian-evaluation.json` for the actually installed posterior, its measured dollar MAE, historical coverage and selected likelihood. Training completion alone is not deployment.

Base asking model: XGBoost normal AFT with exact positive observed asking-price bounds. It receives vehicle facts and one description feature: the arithmetic mean of two group-held-out MiniLM dollar-price predictions. The frozen int8 encoder and Ridge price heads compute that scalar internally; raw embeddings, TF-IDF and UMAP are not tree inputs. Auction: structured-only XGBoost log-dollar median quantile regression. Both have 300 rounds and depth five.

The original five-fold OOF log-residual ranking removes 25% of eligible training rows once. Validation/test cars are not trimmed. Subsequent OOF diagnostics are selection-conditional, not nested evaluation of trimming; valid unusual cars may have been removed.

The [Bayesian redesign](../../ROBUST_BAYESIAN.md) changes only the residual layer: nonlinear fixed effects, cross-classified pooled vehicle details, make-specific age slopes and heteroscedastic noise. Gaussian, Student-t and contamination candidates compete on separate validation cars; other validation cars calibrate risk-stratified historical outcome intervals. Promotion requires paired test improvements, monitored convergence, acceptable historical coverage, matching base identities, real inference and serving-budget checks. A failing market retains its preceding posterior.

All accuracy evidence is historical. The original test cohort was also reported in previous releases, not collected as a new external benchmark. Today’s transport, forecast months, generation analogues and current-date interval coverage remain unvalidated. Overall coverage is not conditional coverage for every car type. Asking and auction describe different populations, without verified censoring bounds or an identified dealer-profit relationship.
