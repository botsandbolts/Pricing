# CatBoost residuals and the hierarchical price laboratory

## Business English

Two markets, two questions: Craigslist asks describe what sellers advertise; auction results describe wholesale transactions. A rare car gets fewer direct comparisons. A hierarchical correction learns from its model, then its make, then the whole market, while giving sparse groups wider historical ranges. It does not turn an asking price into a completed sale, a future price guarantee, or a dealer profit estimate.

The five-fold CatBoost experiment fitted ask AFT on exact asking prices and bid median quantile on log auction prices. Craigslist descriptions were TF–IDF encoded and projected into eight UMAP coordinates within each ask training fold. Held-out cars never fitted that fold's text projection. The Bayesian layer uses **log(observed price / out-of-fold CatBoost price)**, so it corrects multiplicative pricing bias rather than dollar residuals. Validation calibrates a 90% historical outcome range; untouched test measures dollar MAE and empirical coverage. Low support means more borrowing and larger uncertainty.

## Technical English

For market `m`, row `i`, make `j`, model-within-make `k`, historical age `a` and cross-fitted base prediction `b`, let `r_i = log(y_i/b_i)`. The current joint Gaussian Gibbs implementation fits `r_i = μ_m + β_m(a_i-ā_m)/10 + u_mj + v_mjk + ε_i`, with Gaussian hierarchical effects and inverse-gamma variance priors. Ask and bid are **separate market posteriors**: a joint cross-market coupling has not been trained or validated. The base objectives differ and the advertised ask dataset supplies no observed censoring bounds; exact `[P,P]` AFT labels do not establish interval censoring. No likelihood comparison by LOO ELPD has been run. Those are research candidates, not results.

For each market, calibration rows require five genuine group-held-out predictions. The Bayesian posterior is fitted only there. The 90% scale is selected on chronological validation by a finite-sample conformal quantile of `|log(y/p)| / predictive_sd_log`; untouched chronological test reports coverage and USD MAE. Coverage for today's transported prices is **not** verified. The installed live hierarchical page currently uses an earlier XGBoost random forest and its own posterior. The newer CatBoost experiment is not a live estimator until its paired serving model, text projector, and posterior have been installed and tested together.

The ask Actions artifact is about 2.76 GB because UMAP retains the training reference set for `transform`; loading all five projectors in a Streamlit instance is unsuitable. Serve only the calibrated refit plus its matching transformer after a size and memory audit. Never use OOF fold weights for an arbitrary live car or replace the existing live model with residual predictions alone.
