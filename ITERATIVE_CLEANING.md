# Five-stage CatBoost release

The proposed new default cleaning recipe refits a **tabular CatBoost median-regression cleaner** five times. Each stage removes `ceil(0.05 * current_survivors)` separately within the original training, validation and test phases. This leaves approximately `0.95**5 = 77.38%`, rather than removing 25% overall. Integer rounding changes the exact count slightly. Removed rows never return; no old OutlierTree or quartile masks are reused.

## Fitting and assessment

1. Preserve the canonical full eligible Craigslist and wholesale-auction cohorts, chronological phases and VIN-overlap exclusions. Learn rare-model pooling from original training only. Make/model remain native categorical features.
2. At every stage, fit three CatBoost models on folds of surviving training rows; only out-of-fold predictions score training residuals. A fourth cleaner fits surviving training rows and predicts validation and test. Rank absolute residuals in **log reference-dollar price** independently within each phase. The cleaner excludes MiniLM's stacking feature.
3. On the identical retained cohort, separately assess **CatBoost normal SurvivalAft** and **CatBoost Quantile:alpha=0.5** for **both markets**. AFT receives exact positive dollar bounds `[price, price]` and performs its own log transformation. Quantile receives `log(price)`. Both serve `exp(raw_margin)`, a median-type estimate, without smearing to a mean. There are no verified censoring intervals in these data; no false Tobit/censoring claim is made.
4. Asking final models retain the existing fine-tuned MiniLM description-price scalar. Validate canonical SHA, row order, 3-fold assignments and provenance. Assessment uses train-only OOF text predictions plus the training-only text head for held-out rows. Deployment uses separately full-cohort OOF text predictions with the matching full-cohort serving head. Auction data contain no descriptions.
5. Each final architecture has an unfiltered-training baseline with the same parameters and stopping policy. Report USD MAE, RMSE, bias, log errors and Spearman on the **whole original test**, retained test and excluded test. Retained validation controls early stopping. Test scores do not choose the default: asking AFT and auction quantile are predeclared.
6. After assessment, separately refit each architecture on all retained phases using the selected tree count. Keep assessment weights and per-row original-test predictions. Publish ready manifests only after deployment fits succeed. Promote defaults only after all four artifacts and a completed summary are installed.

**Selection bias:** validation and test outcomes are used for the requested trimming. Their retained scores are optimistic, not unbiased generalization evidence. The original test population is never trimmed for the original-test report; however historical test results have informed development, so that report is also diagnostic. Sequential cleaner selection is adaptive: per-stage cross-fitting is not a fully nested cross-validation of the whole procedure. Large residuals can be real rare/expensive cars. A new, independent external cohort is needed before a general accuracy claim. Historical-price scores do not validate today's time-adjusted individual-car values.

## Parameters and Kaggle

Cleaning: 400 iterations, depth 6, median loss on log price; final: up to 1,500 iterations, depth 8, validation early stopping 100; learning rate .05, L2 10, Bayesian bootstrap, bagging temperature 1, seed 42. GPU uses supported parameters; CPU also uses `rsm=.8`. CatBoost GPU floating-point reductions are not bitwise deterministic.

Kaggle runs GPU cleaning and quantile; SurvivalAft uses CPU because [CatBoost does not support that objective on GPU](https://catboost.ai/docs/en/concepts/loss-functions-regression#used-for-optimization). `train_iterative_catboost_kaggle.ipynb` freezes source plus provenance, attaches the prior canonical dataset and verified MiniLM features, runs the full cohorts, saves every stage and assessment, and packages the four bundles. A saved batch run releases its GPU when execution ends. A failed batch is not a completed release. Check the saved output/log before installing.

CLI:

```bash
python -m scripts.train_iterative_catboost \
  --ask training-cache/full-cohort.jsonl --bid training-cache/bid-cohort.jsonl \
  --text-features training-cache/kaggle-features/minilm-features --device GPU
```

No transformer retraining is needed. The existing MiniLM text head stays paired with its verified feature exports. No Isolation Forest, ExtraTrees, fabricated unsold observations or IMR enters this recipe.

## BART report and interface

After an estimate, a full-width BART briefing combines the supplied car facts and seller description, both price targets, age/market decomposition, historical same-car scenarios at fixed entered mileage, native feature contributions, recorded model evaluations and time-component backtests. The source report explains the other studio panels without claiming that unrequested tools have been run. Historical scenarios are reconstructions, not observed car sales. No visitor analytics or unrelated session data enters the narrative.

The exact source story is downloadable. BART runs locally on CPU, maps/reduces all tokens of a readable qualitative briefing covering the report’s topics and disables numeric output tokens for this qualitative overview. That reduces invented numeric amounts but cannot guarantee factual correctness or exhaustive coverage. Prices still come from CatBoost and the time layer; BART does not determine prices.

The landing banner reads actual installation status. While training it explicitly says the current estimates use the installed models; after complete installation it shows the new formula and measured original-test comparison. It never promises increased accuracy merely because difficult test cars were removed.
