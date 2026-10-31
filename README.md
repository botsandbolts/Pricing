# Car Price Studio

A free, open-source U.S. used-car pricing laboratory. Enter vehicle facts and an advert, compare advertised asking and wholesale auction estimates, inspect the model, and explore date-adjusted scenarios.

[Open the app](https://pricing-5a4k2zenhw2nte9xpshxs3.streamlit.app/)

## Which models are actually serving?

The installed contract is [models/xgboost-hierarchy/release.json](models/xgboost-hierarchy/release.json). Asking description features are declared in `ask/features.json` when present; older bundles without that file use TF-IDF. The GUI reads this contract rather than treating a training job as a finished deployment.

The replacement release uses **one MiniLM description-price feature**, the arithmetic mean of two dollar predictions. XGBoost does not receive raw embedding dimensions or TF-IDF columns in that release. Its workflow is **MiniLM two-price Lemon Squeezer and Bayesian release**. Only a successfully trained, evaluated and smoke-tested bundle is automatically installed. Until then, the existing model remains available with its real feature label.

See [the MiniLM release method](MINILM_RELEASE.md), the GUI’s **The science** panel, and the installed evaluation JSON files. Earlier architecture descriptions and scores are archived in [HISTORICAL_README.md](HISTORICAL_README.md); they are not the current-default specification.

## Pricing method

- Asking: XGBoost normal AFT, using positive dollar bounds. The dataset supplies exact asking prices, so training uses exact bounds; a real interval-censoring process has not been established.
- Wholesale auction: XGBoost median quantile regression on log dollars. The original auction source has no seller descriptions.
- MiniLM price feature: pinned frozen all-MiniLM-L6-v2, compact int8 ONNX inference, a consistent cleanup and 128-token window, and two Ridge log-price heads. Embeddings stay internal to the text-price computation.
- Training text features: two repetitions of group-held-out two-fold fitting give every training car two price predictions from models that excluded its vehicle group. Their prices are averaged in dollars. This fitting is nested within each outer XGBoost training fold. Two complementary training-only heads supply new/validation/test features.
- Lemon Squeezer: five outer vehicle-group-held-out models produce genuine log residuals. Remove the largest quarter of eligible training residuals once, then refit. Validation and test are not trimmed by the outlier rule. Valid rare cars can be removed too; cleaning does not imply greater accuracy.
- Bayesian mode: separate Gaussian make/model-within-make/age corrections fitted to the new models’ OOF log residuals. Validation calibrates historical outcome ranges; test measures coverage. This is not a joint two-market likelihood. Student-t/mixture LOO comparisons and genuine Tobit bounds are not implemented.

The frozen encoder never learns price labels. This release is not a newly fine-tuned supervised MiniLM backbone. The original Kaggle three-fold supervised predictions are preserved and verified; they are not silently reused as nested five-fold features.

## What the two prices mean

An asking estimate is an advertised seller price before negotiation. The auction estimate concerns a different wholesale market and vehicle population. Their difference is not dealer profit, a promised discount or a retail transaction value.

The default date follows today’s calendar; another date can be selected. Historical anchors are transported using separately fitted age/mileage curves and a dated monthly BLS/FRED used-car market index, with bounded forecasting beyond observations. The monthly index is not a daily transaction feed. Retail CPI is also an explicit wholesale proxy. Newer auction generations require an analogue assumption.

Historical MAE is average absolute dollar error on eligible historical test cars. It is not an error guarantee for an individual car, and does not validate today’s extrapolation. All dollar estimates in the GUI are inverse-transformed. The Bayesian comparison can show improvement or worsening independently for each market.

## Explore the studio

Navigation is horizontal on the landing page. Regular and Bayesian pricing remain separate views. Other panels include the two-market historical lab, Value over time, The science, Evaluation, Model lab, Deal Detective and Outlier lab.

After estimating, **Build all explanation graphs** opens the complete current-model suite:

- Complete-function grouped dollar SHAP waterfall.
- Native log-price TreeSHAP beeswarm and reference-gallery importance.
- SHAP dependence and individual mileage sensitivity.
- Native decision paths across nearby fixture cars.
- Year × mileage price sensitivity surface.
- Technical reconstruction checks and downloadable receipt.

The dollar plots include the selected date and optional Bayesian layer. Native SHAP plots explain the historical XGBoost score. Sensitivity surfaces are not SHAP interaction values or causal effects. The small historical reference gallery is not a representative market sample. MiniLM phrase analysis reruns the current complete asking-price function when that release is active and checks that its endpoint matches the displayed price.

The value timeline builds automatically, retains its graph across reruns and updates with mileage assumptions. Past values are reconstructed model scenarios, not observed sales of that car.

BART’s optional overview receives the advert, unverified seller description, valuation and brief evaluation-method background. Deterministic source facts and exact figures remain available separately; an abstractive summary can omit or misstate facts and never changes a valuation.

Deal Detective includes advert import/OCR, historical comparables, buyer briefing, a price challenge and explicit image/link/QR sharing. Analytics are opt-in; GA action events use finite panel/action identifiers rather than seller text or vehicle input values.

## Run and verify

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The release workflow verifies original data identities, rebuilds the OOF models and training trim, fits matching Bayesian posteriors, writes held-out reports and tests real inference under package/memory budgets before publishing. Regression tests cover two-price target exclusion, complete-model explanation reconstruction, navigation, sharing, the persistent timeline, BART source construction and analytics consent. Tests on synthetic fixtures verify code, not real-world accuracy.

## Data and licensing

Asking: original Craigslist Used Cars dataset, with original cohort and text-training evidence supplied through [botsandbolts/pricing on Kaggle](https://www.kaggle.com/datasets/botsandbolts/pricing/). Auction: Kaggle/tunguz Used Car Auction Prices, reported 2014–2015 sales. These are different historical datasets.

Software: MIT, copyright Amit Steinberg. Datasets, pretrained models and dependencies retain their respective licences. The author’s LinkedIn link is in the GUI footer.
