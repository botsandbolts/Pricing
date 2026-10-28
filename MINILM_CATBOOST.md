# MiniLM + CatBoost normal AFT

The requested default is a stacked asking-price model: fine-tuned
`sentence-transformers/all-MiniLM-L6-v2` reads only the seller description and
outputs a scalar dollar `predicted_text_price`. CatBoost's normal SurvivalAft
model combines it with structured vehicle facts and native categorical labels.
There is no make/model transformer. Updating the UI label alone is not a refit:
the trained CatBoost feature schema must actually contain the text-price feature.

## Cross-fitting and assessment

The original successful Kaggle continuation is
`botsandbolts/notebooka6bded45ca`, saved run 362713451. It contains the completed
training-only three-fold OOF export (148,290 rows), separate full-cohort
three-fold OOF export (211,844 rows), and both trained MiniLM heads.
The original cohort SHA256 is
`746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12`.

For assessment, CatBoost training uses only training-partition OOF text prices;
validation and test descriptions are predicted with the training-only MiniLM
head. For deployment refitting, CatBoost receives the separate full-cohort OOF
features and serves the matching full-cohort MiniLM head. No deployment head is
used to manufacture supposedly held-out assessment features.
`pricing.suite_text.validated_features` checks cohort identity, exact row order,
all fold assignments, finite nonnegative prices, model revision and provenance.

The previous requested highest-residual quartile masks and OutlierTree flags
remain fixed. CatBoost is newly fitted on retained cars; the MiniLM checkpoint
itself retains its original fitting population. Because validation/test labels
were used to select retained cars, filtered scores are optimistic. Original
unfiltered test diagnostics also use a previously examined benchmark. Neither
is an unbiased new external test, and neither measures the deployment refit's
out-of-sample accuracy.

CatBoost labels are exact positive dollar bounds `[price, price]`. SurvivalAft
logs them internally. Raw outputs are log-price locations; exponentiation gives
a log-normal conditional median, not the arithmetic mean. No interval-censoring
mechanism has been verified in these sources.

## Auction limitation

The bid source is **Used Car Auction Prices (Kaggle/tunguz)**, with no free-text
seller descriptions. Its existing CatBoost-AFT model therefore remains tabular.
It is not relabelled as a MiniLM-trained bid model. Craigslist is the asking
source. Both source names remain visible in the GUI.

## Reproduction

Recover the original OOF CSV exports and heads from the saved Kaggle output.
Use `scripts.recover_minilm_features` for resumable held-out inference, or the
GPU inference notebook. After feature validation:

```bash
python -m scripts.train_trimmed_catboost_aft \
  --side ask --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --text-features training-cache/minilm-recovered \
  --text-model models/minilm-cleanlab/text-model \
  --output models/minilm-catboost-aft
```

The GUI prefers a completed `models/minilm-catboost-aft` pair over tabular
baselines. Loading rejects a missing text head, mismatched fingerprint or
incompatible feature schema. Native TreeSHAP includes the scalar text feature;
it does not explain individual MiniLM tokens. The six comparison graphs hold
the submitted text-price signal fixed while changing tabular facts.

## Recovered feature checkpoint

The verified feature archive is preserved as `MiniLM_verified_features.zip`,
Library ID `libfile_87be7d639550819186bfe52e9e187a70` (20,715,365 bytes).
Held inference was regenerated on Kaggle notebook
`botsandbolts/notebookf1cefc6227`, saved run **363162642**.
The interactive GPU session was explicitly stopped after saving the output.
The deployed and recovered MiniLM backbone, head, target scaling, windowing and
forward/prediction code were checked for equality. The full-data CatBoost
prefit is withheld from deployment until separate assessment completes; it is
reused only when the source, feature checksum, parameters and selected tree count
match. A prefit alone never creates a ready manifest.
