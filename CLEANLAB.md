# Cleanlab migration: asking and auction-sale models

New training uses **cleanlab 2.9.0 regression CleanLearning**, replacing Isolation
Forest in `pricing.train`, `carbids.train` and `marketbridge.robust`. The structural
auction trainer also supports Cleanlab (enabled by default); its previous version
had no Isolation Forest. All changes are in Pricing, not the separate CarBids repo.

The shared auditor (`pricing/label_quality.py`) uses observed **training log1p
prices** and structured vehicle facts. A cloneable categorical LightGBM auditor
supports Cleanlab's internal CV, three uncertainty resampling rounds, and aleatoric
uncertainty estimation. Cleanlab estimates a suspected-noise fraction internally;
there is no requested removal percentage. Suspicion does not establish bad data:
an accurately recorded optimistic asking price can still be flagged.

Text-price stacking features are excluded from auditing: ordinary precomputed OOF
text predictions can otherwise introduce indirect label leakage during a second
cross-validation. The text-to-price branch still uses its existing OOF protocol
for final LightGBM fitting. Make/model remain categorical, without transformers.

Assessment auditing sees training rows only. Validation and test rows remain
unfiltered and outside all auditor fits and threshold selection. After assessment,
a separate audit/refit can use the full eligible cohort. Assessment and deployment
reports are separate. Full-data refits have no independent holdout.

Each audit saves row IDs, observed prices, out-of-sample audit predictions, quality
scores, flags and retained rows in `label-audit.csv`, with parameters and checksums
in `label-audit.json`. Matching audits resume; changed inputs, code or settings
reject a cache. Errors fail training rather than falling back silently.

## Run new training

CPU auditing / tabular training dependencies:

```bash
pip install -r requirements-cleanlab.txt
python -m marketbridge.robust --cohort training-cache/full-cohort.jsonl --structural models/structural-transfer --output models/robust-asking-cleanlab --cleanlab on
```

For MiniLM, use a GPU environment and `requirements-training.txt`:

```bash
python -m pricing.train --data training-cache/full-cohort.jsonl --output artifacts/car-price-studio-cleanlab --device cuda --folds 3 --epochs 2
python -m carbids.train --data data/auction-presale-text.jsonl --output artifacts/carbids-cleanlab --device cuda --folds 3 --epochs 2
```

The auction text trainer requires genuine pre-sale descriptions and documented
reported sale targets. The public wholesale dataset has no descriptions; do not
substitute generated text or ask prices. The usual provenance checks still apply.
The updated Kaggle notebooks embed the new source and install Cleanlab. Start a
fresh Cleanlab output; legacy Isolation Forest optimizer checkpoints are rejected
under changed configuration/source. Existing legacy runs are not altered remotely.

To train the actual structural auction model with Cleanlab:

```bash
python -m marketbridge.structural --cohort data/cohort.jsonl --index models/structural-transfer/used-car-index.csv --exclude-pricing training-cache/full-cohort.jsonl --output models/structural-transfer-cleanlab --cleanlab on
```

Keep this new upstream model separate. Downstream models must be retrained against
its exact model ID before deployment. Existing version checks intentionally reject
mixing a new auction model with old asking weights. `--cleanlab off` permits an
unfiltered control in both tabular/structural trainers.

## Serving and evaluation status

**No new full-data Cleanlab weights or MAE are claimed by this code migration.**
The installed MiniLM assessment and historical filtered ensemble remain Isolation
Forest models; their manifests, weights and assessments are unchanged. The GUI
labels the MiniLM model as legacy and explains that retraining is required. It
recognizes ready Cleanlab MiniLM candidates at `models/minilm-cleanlab` or
`artifacts/car-price-studio-cleanlab` and a ready ensemble at
`models/robust-asking-cleanlab`. Pending folders never become pricing options.
Best-ensemble selection additionally requires the saved promotion flag.

References: [Cleanlab regression tutorial](https://docs.cleanlab.ai/stable/tutorials/regression.html)
and [CleanLearning API](https://docs.cleanlab.ai/stable/cleanlab/regression/learn.html).
