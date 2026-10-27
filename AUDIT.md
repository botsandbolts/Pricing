# Studio model audit and description examples

The calculator offers 400 real historical seller descriptions, searchable by year,
make, model and snippet. Selecting one copies the entire cleaned description and
matching vehicle facts into editable fields. It clears the previous estimate;
click **Estimate asking price** to calculate. No asking-price label is loaded into
the calculator. These examples come only from the original training partition;
they are demonstrations, not independent evaluation cars.

Reproduce from the public Used Cars Dataset archive (Zenodo record 10457828):

```bash
python -m scripts.prepare_full /path/to/Used-Cars-Dataset.zip --output training-cache/full-cohort.jsonl
python -m scripts.build_description_examples training-cache/full-cohort.jsonl
python -m scripts.evaluate_studio training-cache/full-cohort.jsonl
```

The evaluator runs the installed MiniLM regression weights on **text alone**, then
feeds the resulting dollar-valued `predicted_text_price` and categorical/numeric
vehicle fields into the installed LightGBM. It evaluates the text-only price too.
No target prices are serialized into descriptions; normal inference cleaning and
windowing are used. MAE and RMSE are measured in dollars after inverse transforms.
Isolation Forest does not remove validation or test rows.

New validation and test partitions split the original chronological test pool at
its midpoint timestamp, keeping identical timestamps together. A deterministic SHA-256 ordering selects
1,000 rows per partition for this CPU audit (2,000 total); pass `--per-phase 0`
to recompute all 31,777 original test rows. Per-partition 95% MAE intervals use
2,000 iid row-bootstrap resamples; correlated listings can make these intervals
too narrow. Original training
and tuning-validation rows are excluded. Exact row membership is saved in
`reports/studio-audit/audit-split.csv`. This is a **reused historical holdout**, not
newly collected data or a fresh untouched benchmark. Do not tune against the new
test partition. No model is retrained or selected by this audit.

MiniLM assessment weights were fitted only on the original training partition.
The deployed tabular and robust ensembles were refitted on the full eligible
cohort. Their recomputed audit scores therefore have **training overlap**, are
labelled in-sample, and must not be used to rank generalization performance. The
GUI separately displays their original pre-refit chronological test assessments.
A genuinely new held-out comparison of all architectures requires either new
eligible listings or refitting every learned component on a common new split.

The compact report records dataset and manifest SHA-256 values. Row-wise
predictions remain local and inference checkpoints allow continuation. Examples
have monetary expressions, URLs, email addresses and common VIN/phone formats
redacted. They remain historical unverified seller claims, not current valuations
or confirmed sales prices.

Source attribution: Tom, *Used Cars Dataset*, Zenodo, 2024, DOI
10.5281/zenodo.10457828, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
UI examples are selected, normalized and redacted adaptations.
