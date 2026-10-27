# Fresh Cleanlab rerun: asking and auction data

All four audits were recomputed in a new output directory from freshly downloaded
source archives and fully rebuilt cohorts. No row-level audit cache was restored.
The asking cohort SHA-256 matches the original preparation. Auction preparation
again produced 550,291 canonical sales; 110 known asking VIN overlaps were removed.

| Market | Audit stage | Input rows | Retained | Flagged |
|---|---|---:|---:|---:|
| Asking | Training-only assessment | 147,777 | 147,777 | 0 |
| Asking | Full deployment candidates | 211,127 | 211,127 | 0 |
| Auction | Training-only assessment | 384,198 | 384,198 | 0 |
| Auction | Full deployment candidates | 550,181 | 550,181 | 0 |

Cleanlab 2.9.0 selected zero exclusions automatically. No fixed percentage or manual
threshold was imposed. The assessment audits excluded validation/test rows. The
separate deployment audits used all eligible fitting candidates after assessment.
No text-price, model-prediction or MMR feature entered these vehicle-facts audits.

All four output CSV checksums and input/configuration fingerprints **exactly match**
the previously fitted models' audits. Consequently, the existing AFT weights
already use the same retained samples; this rerun does not require changing weights
or substituting training scores for held-out evaluation. Zero flags does not prove
every label is correct; it is this configured Cleanlab procedure's result.

`summary.json` and the four `label-audit.json` files record the complete settings,
counts and hashes. Large reconstructable row-level CSVs and duplicate category
schemas are not committed.

```bash
python -m scripts.audit_market_labels \
  --ask training-cache/full-cohort.jsonl \
  --bid training-cache/auction-cohort.jsonl \
  --output reports/cleanlab-rerun-NEW-RUN
```
