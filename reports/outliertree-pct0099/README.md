# OutlierTree rerun: pct_outliers=0.099

All four full-data audits completed. No cars were deleted and no pricing model was refitted.

| Dataset | Fit scope | Cars | Flagged | Percentage | Primary price flags |
|---|---|---:|---:|---:|---:|
| ask | assessment | 147,777 | 19 | 0.0129% | 0 |
| ask | full_cohort | 211,127 | 31 | 0.0147% | 3 |
| bid | assessment | 384,198 | 74 | 0.0193% | 3 |
| bid | full_cohort | 550,181 | 74 | 0.0135% | 3 |

Previous completed-run log at pct_outliers=0.01 recorded 19 full-cohort asking flags and 63 full-cohort auction flags. The new totals are 31 and 74 respectively. Previous row-level outputs were cleared from the workspace; the comparison is aggregate counts only, not a matched-row sensitivity analysis.

The full asking fit has 14 primary age flags, 14 mileage flags and 3 price flags. Price examples include a 2015 Jeep Renegade listed at $140,000, a 2013 Chevrolet Spark listed at $123,456, and a 2015 Nissan Altima listed at $1,995. These may be mistakes, placeholders, damaged cars or legitimate unusual records; the audit alone does not establish which.

Age flags include modern Fords with unknown model names assigned to conditional groups dominated by classics. Review those groups carefully rather than treating normal modern cars as errors.

All input cohorts match the SHA256 and eligibility counts from the existing Cleanlab audit. Categories are native, with rare model-name pooling recorded per fitting partition. See [methodology](../../OUTLIERTREE.md) and the per-stage report/flag exports for exact settings and conditions.

The full-cohort fit is descriptive, including validation/test, and cannot be used to filter held-out evaluation. pct_outliers=0.099 is not a 9.9% deletion quota. No new MAE or model-quality claim is made.
