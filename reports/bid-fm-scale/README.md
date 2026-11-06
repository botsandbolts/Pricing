# FM family rank scale: original wholesale auction data

Five architectures, 26 registered full-vs-Lemon variants. Only original
validation used to choose the winner, rank, cohort, and blend coefficient.
Winner: deepfm32_full; original validation MAE USD 1725.90.
Original test MAE USD 1877.61; matched Lemon XGBoost USD 1920.38.
Published full-data XGBoost test MAE USD 1800.96; aggregate unpaired.

## Full versus Lemon, family best on original validation

| Family | Cohort | Best variant | Rank | Validation USD MAE |
|---|---|---|---:|---:|
| fm | full | fm64_full | 64 | 1939.20 |
| fm | lemon | fm16_lemon | 16 | 1897.61 |
| ffm | full | ffm8_full | 8 | 1988.47 |
| ffm | lemon | ffm16_lemon | 16 | 1868.76 |
| hofm3 | full | hofm316_full | 16 | 1888.46 |
| hofm3 | lemon | hofm332_lemon | 32 | 1940.00 |
| fieldgate | full | fieldgate16_full | 16 | 1980.27 |
| fieldgate | lemon | fieldgate32_lemon | 32 | 2003.64 |
| deepfm | full | deepfm32_full | 32 | 1725.90 |
| deepfm | lemon | deepfm16_lemon | 16 | 1798.50 |

No models deployed; test is evaluated only for a single validation-selected model.
Any XGBoost blend uses matched original validation predictions, not leaked in-sample residual targets.
