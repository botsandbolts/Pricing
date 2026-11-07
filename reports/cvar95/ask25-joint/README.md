# Ask · 25 combined architecture and hyperparameter experiments

**Status:** complete
**Audited successful trials:** 25/25
**Selection:** original-validation dollar CVaR95 (worst 5%).
RMSE and MAE are secondary. Original test is untouched.
No 2026-sale accuracy established. No production model changes.

| Rank | Trial | Architecture | CVaR95 (USD) | RMSE (USD) | MAE (USD) |
|---:|---:|---|---:|---:|---:|
| 1 | 15 | resmlp | 18,069 | 5,718 | 2,836 |
| 2 | 16 | dcnv2 | 18,308 | 5,809 | 2,846 |
| 3 | 18 | dcnv2 | 18,636 | 5,858 | 2,893 |
| 4 | 23 | dcnv2 | 18,748 | 5,893 | 2,861 |
| 5 | 8 | dcnv2 | 18,927 | 5,966 | 2,921 |
| 6 | 1 | dcnv2 | 18,987 | 6,067 | 3,087 |
| 7 | 9 | dcnv2 | 19,004 | 6,000 | 3,003 |
| 8 | 19 | dcnv2 | 19,009 | 5,976 | 3,054 |
| 9 | 0 | dcnv2 | 19,063 | 6,001 | 3,003 |
| 10 | 21 | dcnv2 | 19,095 | 6,059 | 3,021 |
| 11 | 12 | dcnv2 | 19,106 | 6,021 | 3,034 |
| 12 | 24 | resmlp | 19,132 | 5,926 | 2,892 |
| 13 | 13 | dcnv2 | 19,159 | 6,005 | 3,021 |
| 14 | 14 | dcnv2 | 19,247 | 6,022 | 2,978 |
| 15 | 3 | dcnv2 | 19,291 | 6,130 | 3,100 |
| 16 | 6 | dcnv2 | 19,330 | 6,238 | 3,288 |
| 17 | 17 | dcnv2 | 19,401 | 6,122 | 3,052 |
| 18 | 7 | dcnv2 | 19,452 | 6,117 | 3,024 |
| 19 | 11 | dcnv2 | 20,005 | 6,263 | 3,188 |
| 20 | 2 | dcnv2 | 20,267 | 6,333 | 3,211 |
| 21 | 10 | resmlp | 20,444 | 6,345 | 3,249 |
| 22 | 22 | dcnv2 | 20,470 | 6,365 | 3,194 |
| 23 | 20 | resmlp | 20,730 | 6,367 | 3,285 |
| 24 | 5 | resmlp | 21,709 | 6,909 | 3,888 |
| 25 | 4 | dcnv2 | 23,198 | 7,288 | 4,007 |

## Comparison against re-fitted original architecture
Validation CVaR95 change: $994 (+5.21% lower). This is a same-study architecture comparison, NOT a production-model or 2026 market accuracy guarantee.

## Required final gate
Choose a single candidate using original validation CVaR95. Assess its native weights on the untouched original test once, check inference parity, cold-start CPU/RAM and current-date scenario assumptions before considering deployment.
