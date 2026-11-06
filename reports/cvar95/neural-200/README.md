# Neural experiment grid · CVaR95-first

**Status:** complete
**Trials:** 200/200; 50 each for Ask hyperparameters, Bid hyperparameters, Ask structure, Bid structure.

Source-time validation only. No 2026-sale validation or automatic deployment.
All 200 configurations are deterministic and original TEST stays unexamined.

| Group | Valid trials | Best trial | CVaR95 (USD) | RMSE (USD) | MAE (USD) |
|---|---:|---:|---:|---:|---:|
| ask_hparams | 50 | 39 | 18,275 | 5,795 | 2,820 |
| bid_hparams | 50 | 18 | 8,721 | 2,976 | 1,684 |
| ask_structure | 50 | 44 | 18,393 | 5,782 | 2,847 |
| bid_structure | 50 | 6 | 8,803 | 2,969 | 1,719 |

## Required next gate
Choose challengers using original validation CVaR95; compare the chosen winner(s) on the untouched original test only once, then verify full native-checkpoint serving parity and current-date transport limitations.
Do not call a candidate production-ready solely because it wins validation.
