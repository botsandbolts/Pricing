# Neural experiment grid · CVaR95-first

**Status:** partial_or_invalid
**Trials:** 66/200; 50 each for Ask hyperparameters, Bid hyperparameters, Ask structure, Bid structure.

Source-time validation only. No 2026-sale validation or automatic deployment.
All 200 configurations are deterministic and original TEST stays unexamined.

| Group | Valid trials | Best trial | CVaR95 (USD) | RMSE (USD) | MAE (USD) |
|---|---:|---:|---:|---:|---:|
| ask_hparams | 48 | 18 | 18,081 | 5,789 | 2,883 |
| bid_hparams | 17 | 0 | 8,804 | 3,005 | 1,694 |
| ask_structure | 1 | 1 | 23,981 | 8,301 | 5,617 |
| bid_structure | 0 | — | — | — | — |

## Required next gate
Choose challengers using original validation CVaR95; compare the chosen winner(s) on the untouched original test only once, then verify full native-checkpoint serving parity and current-date transport limitations.
Do not call a candidate production-ready solely because it wins validation.

## Missing trials
[{'group': 'ask_hparams', 'trial': 26}, {'group': 'ask_hparams', 'trial': 47}, {'group': 'ask_structure', 'trial': 0}, {'group': 'ask_structure', 'trial': 2}, {'group': 'ask_structure', 'trial': 3}, {'group': 'ask_structure', 'trial': 4}, {'group': 'ask_structure', 'trial': 5}, {'group': 'ask_structure', 'trial': 6}, {'group': 'ask_structure', 'trial': 7}, {'group': 'ask_structure', 'trial': 8}, {'group': 'ask_structure', 'trial': 9}, {'group': 'ask_structure', 'trial': 10}, {'group': 'ask_structure', 'trial': 11}, {'group': 'ask_structure', 'trial': 12}, {'group': 'ask_structure', 'trial': 13}, {'group': 'ask_structure', 'trial': 14}, {'group': 'ask_structure', 'trial': 15}, {'group': 'ask_structure', 'trial': 16}, {'group': 'ask_structure', 'trial': 17}, {'group': 'ask_structure', 'trial': 18}, {'group': 'ask_structure', 'trial': 19}, {'group': 'ask_structure', 'trial': 20}, {'group': 'ask_structure', 'trial': 21}, {'group': 'ask_structure', 'trial': 22}, {'group': 'ask_structure', 'trial': 23}, {'group': 'ask_structure', 'trial': 24}, {'group': 'ask_structure', 'trial': 25}, {'group': 'ask_structure', 'trial': 26}, {'group': 'ask_structure', 'trial': 27}, {'group': 'ask_structure', 'trial': 28}]
