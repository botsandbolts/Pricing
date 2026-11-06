# DeepCross / exact-log-normal AFT architecture and hyperparameter sweep

Purpose: optimize the best neural family from the original historical Craigslist asking-price benchmark without changing its data, validation, test, loss definition, or production deployment.

**36 independent training runs:** 18 predeclared architectural/optimizer choices × 2 fixed original text-feature sets (train-only TF-IDF→24 SVD components; the same set with frozen, audited two-out-of-bag MiniLM price scalar).

Search axes: low-rank cross network depth 1/3/5, cross projection rank 16/32/64, learned 2- or 4-expert cross mixtures, cross residual scale, deep width 128/256/384, deep residual blocks 2/3, SiLU in deep/out head, dropout 0.12/0.25/0.35, AdamW learning rate 0.0006/0.001/0.0015, weight decay 0.025/0.08, batch 768/1024. The original DeepCross implementation is included unchanged as the `baseline` arm.

All models use full original **147,777** training rows with the same chronological last **14,777** internal stopping rows; 133,000 rows update weights. Validation has 31,668 rows and test has 31,682. The validation MAE alone selects the winner, and paired original test comparisons are reported after selection. All model checkpoints remain as GitHub Actions artifacts. No model is deployed automatically.

Source data are not re-downloaded: the job fetches the successful `neural-ask-frozen-features` artifact (run 42224932982) and the successful `ask-minilm-2oob-features` artifact (run 42244440154). The MiniLM variant fits no new price heads or pretrained encoders; it reuses the verified two-out-of-bag scalar. Full-data XGBoost is an aggregate comparison, whereas Lemon XGBoost and the two previously trained DeepCross+AFT models are exact per-row matched comparators.

**Caveat:** Hyperparameter searches can overfit validation. These are single-seed exploratory results; confirm the best 2–3 configurations on additional random seeds and a fresh time-based holdout before production use.

Workflow: `.github/workflows/deepcross-aft-tuning.yml`  
Output after verification: `reports/deepcross-aft-tuning/report.json` and `leaderboard.csv`.
