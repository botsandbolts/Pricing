# FT-Transformer Huber Lemon bid tuning

Independent 29-trial architectural and regularization search centered on the
best completed FT-Transformer/HUBER/Lemon candidate.

Reuses the **original** 550,181 historical wholesale auction price observations,
their VIN-disjoint chronological split (384,198 full training, 82,380 validation,
83,603 untouched test) and the **exact frozen 25% training-only Lemon** policy.
Lemon training: 288,148 rows; fit subset 259,334; internal stopper 28,814.

Architecture: attention token width (32/48/64/96), 1-4 transformer layers,
2/4/8 heads, feed-forward multiplier 2/4/6, output head width, pre/post norm.
Regularization: attention and MLP dropout, token dropout, AdamW weight decay,
numeric jitter, learning rate, batch size, Huber beta, and combinations.

**Test isolation:** Train jobs produce validation predictions only. Once all 29
trials pass exact original ID / price checks, the report chooses the lowest
validation MAE and evaluates the selected ONE model on original test. No test
peeking across the architecture search. Strict exact-ID paired benchmarking
against frozen Lemon XGBoost. Full XGBoost is a published aggregate comparator.

Experiment remains separate from the production pricing app.

Workflow: .github/workflows/ft-bid-tuning.yml
Final report: reports/ft-bid-tuning/comparison.json
