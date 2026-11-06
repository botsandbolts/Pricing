# Ask-only 25-trial optimization · research and experiment contract

## Why reduce to 25 with only three running at a time?

The earlier 200-arm strategy requested 200 concurrent runners. The replacement GitHub
Actions matrix has exactly 25 Ask trials and `max-parallel: 3`. This setting caps
concurrently executing matrix jobs, **not** total billed runner minutes or account
availability. GitHub's matrix documentation:
https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations

## Base model and unchanged data

The serving model is the **BGE-small-en-v1.5 frozen 256-token embedding**
(384 features) followed by the **full DeepCross/AFT prediction network**.
No Ridge/text-price substitute is introduced. Original 2021 Craigslist cohort:
147,777 training, 31,668 validation, 31,682 untouched test listings.

Every trial reuses the frozen, provenance-checked BGE embeddings and exact source
identities. Internal stopping uses the last 14,777 **training** listings, while
the remaining 133,000 fit model weights. The original validation cohort is scored
once per candidate and the original test is never accessed during selection.

## Evidence-based design

1. **DCN-V2, low-rank mixtures and controlled depth:** mixture-of-experts cross
   layers are especially efficient for tabular interaction learning. This search
   varies cross-layer count 2–4, rank 16–64, experts 1–3, network width and
   embedding projection. This is a hypothesis, not a performance guarantee.
   Source: Wang et al., *DCN V2* (WWW 2021).
   https://doi.org/10.1145/3442381.3450078
2. **Residual MLP comparison:** ResNet-like tabular networks are strong, compact
   baselines; no architecture is universally superior to gradient-boosted trees.
   Five of the 25 trials use a residual MLP with the identical raw BGE input.
   Source: Gorishniy et al., *Revisiting Deep Learning Models for Tabular Data*
   (NeurIPS 2021).
   https://papers.nips.cc/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html
3. **Optimizer and learning-rate practice:** AdamW is the default; learning rate,
   beta1, weight decay, batching, dropout, warmup, cosine/linear decay and gradient
   clipping vary jointly with architecture. Warmup and decay can help but must
   be measured rather than assumed.
   https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/learning-rate
   https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW
4. **Loss aligned with catastrophic dollars:** exact-event log-normal AFT remains
   the main likelihood objective. A small subset of candidates adds a **bounded**
   top-decile dollar-error term, with `tail_weight` 0–0.07. This is a research
   ablation because over-emphasizing atypical asking amounts may be harmful.
   Selection uses the original held-out **absolute dollar CVaR95** regardless
   of objective.
5. **Holdout hygiene:** all 25 run independently with fixed seeds and immutable
   source fingerprints. Hyperparameter search on one validation set can itself
   overfit; use the untouched original test *only once* after choosing a winner,
   then perform native checkpoint serving-parity checks before promoting it.
   https://pmc.ncbi.nlm.nih.gov/articles/PMC11573893/

## What will count as a successful experiment?

- Precisely 25 complete training jobs, with 25 unique deterministic settings.
- Same original validation IDs and historical target prices for all trials.
- Lowest average absolute error among the **worst 5% of errors** ranks first;
  RMSE and MAE are secondaries.
- Native `weights.pt`, `validation.csv.gz`, and `result.json` per job.
- A readable report under `reports/cvar95/ask25-joint/`, clearly flagged
  **partial** if any candidate failed or was missing.
- No automatic production deploy, no 2026 individual-sales accuracy claims,
  and no repeated text-encoder computation.

## Expected output

The report will show which depth/rank/optimizer/tail-loss choices won on the
original 2021 validation set and their CVaR95 improvement versus a re-trained
baseline architecture. A gain there is an experiment, not evidence of improved
November 2026 market performance.
