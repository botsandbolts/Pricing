"""Publish a STRICTLY matched 18-arm neural wholesale auction vs XGBoost audit.

Requires all 18 completed arms plus the exact frozen Lemon XGBoost
per-row holdout artifact. A failed/incomplete sweep cannot publish a winner.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha_ids(ids):
    return hashlib.sha256("\n".join(sorted(map(str, ids))).encode()).hexdigest()


def _scores(truth, pred):
    y, p = np.asarray(truth, float), np.asarray(pred, float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Invalid benchmark predictions")
    error = p - y
    return {"rows": len(y), "mae_usd": float(np.abs(error).mean()),
            "rmse_usd": float(np.sqrt((error ** 2).mean())),
            "mean_bias_usd": float(error.mean()),
            "rmse_log": float(np.sqrt(np.square(np.log(p) - np.log(y)).mean()))}


def reference(path):
    base = pd.read_csv(path, dtype={"row_id": str})
    required = {"row_id", "split", "observed_price", "predicted_price"}
    if not required.issubset(base):
        raise ValueError("XGBoost baseline lacks exact heldout evidence")
    base = base.loc[base.split.isin(("validation", "test")), list(required)].copy()
    if base.row_id.duplicated().any():
        raise ValueError("Duplicate baseline holdouts")
    return base.set_index("row_id")


def paired_interval(err_model, err_ref, seed=42):
    diff = np.asarray(err_model, float) - np.asarray(err_ref, float)
    rng = np.random.default_rng(seed)
    means = []
    # Same listing sampled for the two models: paired bootstrap.
    for _ in range(300):
        indices = rng.integers(0, len(diff), size=len(diff))
        means.append(float(diff[indices].mean()))
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def analyze(root, base_file, published_full, out):
    root, out = Path(root), Path(out)
    results = sorted(root.rglob("result.json"))
    if len(results) != 18:
        raise ValueError(f"Need all 18 independent completed arms, found {len(results)}")
    baseline = reference(base_file)
    seen = set()
    rows = []
    for path in results:
        record = json.loads(path.read_text())
        arm = (record["architecture"], record["loss"], record["train_cohort"])
        if arm in seen or record["status"] != "completed" or record.get("market") != "bid":
            raise ValueError("Duplicated/incomplete neural arm")
        seen.add(arm)
        predict = pd.read_csv(path.with_name("predictions.csv.gz"), dtype={"row_id": str})
        if predict.row_id.duplicated().any() or len(predict) != len(baseline):
            raise ValueError("Missing/duplicated prediction identities")
        predict = predict.set_index("row_id")
        if set(predict.index) != set(baseline.index):
            raise ValueError("Neural vs XGBoost holdout IDs do not match")
        aligned = predict.reindex(baseline.index)
        if not (aligned.split.to_numpy() == baseline.split.to_numpy()).all():
            raise ValueError("Neural vs XGBoost phase disagreement")
        if not np.allclose(aligned.observed_price.to_numpy(float),
                           baseline.observed_price.to_numpy(float), atol=.05, rtol=1e-6):
            raise ValueError("The evaluated targets differ: not a valid comparison")
        for split in ("validation", "test"):
            base = baseline.loc[baseline.split.eq(split)]
            current = aligned.loc[aligned.split.eq(split)]
            if _sha_ids(current.index) != record["holdout_row_sha256"][split]:
                raise ValueError("Canonical holdout fingerprint mismatch")
            y = current.observed_price.to_numpy(float)
            pred = current.predicted_price.to_numpy(float)
            ref = base.predicted_price.to_numpy(float)
            score = _scores(y, pred)
            if not np.isclose(score["mae_usd"],
                              record["original_validation" if split == "validation" else "original_test"]["mae_usd"],
                              atol=.05):
                raise ValueError("Metric/result artifact disagreement")
            diff = np.abs(pred - y)
            rdiff = np.abs(ref - y)
            rows.append({
                "architecture": arm[0], "loss": arm[1],
                "train_cohort": arm[2], "partition": split,
                "train_rows_fit": record["train_fit"],
                "best_epoch": record["best_epoch"], **score,
                "reference_model": "XGBoost log-price quantile 0.5 + frozen training-only Lemon trim (no hierarchy)",
                "reference_mae_usd": float(rdiff.mean()),
                "delta_mae_vs_xgboost_usd": float((diff - rdiff).mean()),
                "fraction_listings_beating_xgboost": float((diff < rdiff).mean()),
                "paired_delta_95pct_bootstrap_usd": paired_interval(diff, rdiff),
            })
    frame = pd.DataFrame(rows)
    val = frame.loc[frame.partition == "validation"].sort_values(
        ["mae_usd", "architecture", "loss", "train_cohort"]).reset_index(drop=True)
    if len(val) != 18 or len(frame) != 36:
        raise ValueError("Incomplete validation/test arms")
    ranked = val[["architecture", "loss", "train_cohort"]].copy()
    ranked["validation_rank"] = np.arange(1, 19)
    frame = frame.merge(ranked, on=["architecture", "loss", "train_cohort"], validate="many_to_one")
    frame = frame.sort_values(["validation_rank", "partition"])
    published = json.loads(Path(published_full).read_text())
    if published["scores"]["validation"]["n"] != 82380 or published["scores"]["test"]["n"] != 83603:
        raise ValueError("Published full XGBoost sample sizes do not match original cohort")
    champion = val.iloc[0]
    summary = {
        "status": "complete", "market": "bid", "target": "2014/15 reported wholesale auction sale prices in XGBoost's 2015-normalized USD",
        "comparison_rule": "Validation MAE chooses architecture/loss/cohort; test never chooses a winner",
        "winner_by_original_validation": {
            "architecture": champion.architecture, "loss": champion.loss,
            "train_cohort": champion.train_cohort,
            "validation_mae_usd": float(champion.mae_usd)},
        "architectures": ["resmlp", "deepcross", "fttransformer"],
        "losses": ["huber", "quantile50", "aft"],
        "train_cohorts": ["full", "lemon"],
        "tested_arms": 18,
        "holdout_rows": {"validation": 82380, "test": 83603},
        "frozen_xgboost_lemon_reference": {
            "type": "Exact per-row paired validation/test",
            "validation": _scores(baseline.loc[baseline.split.eq("validation"), "observed_price"],
                                  baseline.loc[baseline.split.eq("validation"), "predicted_price"]),
            "test": _scores(baseline.loc[baseline.split.eq("test"), "observed_price"],
                            baseline.loc[baseline.split.eq("test"), "predicted_price"])},
        "published_full_xgboost_reference": {
            "type": "Published original fivefold summary only; per-row identity not independently rechecked",
            "validation": published["scores"]["validation"],
            "test": published["scores"]["test"]},
        "methodology": {
            "split": "Original VIN-disjoint chronological partitions; no original holdout filtering",
            "preprocessing": "Fitted only on original training; pre-sale numeric and categorical facts; NO descriptions",
            "regularization": "AdamW weight decay, dropout, layer norm, gradient clipping, mild input noise, cosine LR and early stopping",
            "aft": "Exact log-normal likelihood only: observed auction sales are not censored",
            "hierarchy": "None",
            "evaluation": "All arms and frozen Lemon XGBoost assessed on identical row_ids and observed prices",
            "uncertainty": "Paired listing bootstrap for MAE difference, assumes rows independent; may understate correlated uncertainty",
            "warning": "Original full-cohort bid XGBoost reference is an aggregate with same counts, not an audited per-listing paired comparator."}}
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps({
        **summary, "results": frame.to_dict(orient="records")}, indent=2, allow_nan=False))
    frame.to_csv(out / "comparison.csv", index=False)
    (out / "README.md").write_text(
        "# Neural historical auction benchmark\n\n"
        "18 genuinely fitted wholesale neural experiments, 3 architectures × 3 losses × "
        "2 train-only outlier policies. Models and individual predictions remain "
        "in the immutable GitHub Actions run artifacts. Rows in comparison.csv "
        "include the original validation and test sets, and the MAE difference "
        "relative to the matched XGBoost Lemon benchmark. Test errors are audit "
        "results, NOT used to choose the winning architecture.\n")
    print("Winner (validation only):", summary["winner_by_original_validation"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="training-cache/neural-bid-results")
    p.add_argument("--reference", default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--full-xgboost", default="models/fivefold-live/bid/evaluation.json")
    p.add_argument("--output", default="reports/neural-bid-benchmark")
    args = p.parse_args()
    analyze(args.results, args.reference, args.full_xgboost, args.output)
