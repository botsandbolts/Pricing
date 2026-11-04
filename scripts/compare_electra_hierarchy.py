"""ELECTRA-only Bayesian features; identical frozen OOF XGBoost in both arms."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from pricing import robust_hierarchy as baseline
from pricing import robust_electra_hierarchy as augmented
from pricing.bayesian_hierarchy import interval, save
from pricing.hierarchical import scores
from pricing.electra_embeddings import merge_shards, file_sha256
from scripts.compare_robust_bayes import prepare, paired_bootstrap


def main():
    out = Path("electra-hierarchy-results")
    out.mkdir(exist_ok=True)
    source = Path("frozen-baseline/ask/hierarchy-input.csv.gz")
    d = prepare(source, "ask")
    manifest = json.loads(Path("training-cache/electra-manifest.json").read_text())
    ids, all_vectors, receipts = merge_shards("electra-shards", manifest)
    lookup = pd.Index(ids).get_indexer(d.row_id.astype(str))
    if (lookup < 0).any():
        raise ValueError("Missing ELECTRA row for frozen XGBoost input")
    vectors = all_vectors[lookup]
    del all_vectors
    calibration = d.split.eq("calibration").to_numpy()
    heldout = ~calibration
    reducer = UMAP(
        n_components=32, n_neighbors=30, min_dist=.1, metric="cosine",
        n_epochs=200, random_state=42, transform_seed=42, n_jobs=1, low_memory=True,
    )
    print("Fit cosine UMAP on calibration ELECTRA vectors only", flush=True)
    coords = np.empty((len(d), 32), dtype=np.float32)
    coords[calibration] = reducer.fit_transform(vectors[calibration])
    coords[heldout] = reducer.transform(vectors[heldout])
    scaler = StandardScaler().fit(coords[calibration])
    coords = scaler.transform(coords)
    if not np.isfinite(coords).all():
        raise ValueError("Non-finite ELECTRA/UMAP coordinates")
    for j in range(32):
        d[f"electra_umap_{j}"] = coords[:, j]
    joblib.dump(reducer, out / "calibration-umap.joblib")
    joblib.dump(scaler, out / "calibration-scaler.joblib")
    (out / "encoding-audit.json").write_text(json.dumps({
        "manifest": manifest, "shards": receipts,
        "frozen_xgboost_input_sha256": file_sha256(source),
        "aligned_rows": len(d), "calibration_rows": int(calibration.sum()),
    }, indent=2))
    cal = d.loc[calibration].copy()
    select, interval_cal = baseline.split_validation(d.loc[d.split.eq("validation")])
    test = d.loc[d.split.eq("test")].copy()
    rows, predictions, details = [], {}, {}
    for arm, module in [("baseline", baseline), ("electra_hierarchy", augmented)]:
        candidates, bundles = [], {}
        for likelihood in ["gaussian", "student_t", "mixture"]:
            print(arm, likelihood, "fitting", flush=True)
            meta, arrays = module.fit(cal, likelihood, warmup=240, draws=240)
            pt = module.predict(meta, arrays, select)
            entry = {
                "likelihood": likelihood,
                "validation_mae_usd": scores(select.observed_price, pt.corrected_price)["mae_usd"],
                "max_split_rhat": float(max(np.max(v) for v in meta["rhat"].values())),
            }
            candidates.append(entry)
            bundles[likelihood] = (meta, arrays)
            print(json.dumps(entry), flush=True)
        eligible = [c for c in candidates if np.isfinite(c["max_split_rhat"]) and c["max_split_rhat"] < 1.1]
        details[arm] = {"candidates": candidates}
        (out / f"{arm}-candidates.json").write_text(json.dumps(candidates, indent=2))
        if not eligible:
            raise ValueError(arm + ": no candidate passed convergence; no result published")
        best = min(eligible, key=lambda c: c["validation_mae_usd"])
        meta, arrays = bundles[best["likelihood"]]
        meta, multiplier = module.calibrate_intervals(meta, arrays, interval_cal)
        pt = module.predict(meta, arrays, test)
        low, high = interval(pt, multiplier)
        predictions[arm] = pt.corrected_price.to_numpy()
        result = {
            "point_scores": scores(test.observed_price, pt.corrected_price),
            "test_rows": len(test),
            "test_coverage": float(((test.observed_price >= low) & (test.observed_price <= high)).mean()),
            "selected_likelihood": best["likelihood"],
            "selection_validation_mae_usd": best["validation_mae_usd"],
            "validation_multiplier": float(multiplier),
        }
        meta["frozen_xgboost_input_sha256"] = file_sha256(source)
        if arm == "electra_hierarchy":
            meta["representation"] = manifest
        details[arm].update(result)
        save(out / arm, meta, arrays, result)
        rows.append({
            "Model": arm, "MAE ($)": result["point_scores"]["mae_usd"],
            "Selection validation MAE ($)": best["validation_mae_usd"],
            "90% coverage": result["test_coverage"], "Test cars": len(test),
        })
    report = {
        "comparison": rows,
        "paired_improvement": paired_bootstrap(test, predictions["electra_hierarchy"], predictions["baseline"]),
        "details": details, "encoder": manifest,
        "frozen_xgboost_input_sha256": file_sha256(source),
        "architecture": "Identical frozen facts + MiniLM-price XGBoost AFT predictions in both arms. Frozen ELECTRA features enter ONLY the Bayesian residual mean after calibration-only UMAP and scaling.",
        "umap": {"metric": "cosine", "n_components": 32, "n_neighbors": 30,
                 "min_dist": .1, "n_epochs": 200, "random_state": 42,
                 "fit_scope": "calibration descriptions only; validation/test transformed only",
                 "scaling": "calibration-only StandardScaler",
                 "coefficient_prior": "Normal(0, 0.2^2)"},
        "historical_references": {
            "bge_in_hierarchy": json.loads(Path("reports/hierarchy-umap-experiment/comparison.json").read_text()),
            "bge_in_xgboost": json.loads(Path("reports/bge-umap-experiment/comparison.json").read_text()),
        },
        "limits": "Research only. ELECTRA-Small is a token-discrimination encoder, not a sentence-similarity-trained model; masked mean pooling is an experimental feature choice. Historical test examined in earlier experiments; fixed earlier Lemon cleaning. Likelihood selected on selection-validation cars; intervals calibrated on separate validation cars. No current-market or serving-readiness claim.",
    }
    dest = Path("reports/electra-hierarchy-experiment")
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "comparison.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame({
        "row_id": test.row_id, "vehicle_group": test.vehicle_group,
        "observed": test.observed_price, **predictions,
    }).to_csv(out / "test-predictions.csv.gz", index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
