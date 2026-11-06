"""Fail-closed original wholesale validation leaderboard for the delayed 25 Bid trials."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.bid25_joint_search import N_TRIALS,trial_config
from scripts.cvar95_historical import risk
from scripts.neural_bid_prepare import sha_ids


def collect(folder):
    results={}
    issues=[]
    expected_sha=None;expected_cohort=None
    for file in sorted(Path(folder).rglob("result.json")):
        try:
            rec=json.loads(file.read_text())
            i=rec["trial"]
            if not isinstance(i,int) or not 0<=i<N_TRIALS or i in results:
                raise ValueError("Unknown or duplicated Bid trial")
            if (rec.get("status")!="complete" or rec.get("market")!="bid"
                or rec.get("test_evaluated") is not False
                or rec.get("original_test_and_production_weights_untouched") is not True):
                raise ValueError("Incomplete training or original TEST leaked")
            cfg=trial_config(i)
            if rec.get("config")!=cfg or rec.get("validation_blend_weight_fixed_before_training")!=cfg["neural_weight"]:
                raise ValueError("Unregistered or altered trial configuration")
            prov=rec["provenance"]
            if (prov.get("original_full_train")!=384198
                or prov.get("validation_rows")!=82380
                or prov.get("original_test_rows")!=83603
                or prov.get("original_test_used") is not False):
                raise ValueError("Original full auction split differs")
            if not file.with_name("weights.pt").is_file():
                raise ValueError("Missing original-native PyTorch checkpoint")
            frame=pd.read_csv(file.with_name("validation.csv.gz"),dtype={"row_id":str})
            required={"row_id","split","observed_price","predicted_price",
                      "neural_price","lemon_xgb_price"}
            if (len(frame)!=82380 or not required.issubset(frame.columns)
                or frame.row_id.isna().any() or frame.row_id.duplicated().any()
                or not frame.split.eq("validation").all()):
                raise ValueError("Noncomparable original wholesale heldout vehicles")
            sha=sha_ids(frame.row_id.astype(str).to_numpy())
            if sha!=prov["holdout_row_sha256"]["validation"]:
                raise ValueError("Original VIN-disjoint holdout SHA changed")
            if expected_sha is None:
                expected_sha=sha
                expected_cohort=prov["dataset_sha256"]
            if sha!=expected_sha or prov["dataset_sha256"]!=expected_cohort:
                raise ValueError("Candidate trained on a different cohort")
            actual=frame.observed_price.to_numpy(float)
            price=frame.predicted_price.to_numpy(float)
            n=frame.neural_price.to_numpy(float)
            lemon=frame.lemon_xgb_price.to_numpy(float)
            if not np.allclose(price,cfg["neural_weight"]*n+
                               (1-cfg["neural_weight"])*lemon,rtol=1e-6,atol=.05):
                raise ValueError("Neural/XGBoost validation blend is inconsistent")
            score=risk(actual,price);neural_score=risk(actual,n)
            for key in ("CVaR95_absolute_USD","RMSE_USD","MAE_USD"):
                if (abs(score[key]-rec["validation"][key])>.05 or
                    abs(neural_score[key]-rec["neural_only_validation"][key])>.05):
                    raise ValueError("Validation score differs from per-vehicle predictions")
            results[i]={"trial":i,"architecture":cfg["family"],"config":cfg,
                        "validation":score,"neural_only_validation":neural_score,
                        "best_epoch":rec["best_epoch"],
                        "elapsed_seconds":rec["elapsed_seconds"]}
        except (OSError,ValueError,KeyError,TypeError) as exc:
            issues.append({"file":str(file),"problem":str(exc)})
    missing=sorted(set(range(N_TRIALS))-set(results))
    ordered=sorted(results.values(),key=lambda x:(
        x["validation"]["CVaR95_absolute_USD"],x["validation"]["RMSE_USD"],x["trial"]))
    return {"status":"complete" if len(results)==25 and not issues and not missing
             else "partial_or_invalid",
            "planned_trials":25,"successful_audited_trials":len(results),
            "missing":missing,"invalid":issues,"original_test_used":False,
            "CVaR95_primary":"original 2015-normalized wholesale validation, absolute USD top 5%",
            "secondary":["RMSE_USD","MAE_USD"],
            "not_2026_validated":True,"automatic_deployment":False,
            "baseline_refit":results.get(0),
            "best":ordered[0] if ordered else None,
            "ranked":ordered}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    doc=collect(a.artifacts)
    path=Path(a.output);path.mkdir(parents=True,exist_ok=True)
    (path/"results.json").write_text(json.dumps(doc,indent=2,allow_nan=False))
    lines=["# Delayed 25-trial Bid neural architecture / hyperparameter search","",
        "**Status:** "+doc["status"],
        f"**Successful, audited trials:** {doc['successful_audited_trials']}/25",
        "**Primary ranking:** absolute-dollar CVaR95 among worst 5% of original wholesale validation transactions.",
        "All 25 use exact original VIN-disjoint auction validation identities; Lemon XGBoost is frozen.",
        "No original test scoring, no automatic change to the deployed Bid model, and no 2026 accuracy claims.",
        "",
        "| Rank | Trial | Structure | Neural % | CVaR95 ($) | RMSE ($) | MAE ($) |",
        "|---:|---:|---|---:|---:|---:|---:|"]
    for place,row in enumerate(doc["ranked"],1):
        v=row["validation"]
        lines.append(f"| {place} | {row['trial']} | {row['architecture']} | "
                     f"{row['config']['neural_weight']*100:.0f} | "
                     f"{v['CVaR95_absolute_USD']:,.0f} | "
                     f"{v['RMSE_USD']:,.0f} | {v['MAE_USD']:,.0f} |")
    if doc["baseline_refit"] and doc["best"]:
        a=doc["baseline_refit"]["validation"]["CVaR95_absolute_USD"]
        b=doc["best"]["validation"]["CVaR95_absolute_USD"]
        lines.extend(["","## Comparison to freshly trained original architecture",
            f"Validation CVaR95 improved by ${a-b:,.0f} "
            f"({100*(a-b)/a:+.2f}%) relative to trial 0.",
            "Trial 0 replicates the deployed DeepFM16 / Lemon weighting and architecture, "
            "not the original learned weights; this is not a live market comparison."])
    if doc["missing"]:lines.extend(["","Missing candidate IDs: "+str(doc["missing"])])
    if doc["invalid"]:lines.extend(["","Invalid receipts: "+str(doc["invalid"])])
    lines.extend(["","## Promotion gate","An independent audit must choose the winning candidate "
        "on validation CVaR95, evaluate the untouched original test exactly once, "
        "verify native preprocessing/serving parity and CPU-memory limits, and "
        "separately evaluate current-date extrapolation before production deployment."])
    (path/"README.md").write_text("\n".join(lines)+"\n")
    print("BID25_VALIDATION_REPORT",doc["status"],doc["successful_audited_trials"],flush=True)
    if doc["status"]!="complete":raise SystemExit(2)


if __name__=="__main__":main()
