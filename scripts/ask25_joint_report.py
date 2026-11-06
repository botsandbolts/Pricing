"""Audited final validation-only leaderboard for 25 Ask BGE research candidates."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.ask25_joint_search import TRIALS,trial_config
from scripts.cvar95_historical import risk
from scripts.raw_embedding_ask_encode import id_hash


def collect(artifacts):
    checked={}
    issues=[]
    expected_hash=None
    expected_cohort=None
    for result in sorted(Path(artifacts).rglob("result.json")):
        try:
            d=json.loads(result.read_text())
            idx=d["trial"]
            if not isinstance(idx,int) or not 0<=idx<TRIALS:
                raise ValueError("Unregistered trial")
            if idx in checked:
                raise ValueError("Duplicate trial "+str(idx))
            if d["status"]!="completed" or d.get("original_test_scored") is not False:
                raise ValueError("Trial incomplete or original test accessed")
            if d.get("joint_structure_and_hyperparameters")!=trial_config(idx):
                raise ValueError("Unregistered structure / hyperparameter configuration")
            m=d["provenance"]
            if (m["training_rows"],m["train_only_early_stopping_rows"],
                m["validation_rows"],m["original_test_rows"])!=(133000,14777,31668,31682):
                raise ValueError("Original split proportions changed")
            if m.get("test_used") is not False:
                raise ValueError("Original test was accessed")
            if not result.with_name("weights.pt").is_file():
                raise ValueError("Missing native PyTorch weight checkpoint")
            x=pd.read_csv(result.with_name("validation.csv.gz"),dtype={"row_id":str})
            if (len(x)!=31668 or x.row_id.isna().any()
                or x.row_id.duplicated().any()
                or not x.split.eq("validation").all()):
                raise ValueError("Validation identities or split invalid")
            sha=id_hash(x.row_id.astype(str).to_numpy())
            if sha!=m["holdout_row_sha256"]["validation"]:
                raise ValueError("Original validation row fingerprint mismatch")
            if expected_hash is None:
                expected_hash=sha
                expected_cohort=m["cohort_sha256"]
            if sha!=expected_hash or m["cohort_sha256"]!=expected_cohort:
                raise ValueError("Trials use different original cohorts")
            metrics=risk(x.observed_price.to_numpy(float),
                         x.predicted_price.to_numpy(float))
            for key in ("CVaR95_absolute_USD","RMSE_USD","MAE_USD"):
                if (not np.isfinite(float(d["original_validation"][key]))
                    or abs(metrics[key]-d["original_validation"][key])>.05):
                    raise ValueError("Reported metric mismatches actual per-vehicle predictions: "+key)
            checked[idx]={
                "trial":idx,
                "architecture":d["architecture"],
                "configuration":d["joint_structure_and_hyperparameters"],
                "validation":metrics,
                "training": {
                    "best_epoch":d["best_epoch"],
                    "epochs_completed":d["epochs_completed"],
                    "time_budget_limited":d["time_budget_limited"],
                    "elapsed_seconds":d["elapsed_seconds"],
                }}
        except (OSError,TypeError,KeyError,ValueError) as exc:
            issues.append({"path":str(result),"problem":str(exc)})
    missing=sorted(set(range(TRIALS))-set(checked))
    ordered=sorted(checked.values(),key=lambda z:
                   (z["validation"]["CVaR95_absolute_USD"],
                    z["validation"]["RMSE_USD"],z["trial"]))
    return {
        "status":"complete" if len(checked)==TRIALS and not missing and not issues
                 else "partial_or_invalid",
        "planned_trials":TRIALS,
        "successful_audited_trials":len(checked),
        "failed_or_invalid_artifacts":issues,
        "missing_trials":missing,
        "primary_metric":"Original 2021 asking-price validation absolute-dollar CVaR95 (worst 5%)",
        "secondary_metrics":["RMSE_USD","MAE_USD"],
        "original_test_accessed":False,
        "current_november_2026_accuracy_verified":False,
        "no_production_promotion":True,
        "baseline":"trial 0: same deployed BGE+DeepCross/AFT architecture re-fitted",
        "baseline_trial0":checked.get(0),
        "best_trial":ordered[0] if ordered else None,
        "ranked_validation":ordered,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    report=collect(a.artifacts)
    target=Path(a.output)
    target.mkdir(parents=True,exist_ok=True)
    (target/"results.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    md=["# Ask · 25 combined architecture and hyperparameter experiments","",
        "**Status:** "+report["status"],
        "**Audited successful trials:** "+str(report["successful_audited_trials"])+"/25",
        "**Selection:** original-validation dollar CVaR95 (worst 5%).",
        "RMSE and MAE are secondary. Original test is untouched.",
        "No 2026-sale accuracy established. No production model changes.","",
        "| Rank | Trial | Architecture | CVaR95 (USD) | RMSE (USD) | MAE (USD) |",
        "|---:|---:|---|---:|---:|---:|"]
    for rank,item in enumerate(report["ranked_validation"],1):
        m=item["validation"]
        md.append(f"| {rank} | {item['trial']} | {item['architecture']} | "
                  f"{m['CVaR95_absolute_USD']:,.0f} | "
                  f"{m['RMSE_USD']:,.0f} | {m['MAE_USD']:,.0f} |")
    if report["baseline_trial0"] and report["best_trial"]:
        best=report["best_trial"]["validation"]["CVaR95_absolute_USD"]
        baseline=report["baseline_trial0"]["validation"]["CVaR95_absolute_USD"]
        md.extend(["","## Comparison against re-fitted original architecture",
                   f"Validation CVaR95 change: ${baseline-best:,.0f} "
                   f"({(baseline-best)/baseline*100:+.2f}% lower). "
                   "This is a same-study architecture comparison, NOT a production-model "
                   "or 2026 market accuracy guarantee."])
    if report["missing_trials"]:
        md.extend(["","## Missing trial IDs",str(report["missing_trials"])])
    if report["failed_or_invalid_artifacts"]:
        md.extend(["","## Artifact contract failures",str(report["failed_or_invalid_artifacts"])])
    md.extend(["","## Required final gate",
        "Choose a single candidate using original validation CVaR95. "
        "Assess its native weights on the untouched original test once, "
        "check inference parity, cold-start CPU/RAM and current-date scenario assumptions "
        "before considering deployment."])
    (target/"README.md").write_text("\n".join(md)+"\n")
    print(json.dumps({"status":report["status"],
                      "successful":report["successful_audited_trials"],
                      "missing":report["missing_trials"]}),flush=True)
    if report["status"]!="complete":
        raise SystemExit(2)


if __name__=="__main__":
    main()
