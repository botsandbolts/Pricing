"""CVaR95 audit for historical models selected before CVaR95 became primary.

The previous models are NOT reselected by test. This evaluates exactly those
checkpoint predictions already assessed in the repository. No training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def risk(actual, predicted):
    y=np.asarray(actual,dtype=float)
    p=np.asarray(predicted,dtype=float)
    if y.ndim!=1 or y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or (y<=0).any() or (p<=0).any():
        raise ValueError("Invalid original absolute dollar errors")
    error=np.abs(p-y)
    k=int(np.ceil(len(error)*.05))
    worst=np.partition(error,len(error)-k)[-k:]
    return {"rows":len(error),"worst_five_pct_rows":k,
            "CVaR95_absolute_USD":float(worst.mean()),
            "VaR95_absolute_USD":float(np.percentile(error,95)),
            "MAE_USD":float(error.mean()),
            "RMSE_USD":float(np.sqrt(np.mean(np.square(p-y)))),
            "worst_one_pct_USD":float(
                np.partition(error,len(error)-int(np.ceil(len(error)*.01)))[-int(np.ceil(len(error)*.01)):].mean())}


def frame(path):
    f=pd.read_csv(path,dtype={"row_id":str})
    if "row_id" not in f or f.row_id.duplicated().any() or f.row_id.isna().any():
        raise ValueError("Missing/duplicated original heldout row IDs: "+str(path))
    return f


def consistent(a,b):
    left=a.set_index("row_id")
    right=b.set_index("row_id")
    if set(left.index)!=set(right.index):
        raise ValueError("Comparators do not share exactly the same heldout vehicles")
    right=right.reindex(left.index)
    if not np.allclose(left.observed_price.to_numpy(float),
                       right.observed_price.to_numpy(float),atol=.05,rtol=1e-6):
        raise ValueError("Comparator used different original observed prices")
    return left,right


def wholesale(root):
    root=Path(root)
    baseline=None
    rows=[]
    for study in ("fm26","fm10","ft29"):
        path=root/"reports/bid-rmse"/study/"original-test-selected-predictions.csv.gz"
        f=frame(path)
        if len(f)!=83603:raise ValueError("Wrong 2015 original test rows: "+study)
        if baseline is None:baseline=f
        base,checked=consistent(baseline,f)
        if not np.allclose(base.xgboost_lemon_price.to_numpy(float),
                           checked.xgboost_lemon_price.to_numpy(float),atol=.05,rtol=1e-6):
            raise ValueError("Historical Lemon XGBoost prices disagree")
        for label,col in (("standalone","rmse_selected_standalone_prediction"),
                          ("validated_blend","rmse_selected_blend_prediction")):
            v=risk(checked.observed_price,checked[col])
            rows.append({"model":study+" / "+label,
                         "previous_selection_metric":"validation USD RMSE",**v})
    rows.append({"model":"Lemon XGBoost historical baseline",
                 "previous_selection_metric":"original baseline",
                 **risk(baseline.observed_price,baseline.xgboost_lemon_price)})
    return rows


def asking(root,old_dir):
    b=frame(Path(root)/"reports/raw-embedding-ask/winner-original-test.csv.gz")
    if len(b)!=31682 or not b.split.eq("test").all():
        raise ValueError("Wrong original 2021 chronological ask test")
    out=[{"model":"BGE-256 + full DeepCross/AFT",
          "previous_selection_metric":"validation MAE across eight encoders",
          **risk(b.observed_price,b.predicted_price)}]
    for label,filename in (
        ("MiniLM scalar + DeepCross/AFT, 12% dropout","deepcross-aft-tune-minilm-dropout12"),
        ("MiniLM scalar + DeepCross/AFT, 4-expert MoE","deepcross-aft-tune-minilm-moe4")):
        original=frame(Path(old_dir)/filename/"predictions.csv.gz")
        original=original.loc[original.split.eq("test")]
        if len(original)!=31682:raise ValueError("Wrong original older ask test cohort")
        past,current=consistent(original,b)
        out.append({"model":label,"previous_selection_metric":"earlier research validation MAE",
                    **risk(past.observed_price,past.predicted_price)})
    return out


def run(root,old_dir,out):
    bid=wholesale(root)
    ask=asking(root,old_dir)
    report={
      "status":"complete","metric":"CVaR95 absolute dollar error",
      "definition":"Mean of the largest ceil(0.05*N) absolute prediction errors, original-dollar units; lower is better",
      "original_holdout_sizes":{"wholesale":83603,"asking":31682},
      "bid":sorted(bid,key=lambda r:r["CVaR95_absolute_USD"]),
      "ask":sorted(ask,key=lambda r:r["CVaR95_absolute_USD"]),
      "models_were_selected_before_cvar95_became_primary":True,
      "test_used_for_reselection":False,"no_retraining":True,
      "production_models_unchanged":True,
      "full_data_XGBoost_original_test_dollar_CVaR_unavailable":True,
      "caveat":"This identifies CVaR95 among previously validation-selected models only. Choosing the new CVaR-optimal model requires separate original-validation CVaR95 ranking before scoring that choice on test.",
      "theoretical_reference":{"paper":"Optimization of Conditional Value-at-Risk",
         "authors":"Rockafellar and Uryasev (2000)","doi":"https://doi.org/10.21314/JOR.2000.038"}
    }
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    pd.DataFrame(report["bid"]).to_csv(out/"wholesale.csv",index=False)
    pd.DataFrame(report["ask"]).to_csv(out/"asking.csv",index=False)
    lines=["# Original untouched historical CVaR95 audit","",
      "The new metric is the mean absolute dollar error of the worst 5% of",
      "original test predictions. CVaR95 is evaluation ONLY: weights unchanged.",
      "All candidates here were selected using historical validation RMSE/MAE,",
      "not test-based CVaR optimization. Model replacement requires a NEW",
      "validation CVaR selection protocol; this table is descriptive.","",
      "## Wholesale","",
      "| Previously selected model | CVaR95 USD | VaR95 USD | MAE USD | RMSE USD |",
      "|---|---:|---:|---:|---:|"]
    for q in report["bid"]:
        lines.append("| "+q["model"]+" | "+("USD {:,.2f} | "*4).format(
            q["CVaR95_absolute_USD"],q["VaR95_absolute_USD"],q["MAE_USD"],q["RMSE_USD"]))
    lines.extend(["","## Asking","",
      "| Previously selected model | CVaR95 USD | VaR95 USD | MAE USD | RMSE USD |",
      "|---|---:|---:|---:|---:|"])
    for q in report["ask"]:
        lines.append("| "+q["model"]+" | "+("USD {:,.2f} | "*4).format(
            q["CVaR95_absolute_USD"],q["VaR95_absolute_USD"],q["MAE_USD"],q["RMSE_USD"]))
    (out/"README.md").write_text("\n".join(lines)+"\n")
    print("EXACT_CVAR95_HISTORICAL",json.dumps({
        "bid_best_previously_selected":report["bid"][0],
        "ask_best_previously_selected":report["ask"][0],
        "bid_xgb_baseline":report["bid"][-1] if report["bid"][-1]["model"].startswith("Lemon") else next(x for x in report["bid"] if x["model"].startswith("Lemon"))
    }),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=".")
    p.add_argument("--old-ask",default="training-cache/old-ask")
    p.add_argument("--out",default="reports/cvar95/previously-selected")
    a=p.parse_args();run(a.root,a.old_ask,a.out)
