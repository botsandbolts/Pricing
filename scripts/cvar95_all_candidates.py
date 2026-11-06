"""Original holdout, evaluation-only CVaR95 sweep: every predeclared bid
and ask contender. Use CVaR95 on VALIDATION to select before viewing TEST.

Bid 65 = 26 FM-family + 10 FM/RFF + 29 FT.
Ask 44 = 8 direct embedding-only + 36 DeepCross+AFT scalar/SVD.

Training objectives and production models do not change. No test-driven
reselection: test is evaluated ONLY on validation-chosen winner(s).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.cvar95_historical import risk

BID_FAMILIES={"fm26":26,"fm10":10,"ft29":29}
ASK_FAMILIES={"bge8":8,"deepcross36":36}
WEIGHTS=(0.,.1,.2,.35,.5,.75,1.)


def compare(data,baseline,split):
    required={"row_id","split","observed_price","predicted_price"}
    if not required.issubset(data.columns) or data.row_id.duplicated().any():
        raise ValueError("Incomplete/duplicated candidate vehicles")
    if set(data.row_id)!=set(baseline.index):
        raise ValueError("Different validation/listing VIN holdout identities")
    d=data.set_index("row_id").reindex(baseline.index)
    if (not d.split.eq(split).all() or
        not np.allclose(d.observed_price,baseline.observed_price,rtol=1e-6,atol=.05) or
        not np.isfinite(d.predicted_price.to_numpy(float)).all() or
        not (d.predicted_price.to_numpy(float)>0).all()):
        raise ValueError("Target/split or price mismatch")
    return d


def _ref(path,split,n):
    x=pd.read_csv(path,dtype={"row_id":str})
    if not {"row_id","split","observed_price","predicted_price"}.issubset(x.columns):
        raise ValueError("Missing XGBoost original row-paired archived baseline")
    if x.row_id.duplicated().any():raise ValueError("Duplicate XGBoost baseline vehicle")
    x=x.loc[x.split.eq(split)].set_index("row_id")
    if len(x)!=n or not np.isfinite(x.observed_price).all():
        raise ValueError("Original baseline count changed")
    return x


def _list(root,prefix,expected):
    files=sorted(Path(root).rglob("result.json"))
    if len(files)!=expected:
        raise ValueError("Expected "+str(expected)+" audited "+prefix+" artifacts, got "+str(len(files)))
    return files


def bid(args):
    from scripts.bid_rmse_reselect import specs,validate_manifest,checkpoint_predictions
    from scripts.bid_fm_kernel_train import source
    from scripts.neural_bid_prepare import sha_ids
    z,meta=source(args.features)
    val=_ref(args.baseline,"validation",82380)
    test=_ref(args.baseline,"test",83603)
    for split,frame in (("validation",val),("test",test)):
        if sha_ids(frame.index)!=meta["holdout_row_sha256"][split]:
            raise ValueError("Original VIN-disjoint holdout SHA changed")
    y=val.observed_price.to_numpy(float)
    px=val.predicted_price.to_numpy(float)
    entries=[];mixes=[];paths={};seen=set()
    for family,expected in BID_FAMILIES.items():
        for file in _list(Path(args.artifacts)/family,family,expected):
            doc=json.loads(file.read_text())
            name=doc.get("trial")
            if name not in specs(family) or (family,name) in seen:
                raise ValueError("Duplicate/unregistered bid model: "+str((family,name)))
            seen.add((family,name))
            validate_manifest(family,name,doc,meta,z)
            aligned=compare(pd.read_csv(file.with_name("validation.csv.gz"),
                                        dtype={"row_id":str}),val,"validation")
            p=aligned.predicted_price.to_numpy(float)
            m=risk(y,p)
            if abs(m["MAE_USD"]-doc["validation"]["mae_usd"])>.05:
                raise ValueError("Archive validation score differs")
            rank,label,cohort=specs(family)[name]
            entries.append({"study":family,"model":name,"family":rank,
                            "rank_or_components":label,"cohort":cohort,
                            "selection_split":"validation","selection_n":len(y),
                            **m})
            paths[(family,name)]=file.parent
            for weight in WEIGHTS:
                mixed=(1-weight)*px+weight*p
                scored=risk(y,mixed)
                mixes.append({"study":family,"model":name,
                              "fm_neural_weight":weight,
                              "xgb_lemon_weight":1-weight,**scored})
        if len([v for v in seen if v[0]==family])!=expected:
            raise ValueError("Missing bid model in family "+family)
    entries=sorted(entries,key=lambda x:(x["CVaR95_absolute_USD"],x["study"],x["model"]))
    mixes=sorted(mixes,key=lambda x:(x["CVaR95_absolute_USD"],x["study"],x["model"],x["fm_neural_weight"]))
    champion=entries[0]
    blend=mixes[0]
    selected=tuple(dict.fromkeys(((champion["study"],champion["model"]),
                                  (blend["study"],blend["model"]))))
    truth=test.observed_price.to_numpy(float)
    baseline=test.predicted_price.to_numpy(float)
    tested={}
    for family,name in selected:
        frame=checkpoint_predictions(family,name,paths[(family,name)],z,meta)
        if set(frame.index)!=set(test.index):
            raise ValueError("CVaR winner test vehicle identities differ")
        p=frame.reindex(test.index)
        if not np.allclose(p.observed_price,truth,rtol=1e-6,atol=.05):
            raise ValueError("CVaR selected test labels differ")
        tested[(family,name)]=p.predicted_price.to_numpy(float)
    standalone=risk(truth,tested[(champion["study"],champion["model"])])
    alpha=blend["fm_neural_weight"]
    bp=(1-alpha)*baseline+alpha*tested[(blend["study"],blend["model"])]
    blendtest=risk(truth,bp)
    result={
        "market":"bid","status":"complete",
        "primary_metric":"validation_absolute_dollar_CVaR95",
        "validation_models":len(entries),"validation_blends":len(mixes),
        "test_models_scored":len(selected),"original_test_rows":83603,
        "selection_policy":"CVaR95 chosen exclusively on validation; test used once for only previously selected checkpoints",
        "training_objectives_unchanged":True,
        "models_chosen_by_test":False,
        "chosen_standalone":{"study":champion["study"],"model":champion["model"],
                             "validation":champion,"original_test":standalone},
        "chosen_blend":{"study":blend["study"],"model":blend["model"],
                        "neural_weight":alpha,"xgb_lemon_weight":1-alpha,
                        "validation":blend,"original_test":blendtest},
        "original_xgboost_lemon_reference":{"validation":risk(y,px),
                                             "original_test":risk(truth,baseline)},
        "full_data_xgboost_CVaR95":"not verified from per-vehicle original test predictions",
        "validation_ranked_models":entries,
        "validation_ranked_blends":mixes}
    return result,{"standalone":tested[(champion["study"],champion["model"])],
                   "blend":bp,"truth":truth,"xgb_lemon":baseline}


def ask(args):
    # Read immutable predeclared names without importing the Torch trainer:
    # this is a metrics-only job, not a new model-fitting environment.
    import ast
    tree=ast.parse((Path(args.root)/"scripts/deepcross_aft_tune.py").read_text())
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("CONFIGS","FEATURES"):
                    constants[target.id]=ast.literal_eval(node.value)
    if set(constants)!={"CONFIGS","FEATURES"}:
        raise ValueError("Cannot audit original 36-arm DeepCross registry")
    CONFIGS,FEATURES=constants["CONFIGS"],constants["FEATURES"]
    from scripts.raw_embedding_ask_encode import SPECS
    from scripts.raw_embedding_ask_report import audited_validation
    val=_ref(args.baseline,"validation",31668)
    test=_ref(args.baseline,"test",31682)
    old_candidates=_list(Path(args.artifacts)/"deepcross36","DeepCross",36)
    raw_candidates=_list(Path(args.artifacts)/"bge8","BGE",8)
    y=val.observed_price.to_numpy(float)
    rows=[];paths={}
    seen_old=set();seen_raw=set()
    for file in old_candidates:
        doc=json.loads(file.read_text())
        key=(doc.get("feature_variant"),doc.get("variant"))
        if key in seen_old or key not in {(f,c) for f in FEATURES for c in CONFIGS}:
            raise ValueError("Duplicate/nonregistered DeepCross candidate")
        seen_old.add(key)
        if doc.get("status")!="completed" or doc.get("train_fit")!=133000:
            raise ValueError("DeepCross training original fit changed")
        all_rows=pd.read_csv(file.with_name("predictions.csv.gz"),dtype={"row_id":str})
        p=compare(all_rows.loc[all_rows.split.eq("validation")],val,"validation").predicted_price.to_numpy(float)
        scored=risk(y,p)
        if abs(scored["MAE_USD"]-doc["original_validation"]["mae_usd"])>.05:
            raise ValueError("DeepCross validation score is untrusted")
        name=key[0]+"-"+key[1]
        rows.append({"study":"deepcross36","model":name,"architecture":doc["architecture"],
                     "fit":"original fixed training",**scored})
        paths[("deepcross36",name)]=file.parent
    for file in raw_candidates:
        doc=json.loads(file.read_text())
        name=doc.get("candidate")
        if name in seen_raw or name not in SPECS or doc.get("test_predictions_created") is not False:
            raise ValueError("Duplicate/malformed BGE candidate")
        seen_raw.add(name)
        if (doc.get("status"),doc.get("fitting_rows"),doc.get("validation_rows"),
            doc.get("text_price_heads"))!=("complete",133000,31668,0):
            raise ValueError("Wrong BGE frozen training/feature contract")
        p=compare(pd.read_csv(file.with_name("validation.csv.gz"),dtype={"row_id":str}),
                  val,"validation").predicted_price.to_numpy(float)
        scored=risk(y,p)
        if abs(scored["MAE_USD"]-doc["validation"]["mae_usd"])>.05:
            raise ValueError("BGE validation data do not match cached metrics")
        rows.append({"study":"bge8","model":name,
                     "architecture":"embedding-only complete DeepCross/AFT",
                     "fit":"original fixed training",**scored})
        paths[("bge8",name)]=file.parent
    if len(seen_old)!=36 or len(seen_raw)!=8:
        raise ValueError("Incomplete original 44 asking model families")
    rows=sorted(rows,key=lambda q:(q["CVaR95_absolute_USD"],q["study"],q["model"]))
    winner=rows[0]
    key=(winner["study"],winner["model"])
    chosen_test=None
    if key[0]=="deepcross36":
        source=pd.read_csv(paths[key]/"predictions.csv.gz",dtype={"row_id":str})
        aligned=compare(source.loc[source.split.eq("test")],test,"test")
        chosen_test=risk(test.observed_price.to_numpy(float),
                         aligned.predicted_price.to_numpy(float))
    elif key==("bge8","bge-256"):
        # Original report tested ONLY bge-256; other 7 are validation-only.
        source=pd.read_csv(Path(args.root)/"reports/raw-embedding-ask/winner-original-test.csv.gz",
                           dtype={"row_id":str})
        aligned=compare(source,test,"test")
        chosen_test=risk(test.observed_price.to_numpy(float),
                         aligned.predicted_price.to_numpy(float))
    # If a different BGE encoder wins CVaR, defer fresh TEST scoring until
    # a new original-embedding inference job verifies its saved checkpoint.
    result={"market":"ask","status":"complete",
        "primary_metric":"validation_absolute_dollar_CVaR95",
        "validation_models":len(rows),"original_test_rows":31682,
        "training_objectives_unchanged":True,"models_chosen_by_test":False,
        "CVaR95_selection":{"study":winner["study"],"model":winner["model"],
                            "validation":winner,
                            "original_test":chosen_test,
                            "test_status":"scored_after_validation_selection" if chosen_test
                                      else "winner_checkpoint_test_inference_required"},
        "xgboost_lemon_original_reference":{
            "validation":risk(val.observed_price.to_numpy(float),val.predicted_price.to_numpy(float)),
            "test":risk(test.observed_price.to_numpy(float),test.predicted_price.to_numpy(float))},
        "all_original_validation_models_ranked":rows,
        "caveat":"If a BGE encoder other than bge-256 wins CVaR validation, no test score is claimed without independent neural inference."}
    return result,None


def execute(args):
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    report,pred=(bid(args) if args.market=="bid" else ask(args))
    (out/"comparison.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    candidates=report["validation_ranked_models"] if args.market=="bid" else report["all_original_validation_models_ranked"]
    pd.DataFrame(candidates).to_csv(out/"all-models-original-validation-cvar95.csv",index=False)
    if args.market=="bid":
        pd.DataFrame(report["validation_ranked_blends"]).to_csv(
            out/"all-blends-original-validation-cvar95.csv",index=False)
        pd.DataFrame(pred).to_csv(out/"CVaR95-preselected-original-test.csv.gz",index=False)
    print("CVaR95_FULL_FAMILY_AUDIT",json.dumps({
        "market":args.market,
        "n_candidates":len(candidates),
        "winner":report.get("chosen_standalone") or report.get("CVaR95_selection"),
        "blend":report.get("chosen_blend")}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--market",choices=("bid","ask"),required=True)
    p.add_argument("--artifacts",default="training-cache/cvar95-arms")
    p.add_argument("--root",default=".")
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--baseline",required=True)
    p.add_argument("--output",required=True)
    args=p.parse_args();execute(args)
