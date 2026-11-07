"""Install ONLY validation-selected native CVaR95 challengers after original-test audit.

New bundles are isolated from the original, parity-certified two-model release.
No ONNX, joblib, pickle conversion, training in serving, or alteration of the
main pricing calculator. FAIL CLOSED on missing/mismatched research artifacts.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from scripts.cvar95_historical import risk
from scripts.raw_embedding_ask_encode import id_hash
from pricing.neural_champion_live import (
    ASK_FOLDER,BID_FOLDER,CVaR95Asking,CVaR95Wholesale,
)

ASK_TRIAL=15
BID_TRIAL=18


def sha(path):
    with open(path,"rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def compare(original,val,expected_size,identity,split="validation"):
    x=pd.read_csv(val,dtype={"row_id":str})
    expected=original["ids"][original["phase"].astype(str)==split].astype(str)
    if (len(x)!=expected_size or x.row_id.duplicated().any()
        or not x.split.eq(split).all() or set(x.row_id)!=set(expected)
        or id_hash(x.row_id.astype(str).to_numpy())!=identity):
        raise ValueError("The selected research artifact has mismatched original holdout rows")
    aligned=x.set_index("row_id").reindex(expected)
    truth=original["y"][original["phase"].astype(str)==split].astype(float)
    if not np.allclose(truth,aligned.observed_price.to_numpy(float),atol=.05,rtol=1e-6):
        raise ValueError("Research original validation labels differ")
    return aligned,truth


def _record(manifest,verification,out):
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    (out/"verification.json").write_text(json.dumps(verification,indent=2,allow_nan=False))
    for f in out.iterdir():
        if f.suffix in (".onnx",".pkl",".joblib",".npz",".npy",".csv",".parquet"):
            raise ValueError("Converted or original-row data included in release")
        if f.stat().st_size>95000000:
            raise ValueError("Individual release artifact exceeds GitHub size limit")


def ask(args):
    from scripts.raw_embedding_ask_train import frozen_source
    from scripts.ask25_joint_search import trial_config,model_for
    from scripts.neural_ask_train import predict
    from pricing.neural_ask_embedding_live import available as original_available
    if not original_available(args.root):
        raise ValueError("Original BGE/DeepCross release is not verified")
    report=json.loads(Path(args.report).read_text())
    winner=report.get("best_trial") or {}
    params=trial_config(ASK_TRIAL)
    if (report.get("status"),report.get("successful_audited_trials"),
        winner.get("trial"),winner.get("configuration"),
        report.get("original_test_accessed"))!=(
           "complete",25,ASK_TRIAL,params,False):
        raise ValueError("The completed 25-arm CVaR95 Ask winner differs")
    z,emb,source,emeta=frozen_source(args.features,args.embeddings,"bge-256")
    ckpt=torch.load(Path(args.winner)/"weights.pt",map_location="cpu",weights_only=True)
    if (ckpt.get("trial"),ckpt.get("parameters"))!=(ASK_TRIAL,params):
        raise ValueError("Selected Ask native checkpoint does not match validation trial")
    prov=ckpt.get("provenance",{})
    if (prov.get("cohort_sha256"),prov.get("embedding_sha256"),
        prov.get("holdout_row_sha256"))!=(
        source["cohort_sha256"],emeta["embedding_file_sha256"],
        source["holdout_row_sha256"]):
        raise ValueError("Ask cohort/embedding/checkpoint fingerprint mismatch")
    net=model_for(params,source["cardinalities"],384)
    net.load_state_dict(ckpt["state_dict"],strict=True)
    net.eval()
    ds=TensorDataset(torch.from_numpy(z["num"].astype(np.float32)),
       torch.from_numpy(z["cat"].astype(np.int64)),
       torch.from_numpy(emb.astype(np.float32)),
       torch.from_numpy(np.log(z["y"]).astype(np.float32)))
    phase=z["phase"].astype(str)
    val=np.flatnonzero(phase=="validation")
    test=np.flatnonzero(phase=="test")
    archived,truth=compare(z,Path(args.winner)/"validation.csv.gz",31668,
                           source["holdout_row_sha256"]["validation"])
    vpred=predict(net,ds,val,768)
    if not np.allclose(vpred,archived.predicted_price.to_numpy(float),
                       rtol=.00005,atol=.25):
        raise ValueError("Full original validation Ask predictions fail native checkpoint parity")
    vrisk=risk(truth,vpred)
    for key in ("CVaR95_absolute_USD","MAE_USD","RMSE_USD"):
        if abs(vrisk[key]-winner["validation"][key])>.1:
            raise ValueError("Measured research CVaR95 differs from published Ask selection")
    tpred=predict(net,ds,test,768)
    trisk=risk(z["y"][test].astype(float),tpred)
    baseline=pd.read_csv(args.baseline_test,dtype={"row_id":str})
    if "split" in baseline:baseline=baseline.loc[baseline.split.eq("test")]
    if len(baseline)!=31682 or baseline.row_id.duplicated().any():
        raise ValueError("Missing untouched original Ask baseline test rows")
    baseline=baseline.set_index("row_id").reindex(z["ids"][test].astype(str))
    if not np.allclose(z["y"][test],baseline.observed_price,rtol=1e-6,atol=.05):
        raise ValueError("Cannot audit matched original Ask test comparator")
    old=risk(z["y"][test].astype(float),baseline.predicted_price.to_numpy(float))
    out=Path(args.output or ASK_FOLDER)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(args.winner)/"weights.pt",out/"weights.pt")
    manifest={
        "status":"release_candidate_verified","market":"ask","source_study":"ask25-joint",
        "trial":ASK_TRIAL,"config":params,"checkpoint_sha256":sha(out/"weights.pt"),
        "cohort_sha256":source["cohort_sha256"],
        "embedding_sha256":emeta["embedding_file_sha256"],
        "frozen_encoder_revision":emeta["model_revision"],
        "original_holdout_row_sha256":source["holdout_row_sha256"],
        "historical_validation":vrisk,"historical_test":trisk,
        "historical_incumbent_test":old,
        "original_test_scored_after_validation_selection":True,
        "native_pytorch_only":True,"production_main_calculator_unchanged":True,
        "not_2026_market_accuracy":True,
        "description":"Frozen pinned BGE-small 256-token / two-layer projection / three-block ResMLP + AFT trained with a small train-only dollar-tail penalty"
    }
    # Stage temporary candidate metadata for the packaging-only inference check.
    # It is not a verified public release until the final receipt is written.
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"verification.json").write_text(json.dumps({"status":"pending","verified":False}))
    # Existing BGEDeepCrossAsk already verifies frozen encoder and preprocessing.
    engine=CVaR95Asking(args.root,_parity_test=True)
    population=np.random.default_rng(20261107).choice(val,size=32,replace=False)
    direct=torch.from_numpy(z["num"][population].astype(np.float32))
    cats=torch.from_numpy(z["cat"][population].astype(np.int64))
    vectors=torch.from_numpy(emb[population].astype(np.float32))
    with torch.inference_mode():
        raw=engine.model(direct,cats,vectors)[:,0].numpy()
    serving=np.exp(np.clip(raw,np.log(100),np.log(1e6)))
    if not np.allclose(serving,vpred[np.searchsorted(val,population)],
                       rtol=.0001,atol=.5):
        raise ValueError("Research Ask model and installed native serving weights differ")
    verification={
        "status":"passed","verified":True,
        "complete_original_validation_parity":True,
        "original_test_rows_scored":31682,
        "shared_preprocessor_parity":True,
        "live_prediction_checked":True,
        "weight_sha256":sha(out/"weights.pt"),
        "sample_validation_serving_rows":len(population),
        "training_encoder_reused":True,
        "original_test_used_for_selection":False,
        "historical_test_CVaR95_USD":trisk["CVaR95_absolute_USD"],
        "historical_incumbent_CVaR95_USD":old["CVaR95_absolute_USD"]
    }
    _record(manifest,verification,out)
    print("CVaR95_ASK_NATIVE_RELEASE_VERIFIED",json.dumps(verification),flush=True)


def _bid_baseline(path,z):
    baseline=pd.read_csv(path,dtype={"row_id":str})
    if not {"row_id","split","observed_price","predicted_price"}.issubset(baseline):
        raise ValueError("Original Lemon XGBoost baseline missing row-level fields")
    if baseline.row_id.duplicated().any():raise ValueError("Duplicate original auctions")
    phase=z["phase"].astype(str)
    output={}
    for split,n in (("validation",82380),("test",83603)):
        idx=np.flatnonzero(phase==split)
        ids=z["ids"][idx].astype(str)
        frame=baseline[baseline.split.eq(split)].set_index("row_id").reindex(ids)
        if len(frame)!=n or frame.predicted_price.isna().any():
            raise ValueError("Original Lemon XGBoost missing matched wholesale rows")
        if not np.allclose(frame.observed_price,z["y"][idx],rtol=1e-6,atol=.05):
            raise ValueError("Original wholesale observed prices differ")
        output[split]=frame.predicted_price.to_numpy(float)
    return output


def bid(args):
    from scripts.bid_fm_kernel_train import source,partition
    from scripts.neural_200_sweep import spec
    from scripts.bid_fm_scale_train import ScaledAuctionFM,prediction
    from pricing.neural_bid_blend_live import available as original_available
    if not original_available(args.root):
        raise ValueError("Original wholesale DeepFM/XGBoost release is unverified")
    report=json.loads(Path(args.report).read_text())
    winner=report.get("group_winners",{}).get("bid_hparams",{})
    params=spec("bid_hparams",BID_TRIAL)
    if (report.get("status"),report.get("completed_valid_trials"),
        winner.get("trial"),winner.get("config"))!=(
        "complete",200,BID_TRIAL,params):
        raise ValueError("Not the published CVaR95-selected DeepFM16 wholesale challenger")
    z,source_manifest=source(args.features)
    pack=torch.load(Path(args.winner)/"weights.pt",map_location="cpu",weights_only=True)
    if (pack.get("search_group"),pack.get("trial"),
        pack.get("trial_configuration"))!=("bid_hparams",18,params):
        raise ValueError("Wrong saved neural wholesale checkpoint identity")
    model=ScaledAuctionFM(source_manifest["cardinalities"],"deepfm",16,
        float(pack["state_dict"]["fm.base"]))
    model.load_state_dict(pack["state_dict"],strict=True)
    model.eval()
    fit,inner,val,test=partition(z,"full")
    nums=z["num"].astype(np.float32)
    cats=z["cat"].astype(np.int64)
    y=z["y"].astype(float)
    archived,truth=compare(z,Path(args.winner)/"validation.csv.gz",82380,
                           source_manifest["holdout_row_sha256"]["validation"])
    neural_val=prediction(model,nums,cats,val)
    if not np.allclose(neural_val,archived.predicted_price.to_numpy(float),
                       rtol=.00005,atol=.25):
        raise ValueError("New wholesale native checkpoint does not reproduce ALL validation predictions")
    lemon=_bid_baseline(args.baseline,z)
    records=json.loads((Path(args.winner)/"result.json").read_text())
    alpha=float(records["validation"]["neural_weight"])
    if not .5<=alpha<=1.:
        raise ValueError("The original CVaR95-selected blend coefficient is invalid")
    blended=alpha*neural_val+(1-alpha)*lemon["validation"]
    vrisk=risk(y[val],blended)
    for key in ("CVaR95_absolute_USD","RMSE_USD","MAE_USD"):
        if abs(vrisk[key]-winner[key])>.1:
            raise ValueError("The wholesale selection's ORIGINAL validation CVaR95 cannot be reproduced")
    test_neural=prediction(model,nums,cats,test)
    blend_test=alpha*test_neural+(1-alpha)*lemon["test"]
    trisk=risk(y[test],blend_test)
    incumbent=json.loads((Path(args.root)/
        "models/neural-experimental-bid/manifest.json").read_text())
    old={"CVaR95_absolute_USD":float(incumbent["test_cvar95_absolute_usd"]),
        "RMSE_USD":float(incumbent["test_rmse_usd"]),
        "MAE_USD":float(incumbent["test_mae_usd"])}
    out=Path(args.output or BID_FOLDER)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(args.winner)/"weights.pt",out/"weights.pt")
    manifest={
        "status":"release_candidate_verified","market":"bid","source_study":"bid_hparams",
        "trial":18,"config":params,"neural_weight":alpha,
        "lemon_weight":1-alpha,"checkpoint_sha256":sha(out/"weights.pt"),
        "cohort_sha256":source_manifest["cohort_sha256"],
        "original_holdout_row_sha256":source_manifest["holdout_row_sha256"],
        "historical_validation":vrisk,"historical_test":trisk,
        "historical_incumbent_test":old,
        "original_test_scored_after_validation_selection":True,
        "native_pytorch_only":True,"production_main_calculator_unchanged":True,
        "not_2026_market_accuracy":True,
        "price_basis":"2015-index-normalized USD at historical 2014–15 auctions",
        "description":"DeepFM16 with spline main effects, learned pair interactions and nonlinear residual tower; mixed with original Lemon XGBoost"
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (out/"verification.json").write_text(json.dumps({"status":"pending","verified":False}))
    engine=CVaR95Wholesale(args.root,_parity_test=True)
    sample=np.random.default_rng(20261107).choice(val,size=40,replace=False)
    with torch.inference_mode():
        raw=engine.model(torch.from_numpy(nums[sample]),
            torch.from_numpy(cats[sample]))[:].numpy()
    serving=np.exp(np.clip(raw,np.log(100),np.log(1e6)))
    if not np.allclose(serving,neural_val[np.searchsorted(val,sample)],rtol=.0001,atol=.5):
        raise ValueError("Native DeepFM challenger source/serving parity failed")
    verification={
        "status":"passed","verified":True,
        "complete_original_validation_parity":True,
        "original_test_rows_scored":83603,
        "shared_preprocessor_parity":True,
        "live_prediction_checked":True,
        "weight_sha256":sha(out/"weights.pt"),
        "sample_validation_serving_rows":len(sample),
        "original_lemon_xgboost_reused":True,
        "original_test_used_for_selection":False,
        "historical_test_CVaR95_USD":trisk["CVaR95_absolute_USD"],
        "historical_incumbent_CVaR95_USD":old["CVaR95_absolute_USD"]
    }
    _record(manifest,verification,out)
    print("CVaR95_BID_NATIVE_RELEASE_VERIFIED",json.dumps(verification),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--market",required=True,choices=("ask","bid"))
    parser.add_argument("--root",default=".")
    parser.add_argument("--winner",required=True)
    parser.add_argument("--report",required=True)
    parser.add_argument("--features",required=True)
    parser.add_argument("--embeddings",default="training-cache/bge-256")
    parser.add_argument("--baseline",default="benchmark-reference/bid/predictions.csv.gz")
    parser.add_argument("--baseline-test",default="training-cache/ask-bge-report/winner-original-test.csv.gz")
    parser.add_argument("--output")
    args=parser.parse_args()
    (ask if args.market=="ask" else bid)(args)


if __name__=="__main__":
    main()
