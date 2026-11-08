"""One-time, original-cohort audit and packaging of exact BOHB trial 36 / 116.

Never retrain, change feature columns, promote from validation alone, or allow
a misleading successful deployment when original untouched TEST regresses.
Weights remain native PyTorch; original Lemon XGBoost stays unmodified.
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
from scripts.overnight_bohb_neural import build_ask,StructuredBid,_bid_predict
from pricing.overnight_bohb_live import FOLDERS,WINNERS,file_hash
from scripts.release_cvar95_champions import compare,_bid_baseline


REPORT_RUN=42406868569
VAL={"ask":31668,"bid":82380}
TEST={"ask":31682,"bid":83603}
COHORT_IDS={
    "ask":"36268fb7de21cd84bf00dac443907afceb8e7e1f1bf8729d9babff9d09e1976a",
    "bid":"319f0b1bfeeec3b1692092cbb88841282e6a116e8483a4466fed0a3d02896fa8",
}


def _verify_select(market,args):
    report=json.loads(Path(args.report).read_text())
    m=report["compared_to_deployed_verified"][market]
    archived=json.loads((Path(args.winner)/"summary.json").read_text())
    receipt=json.loads((Path(args.winner)/"result.json").read_text())
    if (report.get("status")!="complete" or len(report["records"])!=240
        or report.get("failed_counts")!={"ask":0,"bid":0}
        or report["test_evaluated"] is not False
        or archived.get("market")!=market
        or archived.get("state")!="COMPLETE"
        or archived.get("budget")!=216
        or archived.get("trial")!=WINNERS[market]
        or m["challenger"]["trial"]!=WINNERS[market]
        or m["historical_validation_cohort_sha256"]!=COHORT_IDS[market]):
        raise ValueError("Final BOHB study selection or holdout cohort was altered")
    record=[x for x in report["records"] if (x["market"],x["trial"])==
            (market,WINNERS[market])]
    if len(record)!=1 or record[0]!=archived:
        raise ValueError("Archived candidate does not match the final independent audit")
    if (receipt.get("status")!="completed"
        or receipt.get("trial")!=archived["origin_trial"]
        or receipt.get("trial_configuration")!=archived["config"]
        or receipt.get("test_evaluated") is not False
        or archived.get("test_evaluated") is not False):
        raise ValueError("Archived checkpoint provenance or trial configuration changed")
    ckpt=torch.load(Path(args.winner)/"weights.pt",map_location="cpu",weights_only=True)
    if (ckpt.get("trial")!=archived["origin_trial"]
        or ckpt.get("trial_configuration")!=archived["config"]
        or ckpt.get("search_group")!=market+"_refined"):
        raise ValueError("Checkpoint is not the audited original BOHB training artifact")
    for k in ("CVaR95_absolute_USD","RMSE_USD","MAE_USD"):
        if abs(float(receipt["validation"][k])-float(archived["validation"][k]))>1e-8:
            raise ValueError("BOHB receipt metrics differ from audited trial")
        if abs(float(m["challenger"][k])-float(archived["validation"][k]))>1e-8:
            raise ValueError("Winner metrics differ from final run leaderboard")
    return archived,ckpt,receipt


def _compare_predictions(original,frame,rows,expected,identity,pred):
    aligned,y=compare(original,frame,expected,identity)
    if not np.allclose(aligned.predicted_price.to_numpy(float),pred,
                       atol=.30,rtol=.00007):
        raise ValueError("All ORIGINAL validation predictions fail byte-archived native parity")
    return y


def _check_metrics(actual,registered):
    for name in ("CVaR95_absolute_USD","MAE_USD","RMSE_USD"):
        if abs(actual[name]-registered[name])>.1:
            raise ValueError("BOHB original validation metrics failed independent replay")


def _production_incumbent(root,market):
    path=Path(root)/("models/neural-experimental-ask-cvar95/manifest.json" if
                     market=="ask" else "models/neural-experimental-bid-cvar95/manifest.json")
    doc=json.loads(path.read_text())
    if doc["source_study"]!=("ask25-joint" if market=="ask" else "bid_hparams"):
        raise ValueError("Previous deployed incumbent identity changed")
    return doc["historical_test"]


def _package(root,market,archived,receipt,out,metrics_val,metrics_test,old):
    if metrics_test["CVaR95_absolute_USD"]>=old["CVaR95_absolute_USD"]-.01:
        raise ValueError(
            "Untouched original TEST rejects BOHB deployment: "
            f"{market}: BOHB CVaR95={metrics_test['CVaR95_absolute_USD']:.2f}, "
            f"current={old['CVaR95_absolute_USD']:.2f}")
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(Path(root)/"winner"/"weights.pt",out/"weights.pt")
    manifest={
        "status":"release_candidate_verified","market":market,
        "source_study":"overnight-wild-bohb","source_workflow_run":REPORT_RUN,
        "trial":archived["trial"],"origin_trial":archived["origin_trial"],
        "config":archived["config"],"checkpoint_sha256":file_hash(out/"weights.pt"),
        "original_holdout_row_sha256":receipt["provenance"]["holdout_row_sha256"],
        "cohort_sha256":receipt["provenance"]["dataset_sha256"],
        "historical_validation":metrics_val,"historical_test":metrics_test,
        "historical_incumbent_test":old,
        "neural_weight":archived["config"]["hyperparameters"].get("blend_alpha"),
        "lemon_weight":(
            1.-float(archived["config"]["hyperparameters"]["blend_alpha"])
            if market=="bid" else None),
        "original_test_rows_scored":TEST[market],
        "test_selected":False,"native_pytorch_head":True,
        "bge_in_bid_model":False,"no_duplicate_bge_inference":True,
        "original_lemon_xgboost_reused":market=="bid",
        "production_main_calculator_model":"BOHB",
        "not_2026_market_accuracy":True,
        "price_basis":"2021-05 advertised USD" if market=="ask"
              else "2015-index-normalized wholesale auction USD",
        "bge_encoder_backend":"shared preexisting BGE; INT8 ONNX only if separately audited"
              if market=="ask" else "not applicable; auction has no descriptions"
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    (out/"verification.json").write_text(json.dumps({"status":"pending","verified":False}))
    return manifest


def _write_verified(out,market,manifest,sample_count):
    receipt={
        "status":"passed","verified":True,
        "complete_validation_parity":True,
        "full_original_test_scored":True,
        "original_test_rows_scored":TEST[market],
        "serving_prediction_parity":True,
        "serving_validation_sample_rows":sample_count,
        "candidate_test_cvar95_improves_incumbent":True,
        "weight_sha256":manifest["checkpoint_sha256"],
        "native_original_dataset_fingerprint_validated":True,
        "original_test_not_used_in_hyperparameter_search":True,
        "shared_original_preprocessing_reused":True,
        "no_new_training_or_encoder_heads":True,
    }
    (out/"verification.json").write_text(json.dumps(receipt,indent=2,allow_nan=False))
    print("BOHB_RELEASE_VERIFIED",market,
        json.dumps({"validation":manifest["historical_validation"],
                    "test":manifest["historical_test"],
                    "incumbent_test":manifest["historical_incumbent_test"],
                    "weight_sha256":manifest["checkpoint_sha256"]}),flush=True)


def ask(args,archived,pack,receipt):
    from scripts.raw_embedding_ask_train import frozen_source
    from scripts.neural_ask_train import predict
    from pricing.overnight_bohb_live import BOHBAsking
    z,emb,provenance,meta=frozen_source(args.features,args.embeddings,"bge-256")
    if (provenance["cohort_sha256"]!=receipt["provenance"]["dataset_sha256"]
        or provenance["holdout_row_sha256"]!=receipt["provenance"]["holdout_row_sha256"]
        or provenance["holdout_row_sha256"]["validation"]!=COHORT_IDS["ask"]
        or meta["embedding_file_sha256"]!=receipt["provenance"]["frozen_bge_sha256"]):
        raise ValueError("Ask original cohort or frozen BGE weights changed")
    cfg=archived["config"]
    model=build_ask(provenance["cardinalities"],384,"ask_structure",cfg["structure"])
    model.load_state_dict(pack["state_dict"],strict=True);model.eval()
    ds=TensorDataset(torch.from_numpy(z["num"].astype(np.float32)),
        torch.from_numpy(z["cat"].astype(np.int64)),torch.from_numpy(emb),
        torch.from_numpy(np.log(z["y"]).astype(np.float32)))
    phase=z["phase"].astype(str)
    v=np.flatnonzero(phase=="validation"); t=np.flatnonzero(phase=="test")
    if len(v)!=31668 or len(t)!=31682:raise ValueError("Ask original row partitions changed")
    p=predict(model,ds,v,768)
    y=_compare_predictions(z,Path(args.winner)/"validation.csv.gz",v,VAL["ask"],
                           COHORT_IDS["ask"],p)
    vr=risk(y,p)
    _check_metrics(vr,archived["validation"])
    tp=predict(model,ds,t,768)
    tr=risk(z["y"][t].astype(float),tp)
    old=_production_incumbent(args.root,"ask")
    out=Path(args.output)
    manifest=_package(Path(args.winner).parent,"ask",archived,receipt,out,vr,tr,old)
    engine=BOHBAsking(args.root,_parity_test=True)
    sample=np.random.default_rng(20261108).choice(v,size=32,replace=False)
    with torch.inference_mode():
        q=engine.model(torch.from_numpy(z["num"][sample].astype(np.float32)),
            torch.from_numpy(z["cat"][sample].astype(np.int64)),
            torch.from_numpy(emb[sample]))[:,0].numpy()
    actual=np.exp(np.clip(q,np.log(100.),np.log(1e6)))
    if not np.allclose(actual,p[np.searchsorted(v,sample)],rtol=.0001,atol=.5):
        raise ValueError("BOHB native Ask serving head fails validation row parity")
    _write_verified(out,"ask",manifest,len(sample))


def bid(args,archived,pack,receipt):
    from scripts.bid_fm_kernel_train import source,partition
    from pricing.overnight_bohb_live import BOHBWholesale
    z,src=source(args.features)
    if (src["cohort_sha256"]!=receipt["provenance"]["dataset_sha256"]
        or src["holdout_row_sha256"]!=receipt["provenance"]["holdout_row_sha256"]
        or src["holdout_row_sha256"]["validation"]!=COHORT_IDS["bid"]):
        raise ValueError("Bid original VIN-disjoint cohort changed")
    cfg=archived["config"]
    model=StructuredBid(src["cardinalities"],
                        float(pack["state_dict"]["core.fm.base"]),cfg["structure"])
    model.load_state_dict(pack["state_dict"],strict=True);model.eval()
    _,_,v,t=partition(z,"full")
    if len(v)!=82380 or len(t)!=83603:raise ValueError("Bid original row partitions changed")
    n=z["num"].astype(np.float32);c=z["cat"].astype(np.int64)
    y=z["y"].astype(float)
    lemon=_bid_baseline(args.baseline,z)
    alpha=float(cfg["hyperparameters"]["blend_alpha"])
    p=_bid_predict(model,n,c,v)
    combined=alpha*p+(1-alpha)*lemon["validation"]
    truth=_compare_predictions(z,Path(args.winner)/"validation.csv.gz",v,VAL["bid"],
                               COHORT_IDS["bid"],combined)
    vr=risk(truth,combined);_check_metrics(vr,archived["validation"])
    tp=_bid_predict(model,n,c,t)
    test_p=alpha*tp+(1-alpha)*lemon["test"]
    tr=risk(y[t],test_p)
    old=_production_incumbent(args.root,"bid")
    out=Path(args.output)
    manifest=_package(Path(args.winner).parent,"bid",archived,receipt,out,vr,tr,old)
    engine=BOHBWholesale(args.root,_parity_test=True)
    sample=np.random.default_rng(20261108).choice(v,size=40,replace=False)
    with torch.inference_mode():
        raw=engine.model(torch.from_numpy(n[sample]),
                         torch.from_numpy(c[sample])).numpy()
    direct=np.exp(np.clip(raw,np.log(100.),np.log(1e6)))
    if not np.allclose(direct,p[np.searchsorted(v,sample)],rtol=.0001,atol=.5):
        raise ValueError("BOHB DeepFM serving head does not match validation")
    _write_verified(out,"bid",manifest,len(sample))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--market",required=True,choices=("ask","bid"))
    p.add_argument("--root",default=".")
    p.add_argument("--report",required=True)
    p.add_argument("--winner",required=True)
    p.add_argument("--features",required=True)
    p.add_argument("--embeddings",default="training-cache/bge-256")
    p.add_argument("--baseline",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    archived,ckpt,receipt=_verify_select(args.market,args)
    (ask if args.market=="ask" else bid)(args,archived,ckpt,receipt)


if __name__=="__main__":main()
