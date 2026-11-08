"""One-time immutable Lemon75 DeepCross Ask original TEST retrospective audit.

Do NOT select hyperparameters or promote a model using this test. The
worst 25% are chosen only by the separately frozen pre-Lemon reference's
absolute log-price residual, which requires observed TEST outcomes. This
is a CONDITIONAL post-hoc subset and NOT independent prospective accuracy.
Outputs only after original cohort, prices, model SHA, and fixed validation
mask match the release. No replacement of the proven $8,959.62 validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from pricing.ray_lemon75_ask_live import manifest, FOLDER, same_architecture
from scripts.ask25_joint_search import ResearchAskingNetwork
from scripts.cvar95_historical import risk
from scripts.neural_ask_train import predict
from scripts.raw_embedding_ask_train import frozen_source
from scripts.ray_pbt_ask import architecture_params
from scripts.release_ray_lemon75_ask import reference_retained_mask, sha_ids

TEST_ROWS=31682
REMOVE_TEST=math.ceil(TEST_ROWS*.25)
KEEP_TEST=TEST_ROWS-REMOVE_TEST


def audit(args):
    root=Path(args.root)
    m=manifest(root)
    if m is None:
        raise ValueError("Original deployed Lemon75 Ask release has not passed its verification gate")
    if (m.get("original_test_scored") is not False
            or m.get("training_cohort")!="frozen-lemon-oof-retained-75"
            or m["filtered_validation_rows"]!=23751):
        raise ValueError("Original release identity changed")
    weights=root/FOLDER/"weights.pt"
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=m["native_sha256"]:
        raise ValueError("Exact production model checkpoint checksum differs")
    pack=torch.load(weights,map_location="cpu",weights_only=True)
    architecture=architecture_params("deepcross")
    if (not same_architecture(pack["architecture"],architecture)
            or pack.get("cohort_sha256")!=m["cohort_sha256"]
            or pack.get("original_test_scored") is not False):
        raise ValueError("Original native Ray weights or cohort mismatch")
    z,emb,src,enc=frozen_source(args.features,args.embeddings,"bge-256")
    if (src["cohort_sha256"]!=m["cohort_sha256"]
            or src["holdout_row_sha256"]!=m["holdout_row_sha256"]
            or pack.get("frozen_bge_sha256")!=enc["embedding_file_sha256"]):
        raise ValueError("Pinned historical features, holdout IDs, or BGE changed")
    with np.load(args.features,allow_pickle=False) as original:
        if "lemon" not in original.files:
            raise ValueError("Pre-existing immutable Lemon TRAIN mask is missing")
        z["lemon"]=original["lemon"]
    reference=pd.read_csv(args.reference,dtype={"row_id":str},
                          usecols=["row_id","split","observed_price","residual_log"])
    # Verify exact original 75% validation membership before attempting
    # any test. This is the same independently archived source reference.
    mask=reference_retained_mask(z,reference)
    if len(mask)!=31668 or mask.sum()!=23751:
        raise ValueError("Frozen validation study changed")
    test_ref=reference.loc[reference.split=="test"].copy()
    if len(test_ref)!=TEST_ROWS:
        raise ValueError(
            "Original pre-Lemon reference has no complete TEST residuals; "
            "cannot honestly claim a Lemon-cleaned TEST score")
    phase=z["phase"].astype(str)
    ix=np.flatnonzero(phase=="test")
    ids=z["ids"][ix].astype(str)
    y=z["y"][ix].astype(np.float64)
    if len(ix)!=TEST_ROWS or len(set(ids))!=TEST_ROWS:
        raise ValueError("Original immutable TEST partition changed")
    if (test_ref.row_id.isna().any() or test_ref.row_id.duplicated().any()
            or set(test_ref.row_id.astype(str))!=set(ids)
            or not np.isfinite(test_ref.residual_log.to_numpy(float)).all()):
        raise ValueError("Test residual reference lacks unique matched canonical IDs")
    source_y=test_ref.set_index("row_id").loc[ids,"observed_price"].to_numpy(float)
    if not np.allclose(source_y,y,rtol=1e-6,atol=.005):
        raise ValueError("Original reference test prices do not match immutable labels")
    test_ref["absolute_baseline_log_residual"]=test_ref.residual_log.abs()
    ordered=test_ref.sort_values(
        ["absolute_baseline_log_residual","row_id"],ascending=[False,True])
    retained=ordered.iloc[REMOVE_TEST:].row_id.astype(str)
    kept=np.isin(ids,retained.to_numpy())
    if len(retained)!=KEEP_TEST or int(kept.sum())!=KEEP_TEST:
        raise ValueError("Fixed 25% reference TEST filter row counts differ")
    network=ResearchAskingNetwork(src["cardinalities"],384,architecture)
    network.load_state_dict(pack["state_dict"],strict=True)
    network.eval()
    torch.set_num_threads(2)
    ds=TensorDataset(
        torch.from_numpy(z["num"][ix].astype(np.float32)),
        torch.from_numpy(z["cat"][ix].astype(np.int64)),
        torch.from_numpy(emb[ix].astype(np.float32)),
        torch.from_numpy(np.log(y).astype(np.float32)))
    p=predict(network,ds,np.arange(TEST_ROWS),768)
    if len(p)!=TEST_ROWS or not np.isfinite(p).all() or (p<=0).any():
        raise ValueError("Native DeepCross TEST predictions invalid")
    stats=risk(y[kept],p[kept])
    metric={k:float(stats[k]) for k in
            ("CVaR95_absolute_USD","MAE_USD","RMSE_USD")}
    if not all(np.isfinite(v) and v>0 for v in metric.values()):
        raise ValueError("Calculated Lemon-filtered test metric invalid")
    document={
        "status":"verified","market":"ask",
        "test_scored_after_training_and_validation_selection":True,
        "test_never_used_to_select_hyperparameters":True,
        "model":"ray-lemon75-deepcross-original-fp32",
        "release_date":"2026-11-08",
        "source_run_id":42485146978,
        "model_native_sha256":m["native_sha256"],
        "feature_cohort_sha256":m["cohort_sha256"],
        "original_test_rows":TEST_ROWS,
        "excluded_test_rows":REMOVE_TEST,
        "retained_test_rows":KEEP_TEST,
        "retained_test_ids_sha256":sha_ids(ids[kept]),
        "test_reference_uses_actual_observed_prices":True,
        "test_filter":"exact original pre-Lemon reference top25 absolute log residual, row_id tie breaker",
        "filtered_test":{"rows":KEEP_TEST,**metric},
        "interpretation":"Outcome-conditioned retrospective test subset, optimistic for unseen vehicles; not a prospectively usable filter or independent unbiased generalization accuracy.",
        "original_validation_metric_is_separate":True,
        "production_model_not_replaced":True,
    }
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(document,indent=2,allow_nan=False))
    print("LEMON75_FILTERED_ORIGINAL_TEST_VERIFIED",json.dumps({
        "filtered_test":document["filtered_test"],
        "retained_rows":KEEP_TEST,
        "conditional_test_not_unbiased":True,
        "source_run":42485146978,
        "promote_model":False}),flush=True)
    return document


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default=".")
    p.add_argument("--features",required=True)
    p.add_argument("--embeddings",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--output",default=
        "reports/cvar95/lemon75-ask-cleaned-test/report.json")
    audit(p.parse_args())
