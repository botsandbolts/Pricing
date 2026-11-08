"""Release the original frozen Ray Lemon75 DeepCross Ask checkpoint, in FP32.

No retraining or test peeking. The 75%-retained validation is a fixed,
outcome-dependent retrospective cohort and MUST be labeled accordingly.
Release fails if original checkpoint provenance, frozen features, full
validation replay or exact retained-subset validation fails.
The native weights are deployed unchanged. No neural-head quantization.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from pricing.ray_lemon75_ask_live import _hash, FOLDER, STUDY_RUN, STUDY_ARTIFACT
from scripts.ask25_joint_search import ResearchAskingNetwork
from scripts.cvar95_historical import risk
from scripts.neural_ask_train import predict
from scripts.raw_embedding_ask_train import frozen_source
from scripts.ray_pbt_ask import architecture_params

TRAIN_ROWS = 147777
LEMON_TRAIN = 110832
VALIDATION_ROWS = 31668
RETAINED_VALIDATION = 23751
HOLDOUT_SHA = "aeccb9ddbcf103d8bc603659c1aa3f62efe39d6b3d5f9f5451e3b5a313ca27ac"
KNOWN_RETAINED_CVAR95 = 8959.618612064656




def same_architecture(actual,expected):
    """Strict keys/structure with float serialization tolerance only.

    ConfigSpace/NumPy versions can round the last float bit differently.
    This never permits a changed layer width, family, block count or key.
    """
    import math
    if not isinstance(actual,dict) or set(actual)!=set(expected):
        return False
    for name,value in expected.items():
        got=actual[name]
        if type(value) in (int,float) and type(got) in (int,float):
            if not math.isclose(float(got),float(value),rel_tol=1e-12,abs_tol=1e-12):
                return False
        elif got!=value:
            return False
    return True

def sha_ids(values):
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def metrics(y,p):
    d=risk(np.asarray(y,dtype=np.float64),
           np.asarray(p,dtype=np.float64))
    return {k:float(d[k]) for k in ("rows","CVaR95_absolute_USD",
                                   "MAE_USD","RMSE_USD")}


def reference_retained_mask(z,frame):
    required={"row_id","split","observed_price","residual_log"}
    if not required.issubset(frame.columns) or frame.row_id.isna().any():
        raise ValueError("Missing or invalid original reference fields")
    if frame.row_id.duplicated().any() or not np.isfinite(
            frame.residual_log.to_numpy(float)).all():
        raise ValueError("Reference duplicate row IDs or residuals")
    phase=z["phase"].astype(str)
    ids=z["ids"].astype(str)
    tr=frame.loc[frame.split=="calibration"].copy()
    va=frame.loc[frame.split=="validation"].copy()
    if (len(tr),len(va))!=(TRAIN_ROWS,VALIDATION_ROWS):
        raise ValueError("Original reference data count differs")
    if set(tr.row_id.astype(str))!=set(ids[phase=="train"]):
        raise ValueError("Historical training car IDs differ")
    tr["abs_ref"]=tr.residual_log.abs()
    tr=tr.sort_values(["abs_ref","row_id"],ascending=[False,True])
    kept_train=set(tr.iloc[math.ceil(.25*TRAIN_ROWS):].row_id.astype(str))
    source_train=set(ids[(phase=="train") & z["lemon"].astype(bool)])
    if kept_train!=source_train or len(kept_train)!=LEMON_TRAIN:
        raise ValueError("Lemon training membership differs from original OOF filter")
    va["abs_ref"]=va.residual_log.abs()
    va=va.sort_values(["abs_ref","row_id"],ascending=[False,True])
    retained=va.iloc[math.ceil(.25*VALIDATION_ROWS):].row_id.astype(str)
    if len(retained)!=RETAINED_VALIDATION or sha_ids(retained)!=HOLDOUT_SHA:
        raise ValueError("Lemon retained validation membership differs")
    indices=np.flatnonzero(phase=="validation")
    subset=np.isin(ids[indices],retained.to_numpy())
    if int(subset.sum())!=RETAINED_VALIDATION:
        raise ValueError("Validation subset failed canonical row-ID join")
    compare=va.set_index("row_id").loc[ids[indices]]
    if not np.allclose(compare.observed_price.to_numpy(float),
                       z["y"][indices],rtol=1e-6,atol=.005):
        raise ValueError("Reference validation targets differ")
    return subset


def audit(args):
    src=Path(args.source)
    out=Path(args.output)
    original=json.loads((src/"result.json").read_text())
    if not (original.get("status")=="complete"
            and original.get("study")==
                "ray-tune-population-based-training-frozen-lemon-75"
            and original.get("architecture")=="deepcross"
            and original.get("training_cohort")==
                "frozen-lemon-oof-retained-75"
            and original.get("original_test_scored") is False
            and original.get("autodeploy") is False
            and original["provenance"]["lemon_retained_train_rows"]==LEMON_TRAIN
            and original["provenance"]["train_fit_rows"]==99749
            and original["provenance"]["train_only_stopper_rows"]==11083):
        raise ValueError("Source artifact is not exact train-selected Ray Lemon winner")
    pack=torch.load(src/"weights.pt",map_location="cpu",weights_only=True)
    cfg=architecture_params("deepcross")
    if not (same_architecture(pack.get("architecture"),cfg)
            and pack.get("training_cohort")=="frozen-lemon-oof-retained-75"
            and pack.get("original_test_scored") is False
            and pack.get("cohort_sha256")==original["provenance"]["cohort_sha256"]
            and pack.get("frozen_bge_sha256")==
                original["provenance"]["embedding_file_sha256"]):
        raise ValueError("Native Ray checkpoint does not match audited frozen source")
    z,vec,source,encoder=frozen_source(args.features,args.embeddings,"bge-256")
    # frozen_source intentionally exports only predictive features; Lemon is
    # a separate provenance mask stored in the original immutable NPZ.
    with np.load(args.features,allow_pickle=False) as original_store:
        if "lemon" not in original_store.files:
            raise ValueError("Original frozen Lemon mask missing")
        lemon=original_store["lemon"]
    if lemon.dtype!=np.bool_ or lemon.shape!=(len(z["phase"]),):
        raise ValueError("Frozen Lemon membership geometry differs")
    z["lemon"]=lemon
    if (source["cohort_sha256"]!=pack["cohort_sha256"]
            or encoder["embedding_file_sha256"]!=pack["frozen_bge_sha256"]
            or source["holdout_row_sha256"]!=pack["holdout_row_sha256"]):
        raise ValueError("Original frozen BGE, features or holdout IDs changed")
    phase=z["phase"].astype(str)
    valid=np.flatnonzero(phase=="validation")
    if len(valid)!=VALIDATION_ROWS or int((phase=="test").sum())!=31682:
        raise ValueError("Immutable chronological holdout count changed")
    net=ResearchAskingNetwork(source["cardinalities"],384,cfg)
    net.load_state_dict(pack["state_dict"],strict=True)
    net.eval()
    source_prediction=pd.read_csv(src/"validation.csv.gz",
        dtype={"row_id":str})
    if (len(source_prediction)!=VALIDATION_ROWS
            or not source_prediction.row_id.is_unique
            or not source_prediction.split.eq("validation").all()
            or list(source_prediction.row_id)!=list(z["ids"][valid].astype(str))):
        raise ValueError("Archived original validation row IDs/order differ")
    dataset=TensorDataset(
        torch.from_numpy(z["num"][valid].astype(np.float32)),
        torch.from_numpy(z["cat"][valid].astype(np.int64)),
        torch.from_numpy(vec[valid].astype(np.float32)),
        torch.from_numpy(np.log(z["y"][valid]).astype(np.float32)))
    torch.set_num_threads(2)
    native=predict(net,dataset,np.arange(len(valid)),1024)
    saved=source_prediction.predicted_price.to_numpy(float)
    if not np.allclose(native,saved,rtol=.00005,atol=.5):
        raise ValueError("Native weights cannot reproduce archived 31,668 predictions")
    full=metrics(z["y"][valid],native)
    for key in ("CVaR95_absolute_USD","MAE_USD","RMSE_USD"):
        if abs(full[key]-original["external_original_validation"][key])>.12:
            raise ValueError("Original 31,668-row validation receipts do not match")
    reference=pd.read_csv(args.reference,dtype={"row_id":str},
        usecols=["row_id","split","observed_price","residual_log"])
    keep=reference_retained_mask(z,reference)
    subset=metrics(z["y"][valid][keep],native[keep])
    if abs(subset["CVaR95_absolute_USD"]-KNOWN_RETAINED_CVAR95)>.2:
        raise ValueError("Lemon-filtered CVaR95 differs from fixed source study")
    # The original FP32 checkpoint already reproduced all archived
    # validation predictions and the fixed Lemon75 risk metric above.
    # Do not convert, quantize, export or retrain its prediction head.
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src/"weights.pt",out/"weights.pt")
    if _hash(out/"weights.pt") != _hash(src/"weights.pt"):
        raise ValueError("Unmodified native source checkpoint was not preserved")
    manifest={
        "status":"passed","release_date":"2026-11-08","market":"ask",
        "study_run_id":STUDY_RUN,"artifact_id":STUDY_ARTIFACT,
        "architecture":"deepcross",
        "training_cohort":"frozen-lemon-oof-retained-75",
        "lemon_retained_train_rows":LEMON_TRAIN,
        "lemon_removed_train_rows":36945,
        "filtered_validation_rows":RETAINED_VALIDATION,
        "original_validation_rows":VALIDATION_ROWS,
        "filtered_validation_is_conditional":True,
        "filtered_validation_uses_observed_reference_residuals":True,
        "filtered_validation":subset,
        "original_test_scored":False,
        "native_head_for_explanations":True,
        "native_checkpoint_for_predictions":True,
        "quantization":"none-native-fp32",
        "inference_backend":"pytorch-native-fp32",
        "shared_encoder":"original frozen BGE; separately pinned INT8 if verified",
        "native_sha256":_hash(out/"weights.pt"),
        "cohort_sha256":source["cohort_sha256"],
        "holdout_row_sha256":source["holdout_row_sha256"],
        "retained_validation_ids_sha256":HOLDOUT_SHA,
        "source_selected_best_epoch":original["selected_best_epoch"],
        "legacy_production_unchanged_until_release_verified":True,
        "current_november_2026_accuracy_verified":False,
        "evaluation_interpretation":(
            "Filtered 75% selected retrospectively using observed prices "
            "and independent baseline residuals, not prospective accuracy."),
    }
    verification={
        "status":"passed","verified":True,
        "complete_original_validation_replay":True,
        "filtered_validation_replay":True,
        "native_checkpoint_replay":True,
        "unmodified_checkpoint_sha256":manifest["native_sha256"],
        "production_source_unchanged":True,
        "test_evaluated":False,
        "full_reference_mask_sha256":HOLDOUT_SHA,
        "samples_verified":VALIDATION_ROWS,
        "retained_validation_rows":RETAINED_VALIDATION,
        "native_sha256":manifest["native_sha256"],
        "quantization":"none",
    }
    (out/"manifest.json").write_text(json.dumps(
        manifest,indent=2,allow_nan=False))
    (out/"verification.json").write_text(json.dumps(
        verification,indent=2,allow_nan=False))
    print("LEMON75_NATIVE_FP32_RELEASE_VERIFIED",json.dumps({
        "filtered_validation":subset,
        "original_validation":full,
        "retained_validation_rows":RETAINED_VALIDATION,
        "native_sha256":manifest["native_sha256"],
        "test_scored":False,"automatic_deployment":False}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--features",required=True)
    p.add_argument("--embeddings",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--output",default=FOLDER)
    audit(p.parse_args())
