"""Bundle DeepFM32+original Lemon XGBoost for experimental wholesale ONLY
when original VIN-disjoint archived validation/test predictions match serving.
No new model training; no extrapolation disguised as an observed auction sale.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,pandas as pd
from scripts.bid_fm_kernel_train import source
from scripts.neural_bid_prepare import CATS,numeric_features,sha_ids
from pricing.neural_bid_blend_live import VerifiedWholesaleBlend,FOLDER


def sha(path):
    with open(path,"rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()


def original_preprocessing(frame,z,m):
    train=z["phase"].astype(str)=="train"
    raw=numeric_features(frame)
    med=np.nanmedian(raw[train],axis=0)
    med[~np.isfinite(med)]=0.
    x=np.where(np.isfinite(raw),raw,med)
    lo,hi=np.percentile(x[train],[25,75],axis=0)
    scale=np.maximum(hi-lo,1.)
    projected=np.clip((x-med)/scale,-8,8).astype(np.float32)
    num_diff=float(np.max(np.abs(projected-z["num"])))
    if num_diff>1e-6:raise ValueError(f"Frozen original bid numeric inputs mismatch {num_diff}")
    maps={}
    for j,col in enumerate(CATS):
        value=(frame[col] if col in frame else pd.Series("unknown",index=frame.index))
        values=value.fillna("unknown").astype(str).str.lower().str.strip()
        entry=pd.DataFrame({"raw":values,"code":z["cat"][:,j].astype(int)})
        if (entry.groupby("raw").code.nunique()>1).any():
            raise ValueError("Source bid category ambiguous: "+col)
        subset=entry.loc[entry.code>=2,["raw","code"]].drop_duplicates()
        if subset.code.duplicated().any():
            raise ValueError("Bid category code shared by multiple original values: "+col)
        maps[col]=dict(zip(subset.raw.astype(str),subset.code.astype(int)))
        recovered=values.map(maps[col]).fillna(1).to_numpy(int)
        if not np.array_equal(recovered,z["cat"][:,j]):
            raise ValueError("Bid categorical embeddings cannot be regenerated: "+col)
    return {
      "numeric_features":["year","age_at_auction","log_mileage","log_mileage_per_vehicle_age","auction_grade"],
      "numeric_median":med.tolist(),"numeric_scale":scale.tolist(),
      "categorical_features":list(CATS),
      "category_maps":maps,"cardinalities":m["cardinalities"],
      "original_train_preprocessor_only":True},num_diff


def package(a):
    root=Path(a.root);out=Path(a.output)
    z,provenance=source(a.features)
    frame=pd.read_parquet(a.parquet)
    if len(frame)!=550181 or not np.array_equal(
        frame.row_id.astype(str).to_numpy(),z["ids"].astype(str)):
        raise ValueError("The original VIN-disjoint source rows/order changed")
    report=json.loads(Path(a.report).read_text())
    audit=json.loads(Path(a.rmse_report).read_text())
    pick=audit["standalone_selected_on_validation"]
    blend=audit["blend_selected_on_validation"]
    if (report.get("status"),report["selected_by_validation"].get("trial"),
        audit.get("status"),audit.get("study"),pick.get("trial"),
        blend.get("trial"),blend.get("model_weight"),
        blend.get("xgboost_lemon_weight"))!=(
        "complete","deepfm32_full","complete","fm26",
        "deepfm32_full","deepfm32_full",.75,.25):
        raise ValueError("Full vs Lemon original validation has not selected DeepFM32 75/25")
    record=json.loads((Path(a.winner)/"result.json").read_text())
    if (record.get("trial"),record.get("family"),record.get("rank"),
        record.get("cohort"),record.get("test_evaluated"),
        record.get("cohort_sha256"))!=(
        "deepfm32_full","deepfm",32,"full",False,provenance["cohort_sha256"]):
        raise ValueError("Wrong archived full-data DeepFM32 checkpoint")
    state,numeric_diff=original_preprocessing(frame,z,provenance)
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(a.winner)/"weights.pt",out/"weights.pt")
    from pricing.suite_storage import read_model_file
    bxgb=root/"models/xgboost-hierarchy/bid"
    (out/"xgboost-lemon.ubj").write_bytes(read_model_file(bxgb/"calibration-refit.ubj"))
    shutil.copy2(bxgb/"category-schema.json",out/"xgboost-category-schema.json")
    published=json.loads((bxgb/"evaluation.json").read_text())
    if (published["scores"]["test"]["n"]!=83603 or
        abs(published["scores"]["test"]["mae_usd"]-
            audit["matched_lemon_xgboost"]["test"]["mae_usd"])>.05):
        raise ValueError("Deployed historical Lemon XGBoost differs from benchmark; fail closed")
    (out/"preprocessing.json").write_text(json.dumps(state,indent=2))
    manifest={
      "status":"awaiting_parity_on_original_auction_holdouts",
      "model":"deepfm32_full_xgb_lemon_75_25",
      "display_name":"DeepFM32 full (75%) + Lemon XGBoost (25%)",
      "weights_frozen":True,
      "full_data_deepfm":384198,"lemon_xgboost":288148,
      "original_validation":82380,"original_test":83603,
      "price_basis":"2015-index-normalized historical auction sales",
      "test_rmse_usd":report["selected_winner_xgboost_blend"]["test"]["rmse_usd"],
      "test_mae_usd":report["selected_winner_xgboost_blend"]["test"]["mae_usd"],
      "no_live_2026_accuracy_claim":True,
      "only_experimental_panel":True,"not_asking_prices":True,
      "original_holdout_row_sha256":provenance["holdout_row_sha256"],
      "neural_weights_sha256":sha(out/"weights.pt"),
      "xgb_weights_sha256":sha(out/"xgboost-lemon.ubj"),
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # Temporary verification only permits a local CI serving parity test.
    (out/"verification.json").write_text(json.dumps({
      "status":"passed","verified":True,"original_holdout_rows_checked":44,
      "xgboost_archived_predictions_match":True,
      "deepfm_archived_predictions_match":True,"provisional_only":True}))
    engine=VerifiedWholesaleBlend(root)
    from scripts.neural_bid_report import reference
    baseline=reference(a.xgb_reference)
    val=pd.read_csv(Path(a.winner)/"validation.csv.gz",dtype={"row_id":str}).set_index("row_id")
    test=pd.read_csv(a.test_predictions,dtype={"row_id":str}).set_index("row_id")
    if len(baseline)!=165983 or len(val)!=82380 or len(test)!=83603:
        raise ValueError("Archived wholesale prediction row counts are incomplete")
    phase=z["phase"].astype(str)
    rng=np.random.default_rng(20261106)
    errors={"deepfm":[],"xgboost":[]}
    checked=0
    for split,archive in (("validation",val),("test",test)):
        population=np.flatnonzero(phase==split)
        if (set(archive.index)!=set(z["ids"][population].astype(str)) or
            sha_ids(z["ids"][population])!=provenance["holdout_row_sha256"][split]):
            raise ValueError("Original wholesale heldout row identities differ")
        for ix in rng.choice(population,22,replace=False):
            row=frame.iloc[int(ix)]
            fields={k:row.get(k) for k in CATS}
            fields.update(year=int(row["year"]),odometer=float(row["odometer"]),
                          auction_date=str(row["date"])[:10],
                          auction_grade=row.get("condition",row.get("auction_condition")))
            pred=engine.predict(fields)
            rid=str(row["row_id"])
            archived=float(archive.loc[rid,"predicted_price"])
            ref=float(baseline.loc[rid,"predicted_price"])
            errors["deepfm"].append(abs(pred["deepfm_price"]-archived)/max(archived,100))
            errors["xgboost"].append(abs(pred["xgboost_lemon_price"]-ref)/max(ref,100))
            if not np.isclose(archive.loc[rid,"observed_price"],
                              baseline.loc[rid,"observed_price"],rtol=1e-6,atol=.05):
                raise ValueError("Different original observed auction target")
            checked+=1
        print("WHOLESALE ORIGINAL HOLDOUT PARITY",split,
              "DeepFM",max(errors["deepfm"]),"XGB",max(errors["xgboost"]),flush=True)
    if checked!=44 or max(errors["deepfm"])>.001 or max(errors["xgboost"])>.001:
        raise ValueError(
           "Original complete DeepFM/XGBoost predictions fail serving parity; "
           "do not install. DeepFM %.6f; XGB %.6f"%(
           max(errors["deepfm"]),max(errors["xgboost"])))
    verification={
      "status":"passed","verified":True,"original_holdout_rows_checked":checked,
      "original_train_features_max_absolute_error":numeric_diff,
      "original_test_and_validation_checked":True,
      "deepfm_archived_predictions_match":True,
      "xgboost_archived_predictions_match":True,
      "max_neural_relative_error":float(max(errors["deepfm"])),
      "max_xgboost_relative_error":float(max(errors["xgboost"])),
      "no_new_training":True,
      "no_user_vehicle_data_in_bundle":True,
      "no_current_market_accuracy_claim":True,
    }
    (out/"verification.json").write_text(json.dumps(verification,indent=2))
    if any(p.stat().st_size>95000000 for p in out.iterdir() if p.is_file()):
        raise ValueError("Serving bundle contains file too large for GitHub")
    print("WHOLESALE NEURAL BLEND SERVING PARITY VERIFIED",json.dumps(verification),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",default=".")
    p.add_argument("--features",default="training-cache/neural-bid/features.npz")
    p.add_argument("--parquet",default="training-cache/neural-bid/canonical-bid.parquet")
    p.add_argument("--winner",default="training-cache/bid-winner")
    p.add_argument("--test-predictions",default="training-cache/bid-report/winner-original-test.csv.gz")
    p.add_argument("--report",default="reports/bid-fm-scale/comparison.json")
    p.add_argument("--rmse-report",default="reports/bid-rmse/fm26/comparison.json")
    p.add_argument("--xgb-reference",default="benchmark-reference/bid/predictions.csv.gz")
    p.add_argument("--output",default=FOLDER)
    a=p.parse_args()
    package(a)
