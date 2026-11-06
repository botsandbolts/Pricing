"""Package and test the exact validation-selected DeepCross AFT MiniLM checkpoint.

The train-only TF-IDF/SVD and categorical preprocessing are recreated from the
unchanged 2021 training rows. No release is installed unless ORIGINAL archived
numerical, categorical, text and holdout neural predictions agree. All source
vehicle rows and heldout lists are omitted from the final model bundle.
"""
import argparse,hashlib,json,shutil
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scripts.rebuild_neural_encoders import rebuild
from pricing.neural_experimental_live import DeepCrossAsk

REV="1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

def digest(ids):
    return hashlib.sha256("\n".join(sorted(map(str,ids))).encode()).hexdigest()

def main(args):
    original=Path(args.original);augmented=Path(args.augmented)
    winner=Path(args.winner);target=Path(args.output)
    r=json.loads((winner/"result.json").read_text())
    report=json.loads(Path(args.report).read_text())
    pick=report["winner_by_validation"]
    if ((r.get("status"),r.get("architecture"),r.get("loss"),
         r.get("variant"),r.get("feature_variant"),
         r.get("original_train_rows"),r.get("train_fit"))!=
        ("completed","deepcross","aft","dropout12","minilm",147777,133000)):
        raise ValueError("Wrong selected model result")
    if (pick.get("variant"),pick.get("features"))!=("dropout12","minilm"):
        raise ValueError("Published validation winner differs")
    if abs(r["original_test"]["mae_usd"]-pick["test_mae_usd"])>.02:
        raise ValueError("Winner reported test score mismatch")
    base=json.loads(original.with_suffix(".json").read_text())
    extra=json.loads(augmented.with_suffix(".json").read_text())
    if (base.get("full_train"),base.get("validation"),base.get("test"))!=(147777,31668,31682):
        raise ValueError("Original full asking train/holdout sizes changed")
    if ((extra.get("variant"),extra.get("minilm_encoder_revision"),
         extra.get("minilm_fit_rows"),extra.get("minilm_stopper_rows"))!=
        ("minilm-2-oob-price",REV,133000,14777)):
        raise ValueError("Wrong two-OOB MiniLM provenance")
    if (r["canonical_cohort_sha256"]!=base["cohort_sha256"] or
        base["cohort_sha256"]!=extra["cohort_sha256"]):
        raise ValueError("Source cohort fingerprint changed")
    if r["holdout_row_sha256"]!=base["holdout_row_sha256"]:
        raise ValueError("Original heldout row identities changed")
    heads=json.loads((augmented.parent/"heads.json").read_text())
    if len(heads.get("heads",[]))!=3:
        raise ValueError("Three original single-fold price heads required")
    with np.load(original,allow_pickle=False) as z:
        old={k:z[k] for k in ("ids","phase","num","cat","txt")}
    with np.load(augmented,allow_pickle=False) as z:
        hybrid={k:z[k] for k in ("ids","phase","num","cat","txt")}
    if not np.array_equal(old["ids"],hybrid["ids"]):
        raise ValueError("Original and MiniLM feature order differs")
    frame=pd.read_parquet(args.parquet)
    ids=frame.row_id.astype(str).to_numpy()
    if not np.array_equal(ids,old["ids"].astype(str)):
        raise ValueError("Source Parquet row ids/order differ from original training tensors")
    phase=old["phase"].astype(str)
    train=phase=="train"
    if (len(ids),sum(train),sum(phase=="validation"),sum(phase=="test"))!=(
         211127,147777,31668,31682):
        raise ValueError("Non-original source partitions")
    for section in ("validation","test"):
        if digest(ids[phase==section])!=base["holdout_row_sha256"][section]:
            raise ValueError("Heldout fingerprint mismatch: "+section)
    print("Rebuilding original numeric, categorical and train-only 24-dimensional SVD projector",flush=True)
    n,c,t,cards,state,tfidf,svd=rebuild(frame,train)
    if cards!=base["cardinalities"]:
        raise ValueError("Saved categorical geometry differs")
    if not np.array_equal(c,old["cat"]) or not np.array_equal(c,hybrid["cat"]):
        raise ValueError("Rebuilt training categorical codes differ")
    numeric_max=float(np.max(np.abs(n-old["num"])))
    svd_max=float(np.max(np.abs(t-old["txt"])))
    if numeric_max>1e-5 or svd_max>.003:
        raise ValueError("Recreated original train-only features mismatch, num="+str(numeric_max)+" txt="+str(svd_max))
    if not np.allclose(n,hybrid["num"][:,:7],atol=1e-5,rtol=0):
        raise ValueError("MiniLM hybrid numeric prefix differs")
    if not np.allclose(t,hybrid["txt"],atol=.003,rtol=0):
        raise ValueError("MiniLM hybrid original SVD feature differs")
    target.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(winner/"weights.pt",target/"weights.pt")
    joblib.dump(tfidf,target/"tfidf.joblib",compress=3)
    joblib.dump(svd,target/"svd.joblib",compress=3)
    (target/"preprocessing.json").write_text(json.dumps(state,indent=2))
    (target/"minilm-heads.json").write_text(json.dumps(heads,indent=2))
    manifest=dict(
      status="experimental_verified_serving_candidate",
      variant="dropout12",feature_variant="minilm",architecture="deepcross",
      model_label="DeepCross AFT + MiniLM · 12 percent dropout",
      version="2026-11-06-alpha1",source="Craigslist 2021 advertised asking",
      cardinalities=cards,minilm_encoder_revision=REV,
      minilm_price_log_median=extra["minilm_price_log_median"],
      minilm_price_log_iqr=extra["minilm_price_log_iqr"],
      cohort_sha256=base["cohort_sha256"],
      holdout_row_sha256=base["holdout_row_sha256"],
      training=147777,fit=133000,internal_stopper=14777,
      validation=31668,test=31682,
      validation_mae_usd=r["original_validation"]["mae_usd"],
      test_mae_usd=r["original_test"]["mae_usd"],
      no_wholesale=True,no_current_market_test=True,
      time_transport="Existing independent ageing curve + monthly used-car index",
      benchmark_exploratory_single_seed=True,
      weights_source="GitHub Actions run 42252201025 / deepcross-aft-tune-minilm-dropout12")
    (target/"manifest.json").write_text(json.dumps(manifest,indent=2))
    # Provisional report exists ONLY inside CI before validation. Its content
    # never reaches git if any verification step below fails.
    (target/"verification.json").write_text(json.dumps({"verified":True,"provisional_in_ci":True}))
    serving=DeepCrossAsk(Path(args.root))
    archive=pd.read_csv(winner/"predictions.csv.gz",dtype={"row_id":str})
    if archive.row_id.duplicated().any() or len(archive)!=63350 or set(archive.split)!=set(["validation","test"]):
        raise ValueError("Original archived winner holdout predictions incomplete")
    archive=archive.set_index("row_id")
    # Deterministically choose original withheld rows across both partitions.
    rng=np.random.default_rng(20261106)
    sample=np.concatenate([
      rng.choice(np.flatnonzero(phase=="validation"),size=22,replace=False),
      rng.choice(np.flatnonzero(phase=="test"),size=22,replace=False)])
    relative=[];text_feature=[];checked=0
    for i in sample:
        row=frame.iloc[int(i)]
        if pd.isna(row["year"]) or pd.isna(row["odometer"]):continue
        description="" if pd.isna(row["text"]) else str(row["text"])
        if len(description)>8000 or row["year"]>2021 or row["year"]<1900:continue
        fields={key:row.get(key) for key in state["categorical_names"]}
        fields.update(year=row["year"],odometer=row["odometer"])
        prep=serving.prepare_text(description)
        numerics,categories,text=serving.encode_inputs(fields,prep)
        if not np.array_equal(categories[0],hybrid["cat"][i]):
            raise ValueError("Live category encoding differs from original row")
        if not np.allclose(numerics[0,:7],hybrid["num"][i,:7],atol=2e-5,rtol=0):
            raise ValueError("Live numeric scaler differs from original row")
        if not np.allclose(text[0],hybrid["txt"][i],atol=.003,rtol=0):
            raise ValueError("Live SVD features differ from original row")
        text_feature.append(abs(float(numerics[0,7])-float(hybrid["num"][i,7])))
        prediction=serving.predict_prepared(fields,prep)["price"]
        original_price=float(archive.loc[str(row["row_id"]),"predicted_price"])
        relative.append(abs(prediction-original_price)/max(original_price,1))
        checked+=1
        print("PARITY",checked,"of 44 relative error",round(relative[-1],6),flush=True)
    if checked<30:
        raise ValueError("Insufficient eligible real untouched holdout parity samples")
    if max(text_feature)>.012 or max(relative)>.020 or np.median(relative)>.005:
        raise ValueError("MiniLM+DeepCross holdout inference differs from archived model: "+
                         "text="+str(max(text_feature))+" max_rel="+str(max(relative)))
    verification={
        "verified":True,"status":"passed",
        "original_holdout_samples_checked":checked,
        "maximum_original_numeric_feature_difference":numeric_max,
        "maximum_original_svd_feature_difference":svd_max,
        "maximum_miniLM_scalar_difference":float(max(text_feature)),
        "median_relative_live_vs_original_checkpoint":float(np.median(relative)),
        "maximum_relative_live_vs_original_checkpoint":float(max(relative)),
        "original_holdout_sha256":base["holdout_row_sha256"],
        "retrained_neural_weights":False,
        "validation_and_test_prices_used_to_fit_preprocessors":False,
        "user_descriptions_in_package":False}
    (target/"verification.json").write_text(json.dumps(verification,indent=2))
    if any(p.stat().st_size>95000000 for p in target.iterdir() if p.is_file()):
        raise ValueError("Serving artifact exceeds GitHub file-size budget")
    print("VERIFIED NEURAL LAB ALPHA RELEASE",json.dumps(verification),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--winner",default="training-cache/neural-winner")
    p.add_argument("--original",default="training-cache/neural-ask/features.npz")
    p.add_argument("--augmented",default="training-cache/ask-minilm/full/features.npz")
    p.add_argument("--parquet",default="training-cache/neural-ask/canonical-ask.parquet")
    p.add_argument("--report",default="reports/deepcross-aft-tuning/report.json")
    p.add_argument("--root",default=".")
    p.add_argument("--output",default="models/neural-experimental-ask")
    main(p.parse_args())
