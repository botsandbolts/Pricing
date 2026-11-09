"""Leakage-conscious, non-deduplicating European ASK-price XGBoost/Ray research.

Two million original ads. Price is an observed asking price, not a sale price.
AFT uses exact interval [ask,ask] only. Huber-AFT is a documented mixture
of exact-event Gaussian AFT NLL and pseudo-Huber on the LOG asking price.
Test data is a separate artifact, never downloaded by tuning workers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261109
MIN_PRICE, MAX_PRICE = 100.0, 1_000_000.0
NUMERIC = ("reg_year", "mileage_km", "power_hp", "reg_month",
           "listing_year", "listing_month", "age_at_listing",
           "km_per_year", "log_km", "doors", "seats",
           "images_count", "accident_free")
CATEGORICAL = ("seller_country", "make", "model", "model_group", "version",
               "seller_city", "fuel", "transmission", "color", "source_site",
               "body_type", "drive_type")
FEATURES = list(NUMERIC + CATEGORICAL)
FEATURE_TYPES = ["q"] * len(NUMERIC) + ["c"] * len(CATEGORICAL)
CAT_CAPS = {"version": 2500, "model": 3000, "seller_city": 2000}
INPUT_COLUMNS = ["id", "source_site", "seller_country", "make", "model",
                 "version", "price_eur", "mileage_km", "first_registration",
                 "reg_year", "power_hp", "seller_city", "color", "fuel",
                 "transmission", "first_seen_at", "model_group", "body_type",
                 "drive_type", "doors", "seats", "images_count", "accident_free"]


def split_groups(frame):
    """Group exact full descriptions/specifications for holdout placement ONLY.

    Similar vehicles are never deleted or assumed identical. All rows matching
    an exact, price-independent fingerprint go into the same cohort.
    """
    columns = ["seller_country", "make", "model", "version",
               "first_registration", "reg_year", "mileage_km", "power_hp",
               "seller_city", "color", "fuel", "transmission"]
    # Hash native typed values directly, avoiding a 24-million-cell string copy.
    fingerprints = pd.util.hash_pandas_object(
        frame[columns], index=False, categorize=True).to_numpy(np.uint64)
    return fingerprints


def split_ids(frame):
    fingerprint = split_groups(frame)
    buckets = fingerprint % np.uint64(100)
    phases = np.full(len(frame), "train", dtype="U10")
    phases[buckets >= 80] = "validation"
    phases[buckets >= 90] = "test"
    # Verify no EXACT identical fingerprints occur across phases.
    checks = [np.unique(fingerprint[phases == name])
              for name in ("train", "validation", "test")]
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if np.intersect1d(checks[a], checks[b], assume_unique=True).size:
            raise AssertionError("Exact fingerprint leaked across holdouts")
    return phases


def _date(col):
    return pd.to_datetime(col, errors="coerce", utc=True, format="mixed")


def encode(frame, phases):
    train = phases == "train"
    x = np.empty((len(frame), len(FEATURES)), dtype=np.float32)
    reg_year = pd.to_numeric(frame["reg_year"], errors="coerce")
    km = pd.to_numeric(frame["mileage_km"], errors="coerce")
    hp = pd.to_numeric(frame["power_hp"], errors="coerce")
    dt = _date(frame["first_registration"])
    seen = _date(frame["first_seen_at"])
    listing_year = seen.dt.year.astype("float64")
    age = listing_year - reg_year
    num = {
        "reg_year": reg_year.where(reg_year.between(1950, 2027)),
        "mileage_km": km.where(km.between(0, 2_000_000)),
        "power_hp": hp.where(hp.between(20, 1200)),
        "reg_month": dt.dt.month.astype("float64"),
        "listing_year": listing_year,
        "listing_month": seen.dt.month.astype("float64"),
        "age_at_listing": age.where(age.between(-2, 90)),
        "km_per_year": (km / age.clip(lower=0.5)).where(age.between(0.5, 90)),
        "log_km": np.log1p(km.where(km.between(0, 2_000_000))),
        "doors": pd.to_numeric(frame["doors"], errors="coerce"),
        "seats": pd.to_numeric(frame["seats"], errors="coerce"),
        "images_count": pd.to_numeric(frame["images_count"], errors="coerce"),
        "accident_free": pd.to_numeric(frame["accident_free"], errors="coerce"),
    }
    for i, name in enumerate(NUMERIC):
        x[:, i] = np.asarray(num[name], dtype=np.float32)
    cat_cardinalities = {}
    for i, name in enumerate(CATEGORICAL, start=len(NUMERIC)):
        raw = frame[name].fillna("__missing__").astype(str).str.strip().str.lower()
        raw = raw.replace("", "__missing__")
        freq = raw[train].value_counts()
        vocab = freq.index[:CAT_CAPS.get(name, 4000)]
        mapping = {value: j for j, value in enumerate(vocab)}
        encoded = raw.map(mapping)
        x[:, i] = encoded.to_numpy(dtype=np.float32, na_value=np.nan)
        cat_cardinalities[name] = len(mapping)
    return x, cat_cardinalities


def ids_sha(ids):
    return hashlib.sha256(np.sort(np.asarray(ids, dtype="<i8")).tobytes()).hexdigest()


def prepare(input_path: Path, output: Path):
    """Produce immutable 80/10/10 splits from RAW asks. No row thinning."""
    import pyarrow.parquet as pq
    output.mkdir(parents=True, exist_ok=True)
    source = pq.ParquetFile(str(input_path))
    columns = list(source.schema_arrow.names)
    required = {"price_eur", "make", "model", "reg_year", "mileage_km"}
    missing = required - set(columns)
    if missing:
        raise ValueError("Missing European essential columns: " + repr(sorted(missing)))
    frame = pq.read_table(str(input_path),
                          columns=[c for c in INPUT_COLUMNS if c in columns]).to_pandas()
    for optional in INPUT_COLUMNS:
        if optional not in frame:
            frame[optional] = None
    raw_rows = len(frame)
    original_id = np.arange(raw_rows, dtype=np.int64)
    price = pd.to_numeric(frame["price_eur"], errors="coerce").to_numpy(float)
    eligible = np.isfinite(price) & (price >= MIN_PRICE) & (price <= MAX_PRICE)
    frame = frame.loc[eligible].reset_index(drop=True)
    original_id = original_id[eligible]
    price = price[eligible]
    if len(frame) < 1000:
        raise ValueError("Too few valid asking-price observations")
    phases = split_ids(frame)
    counts = {p:int(np.sum(phases == p)) for p in ("train", "validation", "test")}
    if any(v < 100 for v in counts.values()):
        raise ValueError("A holdout has fewer than 100 rows")
    x, cats = encode(frame, phases)
    for phase in counts:
        idx = np.flatnonzero(phases == phase)
        np.savez_compressed(output/(phase+".npz"), x=x[idx], y=price[idx],
                            row_id=original_id[idx])
    audit = {
        "dataset":"KTHESAT/european-used-car-market-2m-listings",
        "columns_originally_available":columns,
        "optional_feature_columns_missing":[c for c in INPUT_COLUMNS if c not in columns],
        "market":"ask", "target":"price_eur", "target_meaning":"observed advertised asking price",
        "censored_rows":0,"synthetic_intervals":0,"deduplication_applied":False,
        "raw_rows":raw_rows,"eligible_rows":len(frame),
        "invalid_price_rows_excluded":int(raw_rows - len(frame)),
        "split":"deterministic 80/10/10 exact-price-independent fingerprint hash",
        "split_use_only":"fingerprint groups prevent identical observed specs crossing holdouts, but cannot prove cross-site identity",
        "feature_names":FEATURES,"feature_types":FEATURE_TYPES,
        "categorical_levels_fit_on":"train only","categorical_cardinalities":cats,
        "counts":counts,
        "sha256_by_phase":{p:ids_sha(original_id[phases == p]) for p in counts},
        "test_sealed_until_final_aggregation":True,
        "production_model_unchanged":True
    }
    (output/"manifest.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)
    return audit


def load_data(folder, phase, manifest):
    with np.load(Path(folder)/(phase+".npz"),allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float64)
        row_id = np.asarray(data["row_id"], dtype=np.int64)
    if x.shape != (manifest["counts"][phase], len(FEATURES)):
        raise AssertionError("Split matrix count/schema mismatch")
    if ids_sha(row_id) != manifest["sha256_by_phase"][phase]:
        raise AssertionError("Holdout provenance drift: "+phase)
    if not np.isfinite(y).all() or np.any((y < MIN_PRICE)|(y > MAX_PRICE)):
        raise AssertionError("Invalid price in "+phase)
    if np.isinf(x).any():
        raise AssertionError("Infinite encoded feature")
    return x, y


def metrics(y, prediction):
    actual = np.asarray(y, dtype=np.float64)
    pred = np.clip(np.asarray(prediction, dtype=np.float64), MIN_PRICE, MAX_PRICE)
    if pred.shape != actual.shape or not np.isfinite(pred).all():
        raise ValueError("Invalid model predictions")
    errors = np.abs(pred-actual)
    k=max(1, math.ceil(len(errors)*.05))
    tail = np.partition(errors, len(errors)-k)[-k:]
    return {"n":len(errors),
            "cvar95_absolute_eur":float(tail.mean()),
            "mae_eur":float(errors.mean()),
            "rmse_eur":float(np.sqrt(np.mean((pred-actual)**2))),
            "median_ape_pct":float(np.median(100*errors/actual)),
            "bias_eur":float(np.mean(pred-actual))}


def dmatrix(x, y, family, manifest):
    import xgboost as xgb
    d=xgb.DMatrix(x, feature_names=manifest["feature_names"],
                  feature_types=manifest["feature_types"], nthread=2,
                  enable_categorical=True)
    if family.startswith("aft_") and family != "aft_huber":
        # Uncensored, exact positive ASK prices only. Never invent lower/upper intervals.
        d.set_float_info("label_lower_bound", np.asarray(y,dtype=np.float32))
        d.set_float_info("label_upper_bound", np.asarray(y,dtype=np.float32))
    else:
        d.set_label(np.log(np.asarray(y,dtype=np.float32)))
    return d


def validated_config(cfg):
    """Guard former Ray/PBT crashes: all dimensions and loss parameters bounded."""
    family=cfg["family"]
    if family not in ("aft_normal","aft_logistic","aft_huber","huber_log"):
        raise ValueError("Unsupported loss family")
    if not (.012 <= float(cfg["eta"]) <= .20
            and 3 <= int(cfg["max_depth"]) <= 8
            and 1 <= float(cfg["min_child_weight"]) <= 25
            and .55 <= float(cfg["subsample"]) <= 1
            and .55 <= float(cfg["colsample_bytree"]) <= 1
            and 0 <= float(cfg["reg_alpha"]) <= 10
            and .1 <= float(cfg["reg_lambda"]) <= 100
            and 0 <= float(cfg["gamma"]) <= 8
            and int(cfg["max_bin"]) in (128, 256)
            and cfg["grow_policy"] in ("depthwise","lossguide")
            and int(cfg["max_leaves"]) in (16, 32, 64)
            and .18 <= float(cfg["sigma"]) <= 1.25
            and .15 <= float(cfg["huber_delta"]) <= 1.0
            and .1 <= float(cfg["aft_weight"]) <= .8
            and 150 <= int(cfg["rounds"]) <= 650):
        raise ValueError("Out-of-range / crash-prone configuration")
    return cfg


def parameters(config, median_log):
    c = validated_config(config)
    params={
        "tree_method":"hist","device":"cpu","nthread":2,
        "eta":float(c["eta"]),"max_depth":int(c["max_depth"]),
        "min_child_weight":float(c["min_child_weight"]),
        "subsample":float(c["subsample"]),
        "colsample_bytree":float(c["colsample_bytree"]),
        "reg_alpha":float(c["reg_alpha"]),
        "reg_lambda":float(c["reg_lambda"]),
        "gamma":float(c["gamma"]),
        "max_bin":int(c["max_bin"]),
        "grow_policy":c["grow_policy"],
        "max_leaves":int(c["max_leaves"]) if c["grow_policy"]=="lossguide" else 0,
        "max_cat_to_onehot":6,"max_cat_threshold":32,
        "seed":int(c.get("seed",SEED)),"verbosity":0,
        "disable_default_eval_metric":1,
    }
    if c["family"] in ("aft_normal","aft_logistic"):
        params.update(objective="survival:aft",
                      aft_loss_distribution="normal" if c["family"]=="aft_normal" else "logistic",
                      aft_loss_distribution_scale=float(c["sigma"]))
    elif c["family"] == "huber_log":
        params.update(objective="reg:pseudohubererror",
                      huber_slope=float(c["huber_delta"]), base_score=float(median_log))
    else:
        params.update(objective="reg:squarederror",base_score=float(median_log))
    return params


def aft_huber_objective(config):
    """Positive-Hessian robust exact-event AFT + pseudo-Huber on LOG price.

    All intervals are exact, so no claim of actual censoring is made.
    The Gaussian exact-event log-likelihood differs from squared z by a
    target-constant only; sigma is bounded to avoid exploding curvature.
    """
    c=validated_config(config)
    sigma=float(c["sigma"]); delta=float(c["huber_delta"]); w=float(c["aft_weight"])
    def objective(pred, dtrain):
        r=np.clip(np.asarray(pred,dtype=np.float64)-dtrain.get_label(),-12.,12.)
        z=r/delta
        den=np.sqrt(1.+z*z)
        grad=w*r/(sigma*sigma)+(1-w)*r/den
        hess=w/(sigma*sigma)+(1-w)/(den**3)
        return np.clip(grad,-1.e4,1.e4).astype(np.float32), np.clip(hess,1.e-4,1.e4).astype(np.float32)
    return objective


def predict_model(model, dvalid, family):
    prediction=np.asarray(model.predict(dvalid),dtype=np.float64)
    if family not in ("aft_normal", "aft_logistic"):
        prediction=np.exp(np.clip(prediction,math.log(MIN_PRICE),math.log(MAX_PRICE)))
    return np.clip(prediction,MIN_PRICE,MAX_PRICE)


def run_trial(config, features, output, reporter=None):
    """Incremental XGBoost, validation-only selection and bounded objectives."""
    import xgboost as xgb
    manifest=json.loads((Path(features)/"manifest.json").read_text())
    train_x, train_y=load_data(features,"train",manifest)
    valid_x, valid_y=load_data(features,"validation",manifest)
    family=config["family"]
    cfg=validated_config(config)
    fit=dmatrix(train_x,train_y,family,manifest)
    valid=dmatrix(valid_x,valid_y,family,manifest)
    params=parameters(cfg,np.median(np.log(train_y)))
    obj=aft_huber_objective(cfg) if family=="aft_huber" else None
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    booster=None; best_score=math.inf;best=None
    iteration=0;stale=0
    for chunk in range(0,int(cfg["rounds"]),50):
        n=min(50,int(cfg["rounds"])-chunk)
        booster=xgb.train(params,fit,num_boost_round=n,xgb_model=booster,
                          obj=obj,verbose_eval=False)
        iteration+=n
        pred=predict_model(booster,valid,family)
        score=metrics(valid_y,pred)
        if score["cvar95_absolute_eur"] < best_score-0.01:
            best_score=score["cvar95_absolute_eur"]
            best={"model_family":family,"config":dict(cfg),"validation":score,
                  "iterations":iteration,"test_scored":False,
                  "censored_rows":0,"training_rows":len(train_y)}
            booster.save_model(str(output/"winner.ubj"))
            (output/"result.json").write_text(json.dumps(best,indent=2))
            stale=0
        else:
            stale+=1
        if reporter:
            reporter({"cvar95":score["cvar95_absolute_eur"],
                      "best_cvar95":best_score,"mae":score["mae_eur"],
                      "training_iteration":iteration//50,"trees":iteration})
        if stale>=3 and iteration>=150:
            break
    if best is None:
        raise RuntimeError("Training produced no valid checkpoints")
    return best


def default_config(family):
    return {"family":family,"eta":.05,"max_depth":6,"min_child_weight":5,
            "subsample":.85,"colsample_bytree":.85,"reg_alpha":.1,
            "reg_lambda":8.0,"gamma":.1,"max_bin":256,"grow_policy":"depthwise",
            "max_leaves":32,"sigma":.65,"huber_delta":.45,"aft_weight":.35,
            "rounds":400,"seed":SEED}


def baseline(features, output, family):
    name="aft_normal" if family=="aft" else "huber_log"
    result=run_trial(default_config(name),features,output)
    result["baseline"]=family
    (Path(output)/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result


def run_ray(features,output,shard,seconds=3600):
    from ray import tune, train
    import ray
    from ray.tune.schedulers import ASHAScheduler
    output=Path(output).resolve();features=Path(features).resolve()
    output.mkdir(parents=True,exist_ok=True)
    local_root=Path("/tmp")/("eu_ray_%02d"%shard)
    local_root.mkdir(parents=True,exist_ok=True)
    (local_root/"ray").mkdir(parents=True,exist_ok=True)
    os.environ["RAY_TMPDIR"]=str(local_root/"ray")
    ray.init(num_cpus=2,include_dashboard=False,ignore_reinit_error=True,
             _temp_dir=str(local_root/"ray"),log_to_driver=False)
    def trainable(config):
        context=train.get_context()
        trial_id=context.get_trial_id()
        destination=output/"trials"/str(trial_id)
        def report_step(data):
            train.report(data)
        run_trial(config,features,destination,reporter=report_step)
    search={
        "family":tune.choice(["aft_normal","aft_normal","aft_logistic",
                              "aft_huber","aft_huber","aft_huber","huber_log"]),
        "eta":tune.loguniform(.012,.20),
        "max_depth":tune.choice([3,4,5,6,7,8]),
        "min_child_weight":tune.choice([1,2,3,5,8,12,20]),
        "subsample":tune.uniform(.60,1.),
        "colsample_bytree":tune.uniform(.60,1.),
        "reg_alpha":tune.loguniform(.001,10.),
        "reg_lambda":tune.loguniform(.2,80.),
        "gamma":tune.uniform(0.,5.),
        "max_bin":tune.choice([128,256]),
        "grow_policy":tune.choice(["depthwise","lossguide"]),
        "max_leaves":tune.choice([16,32,64]),
        "sigma":tune.uniform(.18,1.0),
        "huber_delta":tune.uniform(.2,.9),
        "aft_weight":tune.uniform(.15,.75),
        "rounds":tune.choice([200,300,450,600]),
        "seed":SEED + 1000*int(shard),
    }
    # ASHA expects one incremental report per 50 trees; it can prune weak trials.
    scheduler=ASHAScheduler(metric="best_cvar95",mode="min",max_t=13,
                            grace_period=2,reduction_factor=3)
    try:
        result=tune.run(trainable,config=search,num_samples=500,
                        scheduler=scheduler,resources_per_trial={"cpu":2},
                        max_concurrent_trials=1,
                        time_budget_s=int(seconds)-120,
                        storage_path=str(local_root),name="search",
                        raise_on_failed_trial=False,verbose=1,
                        metric="best_cvar95",mode="min")
    finally:
        ray.shutdown()
    reports=[]
    for entry in (output/"trials").glob("*/result.json"):
        try:
            item=json.loads(entry.read_text())
            if (entry.parent/"winner.ubj").is_file():
                reports.append((item,entry))
        except (OSError,ValueError,KeyError):
            continue
    if not reports:
        raise RuntimeError("Ray search produced no finished model; check trial exceptions")
    reports.sort(key=lambda v:(v[0]["validation"]["cvar95_absolute_eur"],
                               v[0]["validation"]["mae_eur"]))
    winner,source=reports[0]
    shutil.copy2(source.parent/"winner.ubj",output/"winner.ubj")
    (output/"result.json").write_text(json.dumps(winner,indent=2))
    (output/"trials.json").write_text(json.dumps([x for x,_ in reports],indent=2))
    print(json.dumps({"shard":shard,"evaluated":len(reports),"winner":winner},indent=2),flush=True)
    return winner


def aggregate(root, split_dir, output):
    """Rank ALL available models by validation, test exactly ONE selected winner."""
    import xgboost as xgb
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for f in root.glob("*/result.json"):
        model=f.parent/"winner.ubj"
        if not model.is_file():
            continue
        item=json.loads(f.read_text())
        if item.get("test_scored") is not False:
            raise ValueError("Attempt to aggregate a pre-tested trial")
        reports.append((item,f,model))
    if not reports:
        raise ValueError("No completed model candidates to compare")
    reports.sort(key=lambda v:(v[0]["validation"]["cvar95_absolute_eur"],
                               v[0]["validation"]["mae_eur"]))
    manifest=json.loads((Path(split_dir)/"manifest.json").read_text())
    test_x,test_y=load_data(split_dir,"test",manifest)
    best,source,weights=reports[0]
    classifier=best["model_family"]
    test_dm=dmatrix(test_x,test_y,classifier,manifest)
    model=xgb.Booster()
    model.load_model(str(weights))
    heldout=metrics(test_y,predict_model(model,test_dm,classifier))
    leaderboard=[{"origin":f.parent.name,"family":v["model_family"],
                  "iterations":v["iterations"],"validation":v["validation"],
                  "config":v["config"],"test_scored":False}
                 for v,f,_ in reports]
    out={
        "market":"European listed ASK, not completed sale or BID",
        "dataset_rows":manifest["eligible_rows"],
        "counts":manifest["counts"],
        "deduplication_applied":False,"censored_rows":0,
        "selection":"validation CVaR95 EUR, then validation MAE EUR",
        "models_compared_on_validation":len(leaderboard),
        "leaderboard":leaderboard,
        "winning_model":{"origin":source.parent.name,"family":classifier,
                         "validation":best["validation"],
                         "test_once":heldout,
                         "test_sha256":manifest["sha256_by_phase"]["test"],
                         "config":best["config"],"iterations":best["iterations"]},
        "never_deployed":True
    }
    shutil.copy2(weights,output/"winner.ubj")
    (output/"comparison.json").write_text(json.dumps(out,indent=2))
    print(json.dumps({"winner":out["winning_model"],
                      "compared":len(leaderboard)},indent=2),flush=True)
    return out


def smoke():
    x=np.array([[0.,1.],[1.,0.],[2.,1.]],dtype=np.float32)
    y=np.array([3000.,8000.,11000.])
    for fam in ("aft_normal","aft_logistic","aft_huber","huber_log"):
        c=default_config(fam)
        validated_config(c)
        mini={"feature_names":["a","b"],"feature_types":["q","q"]}
        dm=dmatrix(x,y,fam,mini)
        p=parameters(c,np.median(np.log(y)))
        import xgboost as xgb
        m=xgb.train(p,dm,num_boost_round=2,
                    obj=aft_huber_objective(c) if fam=="aft_huber" else None)
        pred=predict_model(m,dm,fam)
        assert np.isfinite(pred).all(),fam
    bad=default_config("aft_huber");bad["aft_weight"]=1.9
    try: validated_config(bad)
    except ValueError: pass
    else: raise AssertionError("Unbounded AFT-Huber weight escaped")
    print("EUROPE_PRICE_SMOKE_OK",flush=True)


def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="action",required=True)
    a=sub.add_parser("prepare")
    a.add_argument("--input",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a=sub.add_parser("ray")
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--shard",required=True,type=int)
    a.add_argument("--seconds",type=int,default=3600)
    a=sub.add_parser("baseline")
    a.add_argument("--features",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    a.add_argument("--family",choices=["aft","huber"],required=True)
    a=sub.add_parser("aggregate")
    a.add_argument("--root",required=True,type=Path)
    a.add_argument("--splits",required=True,type=Path)
    a.add_argument("--output",required=True,type=Path)
    sub.add_parser("smoke")
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.input,args.output)
    elif args.action=="ray":
        run_ray(args.features,args.output,args.shard,args.seconds)
    elif args.action=="baseline":
        baseline(args.features,args.output,args.family)
    elif args.action=="aggregate":
        aggregate(args.root,args.splits,args.output)
    elif args.action=="smoke":
        smoke()


if __name__=="__main__":
    main()
