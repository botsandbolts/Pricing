"""Exact European ASK-price inference for the sealed-test Gaussian AFT winner.

The training-only vocabulary (packaged alongside the model) MUST be used.
Never infer categorical ids from the user-selected label/order or refit it.
Price is an advertised ASK estimate in euros, not a transaction/bid result.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np

NUMERIC=("reg_year","mileage_km","power_hp","reg_month","listing_year",
         "listing_month","age_at_listing","km_per_year","log_km","doors",
         "seats","images_count","accident_free")
CATEGORICAL=("seller_country","make","model","model_group","version",
             "seller_city","fuel","transmission","color","source_site",
             "body_type","drive_type")
FEATURES=list(NUMERIC+CATEGORICAL)
FEATURE_TYPES=["q"]*len(NUMERIC)+["c"]*len(CATEGORICAL)
ROOT=Path(__file__).resolve().parents[1]/"models"/"europe-ask"


class EuropeanModelUnavailable(RuntimeError):
    pass


def safe_text(value):
    return str(value if value is not None else "__missing__").strip().lower() or "__missing__"


def validate_vehicle(item):
    make=str(item.get("make") or "").strip()
    model=str(item.get("model") or "").strip()
    country=str(item.get("seller_country") or "").strip()
    if not make or not model or not country:
        raise ValueError("Country, make and model are required.")
    year=int(item.get("reg_year",0))
    if not 1950 <= year <= 2026:
        raise ValueError("Registration year must be between 1950 and 2026.")
    km=float(item.get("mileage_km",-1))
    if not math.isfinite(km) or not 0<=km<=2_000_000:
        raise ValueError("Mileage must be between 0 and 2,000,000 km.")
    hp=item.get("power_hp")
    if hp not in ("",None):
        hp=float(hp)
        if not math.isfinite(hp) or not 20<=hp<=1200:
            raise ValueError("Power must be between 20 and 1,200 hp.")
    return item


def encode_vehicle(item,schema):
    """Exact feature order, normalized categories and trained vocabulary ids."""
    validate_vehicle(item)
    when=str(item.get("first_seen_at") or schema["snapshot_date"])
    listing_year=int(when[:4])
    listing_month=int(when[5:7])
    year=int(item["reg_year"])
    km=float(item["mileage_km"])
    hp=item.get("power_hp")
    hp=float(hp) if hp not in ("",None) else float("nan")
    raw_month=item.get("reg_month")
    regmonth=float("nan") if raw_month in (None,"") else int(raw_month)
    if (math.isfinite(regmonth) and not 1<=regmonth<=12) or not 1<=listing_month<=12:
        raise ValueError("Registration/listing month must be 1 through 12.")
    age=listing_year-year
    numeric={
        "reg_year":float(year),"mileage_km":km,"power_hp":hp,
        "reg_month":float(regmonth),"listing_year":float(listing_year),
        "listing_month":float(listing_month),
        "age_at_listing":float(age) if -2<=age<=90 else float("nan"),
        "km_per_year":km/max(.5,age) if .5<=age<=90 else float("nan"),
        "log_km":math.log1p(km),
        "doors":_optional_number(item.get("doors")),
        "seats":_optional_number(item.get("seats")),
        "images_count":_optional_number(item.get("images_count")),
        "accident_free":_optional_number(item.get("accident_free")),
    }
    row=[numeric[key] for key in NUMERIC]
    for key in CATEGORICAL:
        vocab=schema["category_vocabulary"][key]
        raw=safe_text(item.get(key))
        # Fallback to NaN is the exact same unknown-category behavior as training.
        row.append(float(vocab[raw]) if raw in vocab else float("nan"))
    if schema.get("feature_names")!=FEATURES or schema.get("feature_types")!=FEATURE_TYPES:
        raise EuropeanModelUnavailable("Packaged European feature schema is incompatible.")
    result=np.asarray([row],dtype=np.float32)
    if np.isinf(result).any():
        raise ValueError("Out-of-range feature generated")
    return result


def _optional_number(value):
    if value in ("",None):
        return float("nan")
    number=float(value)
    return number if math.isfinite(number) else float("nan")


@lru_cache(maxsize=1)
def load_bundle(path=None):
    import xgboost as xgb
    root=Path(path) if path is not None else ROOT
    model_path=root/"winner.ubj"
    schema_path=root/"schema.json"
    comparison_path=root/"comparison.json"
    if not all(p.exists() for p in (model_path,schema_path,comparison_path)):
        raise EuropeanModelUnavailable(
            "European Gaussian AFT research winner has not passed packaging/release checks.")
    schema=json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("trained_model")!="aft_normal" or schema.get("train_only_vocabulary") is not True:
        raise EuropeanModelUnavailable("Unsupported model bundle or unverified vocabulary.")
    digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if schema.get("model_sha256")!=digest:
        raise EuropeanModelUnavailable("Model checksum mismatch; European pricing disabled.")
    model=xgb.Booster()
    model.load_model(str(model_path))
    comparison=json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison["winning_model"]["origin"]!="eu-ask-candidate-ray-8":
        raise EuropeanModelUnavailable("Bundled model is not the validated rank-1 winner.")
    return model,schema,comparison


def _dmatrix(rows,schema):
    import xgboost as xgb
    return xgb.DMatrix(rows,feature_names=schema["feature_names"],
                       feature_types=schema["feature_types"],
                       enable_categorical=True,nthread=2)


def estimate(vehicle,bundle=None):
    model,schema,_=bundle if bundle is not None else load_bundle()
    encoded=encode_vehicle(vehicle,schema)
    prediction=float(model.predict(_dmatrix(encoded,schema))[0])
    if not math.isfinite(prediction) or prediction<=0:
        raise EuropeanModelUnavailable("The model produced an invalid estimate.")
    return float(np.clip(prediction,100.,1_000_000.))


def batch_estimate(vehicles,bundle=None):
    model,schema,_=bundle if bundle is not None else load_bundle()
    inputs=np.concatenate([encode_vehicle(v,schema) for v in vehicles],axis=0)
    predictions=model.predict(_dmatrix(inputs,schema))
    return np.clip(np.asarray(predictions,dtype=float),100.,1_000_000.)


def explain(vehicle,bundle=None):
    """SHAP feature contributions are on the XGBoost AFT margin (log-price) scale."""
    model,schema,_=bundle if bundle is not None else load_bundle()
    matrix=_dmatrix(encode_vehicle(vehicle,schema),schema)
    try:
        values=np.asarray(model.predict(matrix,pred_contribs=True))[0]
        if len(values)==len(FEATURES)+1 and np.all(np.isfinite(values)):
            return sorted(zip(FEATURES,(float(v) for v in values[:-1])),
                          key=lambda pair:abs(pair[1]),reverse=True)
    except (ValueError,TypeError,RuntimeError):
        pass
    return []


def available(path=None):
    p=Path(path) if path is not None else ROOT
    return all((p/name).is_file() for name in ("winner.ubj","schema.json","comparison.json"))


# European production dispatch: retain the frozen feature encoder and market
# statistics, but route European ASK valuation calls to the validated November 2026 HOFM3 checkpoint.
# No XGBoost fallback: an absent/corrupt neural release must fail closed.
from pricing.europe_hofm3 import (  # noqa: E402,F401
    load_bundle, estimate, batch_estimate, explain, available,
)
