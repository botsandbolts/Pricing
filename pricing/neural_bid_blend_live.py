"""Fail-closed DeepFM32-full(75%)+original Lemon XGBoost(25%) wholesale lab.

The historical target is OBSERVED auction sale amount, in 2015-normalized
USD, not a submitted dealer offer or a November 2026 market forecast.
"""
from datetime import date,datetime
from pathlib import Path
import json
import numpy as np

FOLDER="models/neural-experimental-bid"
FILES=("weights.pt","xgboost-lemon.ubj","xgboost-category-schema.json",
       "preprocessing.json","manifest.json","verification.json")
CATS=("manufacturer","model","type","transmission","state","fuel",
      "drive","title_status","cylinders","paint_color")


def available(root):
    folder=Path(root)/FOLDER
    if not all((folder/name).is_file() for name in FILES):
        return False
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        return bool(m.get("model")=="deepfm32_full_xgb_lemon_75_25"
                    and m.get("weights_frozen") is True
                    and v.get("status")=="passed"
                    and v.get("verified") is True
                    and v.get("original_holdout_rows_checked",0)>=40
                    and v.get("deepfm_archived_predictions_match") is True
                    and v.get("xgboost_archived_predictions_match") is True
                    and v.get("no_new_training") is True)
    except (OSError,ValueError,TypeError,KeyError):
        return False


def _auction_date(value):
    if isinstance(value,str):
        try:value=date.fromisoformat(value[:10])
        except ValueError as error:
            raise ValueError("Use a 2014–2015 historical auction date.") from error
    if isinstance(value,datetime):
        value=value.date()
    if not isinstance(value,date) or not date(2014,1,1)<=value<=date(2015,12,31):
        raise ValueError("Original wholesale auction study supports only 2014–2015.")
    return value


def _norm(value):
    if value is None:return "unknown"
    x=str(value).strip().lower()
    return x if x and x not in ("none","null","nan") else "unknown"


def raw_numeric(fields):
    try:
        year=int(fields["year"])
        miles=float(fields["odometer"])
    except (KeyError,ValueError,TypeError) as e:
        raise ValueError("Model year and odometer miles are required.") from e
    at=_auction_date(fields["auction_date"])
    if year<1886 or year>at.year+1:
        raise ValueError("Model year is outside the original historical auction period.")
    if not np.isfinite(miles) or not 0<=miles<=2e6:
        raise ValueError("Odometer miles must be finite, nonnegative and <= 2 million.")
    age=max(at.year+(at.timetuple().tm_yday-1)/365.25-year,0.)
    grade=fields.get("auction_grade")
    try:g=float(grade)
    except (TypeError,ValueError):g=np.nan
    if not 1<=g<=5:g=np.nan
    return np.array([year,age,np.log1p(miles),
                     np.log1p(miles/max(age,1.)),g],dtype=np.float64)


def checked_price(price,label):
    price=float(price)
    if not np.isfinite(price) or not 100<=price<=1e6:
        raise ValueError(label+" did not produce a trustworthy historical price.")
    return price


class VerifiedWholesaleBlend:
    def __init__(self,root):
        import torch
        import xgboost as xgb
        from scripts.bid_fm_scale_train import ScaledAuctionFM
        from pricing.suite_storage import read_model_file
        root=Path(root)
        if not available(root):
            raise FileNotFoundError("No original-holdout-verified wholesale model bundle")
        folder=root/FOLDER
        self.state=json.loads((folder/"preprocessing.json").read_text())
        self.manifest=json.loads((folder/"manifest.json").read_text())
        s=self.state
        self.median=np.asarray(s["numeric_median"],dtype=float)
        self.scale=np.asarray(s["numeric_scale"],dtype=float)
        self.maps=s["category_maps"]
        self.cards=list(s["cardinalities"])
        if (tuple(s["categorical_features"])!=CATS or
            self.median.shape!=(5,) or self.scale.shape!=(5,) or
            not np.isfinite(self.median).all() or
            not np.isfinite(self.scale).all() or (self.scale<=0).any()
            or set(self.maps)!=set(CATS) or len(self.cards)!=10):
            raise ValueError("Frozen DeepFM input dictionary geometry changed")
        pack=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        if (pack.get("trial"),pack.get("family"),pack.get("rank"),
            pack.get("cohort"),pack.get("cards"),pack.get("seed"))!=(
                "deepfm32_full","deepfm",32,"full",self.cards,20261106):
            raise ValueError("Unexpected neural checkpoint identity")
        self.model=ScaledAuctionFM(self.cards,"deepfm",32,
                                   pack["initial_log_price"])
        self.model.load_state_dict(pack["state_dict"],strict=True)
        self.model.eval()
        self.xgb=xgb.Booster()
        self.xgb.load_model(bytearray(read_model_file(folder/"xgboost-lemon.ubj")))
        self.xgb.set_param({"nthread":2})
        self.xgb_schema=json.loads((folder/"xgboost-category-schema.json").read_text())
        if set(self.xgb_schema)!=set(CATS[:8]):
            raise ValueError("Original XGBoost has different pre-sale categories")

    def features(self,fields):
        raw=raw_numeric(fields)
        n=np.clip((np.where(np.isfinite(raw),raw,self.median)-self.median)
                  /self.scale,-8,8).astype(np.float32)
        categories=np.asarray([self.maps[c].get(_norm(fields.get(c)),1)
                      for c in CATS],dtype=np.int64)
        if any(not (1<=code<card) for code,card in zip(categories,self.cards)):
            raise ValueError("Unknown original embedding category code")
        unknown=[c for c,v in zip(CATS,categories) if v==1]
        return n.reshape(1,5),categories.reshape(1,10),raw,unknown

    def predict(self,fields):
        import torch
        from pricing.hierarchy_sparse_live import base_predict
        numeric,categories,raw,unknown=self.features(fields)
        with torch.inference_mode():
            predicted=self.model(torch.from_numpy(numeric),
                                 torch.from_numpy(categories))
        deepfm=checked_price(np.exp(np.clip(
                    float(predicted[0]),np.log(100),np.log(1e6))),"DeepFM")
        original={**{k:_norm(fields.get(k)) for k in CATS[:8]},
                  "odometer":float(fields["odometer"]),
                  "auction_grade":float(raw[-1]) if np.isfinite(raw[-1]) else None}
        xgb_price,missing=base_predict(self.xgb,self.xgb_schema,None,
                                       original,float(raw[1]),"bid")
        xgb_price=checked_price(xgb_price,"Lemon XGBoost")
        combined=checked_price(.75*deepfm+.25*xgb_price,"Experimental bid blend")
        return {"market":"bid","model":"deepfm32_full_xgb_lemon_75_25",
                "price":combined,"deepfm_price":deepfm,
                "xgboost_lemon_price":xgb_price,
                "historical_price_basis":"2015-index-normalized USD",
                "original_auction_date":_auction_date(fields["auction_date"]).isoformat(),
                "is_completed_auction_sale_estimate":True,
                "live_2026_validated":False,"verified":True,
                "unrecognized_categories":sorted(set(unknown+missing))}


def _cache(root,signature):
    import streamlit as st
    @st.cache_resource(max_entries=2,show_spinner=False)
    def make(root_,signature_):
        return VerifiedWholesaleBlend(root_)
    return make(root,signature)


def load(root):
    folder=Path(root)/FOLDER
    signature=tuple((name,(folder/name).stat().st_mtime_ns,
                     (folder/name).stat().st_size) for name in FILES)
    return _cache(str(root),signature)
