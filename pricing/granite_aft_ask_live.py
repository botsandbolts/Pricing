"""Release-verified Granite R2 97M INT8 ONNX + full-cohort DCNv2 AFT Ask.

The ONLY active U.S. Ask engine. Uses original 2021 features, 2021 model-age
and TRAIN-derived make/model mileage baselines. This is a native FP32 price
head and a frozen IBM QInt8 ONNX text encoder, not a BGE fallback.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

FOLDER=Path("models/us-ask-granite-aft-20261111")
SOURCE_RUN=43034629386
SHARD=0
COHORT_SHA="746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12"
EMBEDDING_SHA="7f2be5c0e2199ca8ca23c480140b6f8b6707b119594efb9f3278c415e7b8e5f2"
ONNX_SHA="a6022dd8220ea6f6595562a1328ee216f4a94faa55362f2f4747c80f1e78772e"
NUMS=("year","age_2021","log_miles","miles_per_year",
      "log_description_chars","log_description_words","description_punctuation")
CATS=("manufacturer","model","type","transmission","state","fuel",
      "drive","title_status","cylinders","paint_color","condition")
NEW_NUMS=("log_expected_annual_miles_make_model",
          "log_odometer_to_model_expected_2021",
          "signed_excess_mileage_vs_2021_expected")
METRICS={"CVaR95_absolute_USD":18448.441254586884,
         "MAE_USD":2805.314098262269,
         "RMSE_USD":5766.246111146663}

def _sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def manifest(root):
    folder=Path(root)/FOLDER
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        p=json.loads((folder/"preprocessing.json").read_text())
        mm=json.loads((folder/"mileage_profiles.json").read_text())
        if not (
            m.get("status")=="verified_for_live_historical_ask"
            and m.get("model")=="DCNv2 + AFT"
            and m.get("source_run_id")==SOURCE_RUN
            and m.get("source_shard")==SHARD
            and m.get("cohort_sha256")==COHORT_SHA
            and m.get("embedding_sha256")==EMBEDDING_SHA
            and m.get("encoder_onnx_sha256")==ONNX_SHA
            and m.get("original_training_rows")==147777
            and m.get("train_fit_rows")==133000
            and m.get("train_only_stopper_rows")==14777
            and m.get("validation",{}).get("rows")==31668
            and m.get("test_scored") is False
            and m.get("training_objective")=="aft_with_train_only_tail_penalty"
            and m["configuration"].get("family")=="dcnv2"
            and m["configuration"].get("loss")=="aft"
            and m.get("best_epoch")==76
            and m["checkpoint_sha256"]==_sha(folder/"weights.pt")
            and m["preprocessing_sha256"]==_sha(folder/"preprocessing.json")
            and m["mileage_profile_sha256"]==_sha(folder/"mileage_profiles.json")
            and tuple(p["numeric_names"])==NUMS
            and tuple(p["categorical_names"])==CATS
            and p["cardinalities"]==m["cardinalities"]
            and mm.get("trained_on_original_train_only") is True
            and mm.get("original_train_rows")==147777
            and tuple(mm["numeric_order"])==NEW_NUMS
            and v.get("status")=="passed"
            and v.get("original_validation_prediction_parity_passed") is True
            and v.get("live_feature_parity_passed") is True
            and v.get("original_test_not_scored") is True
            and v.get("source_run_id")==SOURCE_RUN
            and all(math.isclose(float(m["validation"][k]),expected,abs_tol=.002)
                    for k,expected in METRICS.items())
            and (folder/"weights.pt").is_file()
        ):
            return None
        return m
    except (OSError,KeyError,ValueError,TypeError,OverflowError):
        return None

def available(root):
    return manifest(root) is not None

def raw_numeric(fields,raw):
    try:
        year=int(fields["year"]);miles=float(fields["odometer"])
    except (ValueError,TypeError,KeyError) as exc:
        raise ValueError("Model year and original odometer mileage are required") from exc
    if not 1900<=year<=2021 or not np.isfinite(miles) or not 0<=miles<=2e6:
        raise ValueError("Granite historical Ask supports model years <=2021 and valid miles")
    age=max(2021-year,0)
    return np.asarray([
        year,age,np.log1p(miles),miles/max(age,1),
        np.log1p(len(raw)),np.log1p(len(re.findall(r"\S+",raw))),
        np.log1p(len(re.findall(r"[!?]",raw)))
    ],dtype=np.float64)

def mileage_extra(fields,profiles):
    year=int(fields["year"])
    miles=float(fields["odometer"])
    maker=str(fields.get("manufacturer") or "unknown").strip().lower()
    model=str(fields.get("model") or "unknown").strip().lower()
    rate=profiles["by_make_model"].get(maker,{}).get(model,
         profiles["by_make"].get(maker,profiles["pooled"]))
    rate=float(rate)
    if not (1000<=rate<=50000 and math.isfinite(rate)):
        raise ValueError("Invalid train-only annual mileage profile")
    age=max(2021-year,1.)
    expected=age*rate
    v=np.asarray([
        np.log1p(rate),
        np.log1p(max(miles,0)/max(expected,1000)),
        np.sign(miles-expected)*np.log1p(abs(miles-expected)/10000),
    ],dtype=np.float64)
    center=np.asarray(profiles["median"],dtype=np.float64)
    scale=np.asarray(profiles["iqr"],dtype=np.float64)
    if center.shape!=(3,) or scale.shape!=(3,) or np.any(scale<=0):
        raise ValueError("Invalid train-only mileage normalization")
    return np.clip((v-center)/scale,-8,8).astype(np.float32)

class GraniteAFTAsking:
    def __init__(self,root,*,_parity_test=False):
        import torch
        from scripts.granite_ask_full_ray_4h import AskNetwork,bound_config
        from pricing.granite_ask_live import _GraniteInt8Encoder
        root=Path(root)
        self.folder=root/FOLDER
        self.manifest=json.loads((self.folder/"manifest.json").read_text())
        if not _parity_test and not available(root):
            raise FileNotFoundError("Granite AFT checkpoint or release parity verification missing")
        preprocessing=json.loads((self.folder/"preprocessing.json").read_text())
        self.profiles=json.loads((self.folder/"mileage_profiles.json").read_text())
        self.median=np.asarray(preprocessing["numeric_median"],dtype=np.float64)
        self.scale=np.asarray(preprocessing["numeric_scale"],dtype=np.float64)
        self.maps=preprocessing["category_maps"]
        self.cards=list(preprocessing["cardinalities"])
        if (self.median.shape!=(7,) or self.scale.shape!=(7,) or np.any(self.scale<=0)
                or tuple(preprocessing["numeric_names"])!=NUMS
                or tuple(preprocessing["categorical_names"])!=CATS
                or not np.isfinite(self.median).all()
                or not np.isfinite(self.scale).all()):
            raise ValueError("Historic Ask preprocessing differs")
        ckpt=torch.load(self.folder/"weights.pt",map_location="cpu",weights_only=True)
        config=bound_config(dict(ckpt["config"]))
        if (config!=self.manifest["configuration"]
                or ckpt["source_sha256"]!=COHORT_SHA
                or ckpt["embedding_sha256"]!=EMBEDDING_SHA
                or ckpt["cards"]!=self.cards
                or config["family"]!="dcnv2" or config["loss"]!="aft"
                or ckpt["original_test_scored"] is not False):
            raise ValueError("Granite checkpoint does not match original Ray AFT winner")
        self.price_network=AskNetwork(self.cards,float(ckpt["base"]),config)
        self.price_network.load_state_dict(ckpt["state_dict"],strict=True)
        self.price_network.eval()
        self.encoder=_GraniteInt8Encoder() if not _parity_test else None
        self.encoder_backend="IBM Granite R2 97M · pinned QInt8 ONNX CPU"

    def prepare_text(self,description):
        from scripts.common import clean
        raw=str(description or "")[:8000]
        return {"raw":raw,"embedding":self.encoder.encode(clean(raw))}

    def encode_inputs(self,fields,prepared):
        original=raw_numeric(fields,prepared["raw"])
        numeric=np.clip((original-self.median)/self.scale,-8,8).astype(np.float32)
        extra=mileage_extra(fields,self.profiles)
        category=np.asarray([self.maps[k].get(
            str(fields.get(k) or "unknown").strip().lower(),1) for k in CATS],
            dtype=np.int64)
        if any(not (1<=int(v)<card) for v,card in zip(category,self.cards)):
            raise ValueError("Unknown category index in historic Ask inference")
        vec=np.asarray(prepared["embedding"],dtype=np.float32)
        if vec.shape!=(1,384) or not np.isfinite(vec).all():
            raise ValueError("Granite INT8 vector must be a finite 384-dimensional embedding")
        return np.concatenate((numeric,extra))[None,:],category[None,:],vec

    def predict_prepared(self,fields,prepared):
        import torch
        numeric,categories,embedding=self.encode_inputs(fields,prepared)
        with torch.inference_mode():
            mu=float(self.price_network(torch.from_numpy(numeric),
                torch.from_numpy(categories),
                torch.from_numpy(embedding))[0,0].item())
        price=float(np.exp(np.clip(mu,np.log(100),np.log(1000000))))
        if not np.isfinite(price) or price<=0:
            raise ValueError("Granite DCNv2 AFT returned a nonfinite price")
        return {"market":"ask","price":price,
            "model":"Granite R2 97M · INT8 ONNX · DCNv2 AFT",
            "historical_anchor":"2021-05-01",
            "historical_2021_validation":METRICS,
            "text_price_heads":0,"full_training_cohort":True,
            "native_head_fp32":True,"current_market_accuracy_validated":False}

    def predict(self,description,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))

def load(root):
    import streamlit as st
    folder=Path(root)/FOLDER
    if not available(root):
        raise FileNotFoundError("Verified Granite-only AFT Ask release is unavailable")
    signature=tuple((name,(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns)
                   for name in ("weights.pt","preprocessing.json",
                                "mileage_profiles.json","manifest.json",
                                "verification.json"))
    @st.cache_resource(max_entries=1,show_spinner=False)
    def _cache(path,_signature):
        return GraniteAFTAsking(path)
    return _cache(str(root),signature)
