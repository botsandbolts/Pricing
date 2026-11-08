"""Original holdout-gated BOHB champions for the main Ask/Bid calculator.

Asking consumes the ONE shared pinned BGE encoder (ONNX INT8 if and only if
an independently verified encoder release is installed), and a *native*
PyTorch DeepCross head. Bid consumes no BGE/text encoder; it uses only
the unchanged wholesale preprocessing, a native StructuredBid DeepFM and
the immutable Lemon XGBoost reference. Fail closed on provenance errors.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

FOLDERS={"ask":"models/neural-bohb-ask","bid":"models/neural-bohb-bid"}
WINNERS={"ask":36,"bid":116}
FILES=("weights.pt","manifest.json","verification.json")


def file_hash(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def available(root,market):
    if market not in FOLDERS:
        raise ValueError("Unknown BOHB market")
    folder=Path(root)/FOLDERS[market]
    if not all((folder/f).is_file() for f in FILES):
        return False
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        if not (
            m.get("market")==market and m.get("trial")==WINNERS[market]
            and m.get("source_study")=="overnight-wild-bohb"
            and m.get("status")=="release_candidate_verified"
            and m.get("native_pytorch_head") is True
            and m.get("test_selected") is False
            and m.get("original_test_rows_scored")==(
                31682 if market=="ask" else 83603)
            and v.get("status")=="passed"
            and v.get("verified") is True
            and v.get("complete_validation_parity") is True
            and v.get("full_original_test_scored") is True
            and v.get("serving_prediction_parity") is True
            and v.get("candidate_test_cvar95_improves_incumbent") is True
            and v.get("weight_sha256")==m.get("checkpoint_sha256")
            and file_hash(folder/"weights.pt")==m.get("checkpoint_sha256")
            and m.get("bge_in_bid_model") is False
        ):
            return False
        if market=="ask":
            from pricing.neural_ask_embedding_live import available as original
        else:
            from pricing.neural_bid_blend_live import available as original
        return original(root)
    except (OSError,ValueError,KeyError,TypeError):
        return False


class BOHBAsking:
    """Same live BGE/preprocessor. New, exactly archived, compact Torch head."""
    def __init__(self,root,*,_parity_test=False):
        import torch
        from scripts.overnight_bohb_neural import build_ask
        from pricing.neural_ask_embedding_live import load as original_load
        if not _parity_test and not available(root,"ask"):
            raise FileNotFoundError("Ask BOHB release has not passed full parity")
        folder=Path(root)/FOLDERS["ask"]
        self.manifest=json.loads((folder/"manifest.json").read_text())
        self.original=original_load(root)
        pack=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        cfg=self.manifest["config"]
        if (pack.get("search_group")!="ask_refined"
            or pack.get("trial")!=self.manifest["origin_trial"]
            or pack.get("trial_configuration")!=cfg):
            raise ValueError("Ask BOHB checkpoint configuration/origin mismatch")
        self.model=build_ask(self.original.cards,384,"ask_structure",cfg["structure"])
        self.model.load_state_dict(pack["state_dict"],strict=True)
        self.model.eval()

    def prepare_text(self,description):
        return self.original.prepare_text(description)

    def predict_prepared(self,fields,prepared):
        import torch
        n,c,t=self.original.encode_inputs(fields,prepared)
        with torch.inference_mode():
            raw=self.model(torch.from_numpy(n),torch.from_numpy(c.astype(np.int64)),
                           torch.from_numpy(t))[:,0].numpy()[0]
        price=float(np.exp(np.clip(raw,np.log(100.),np.log(1e6))))
        if not np.isfinite(price) or price<=0:
            raise ValueError("BOHB Ask model produced an invalid price")
        return {"market":"ask","price":price,
                "model":"BGE-256 + BOHB DeepCross Ask #36",
                "historical_anchor":"2021-05-01",
                "text_price_heads":0,
                "price_predictions_from":"native BOHB DeepCross, no Ridge",
                "current_market_accuracy_validated":False}

    def predict(self,description,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))


class BOHBWholesale:
    """Original matching VIN-aware features/XGB, structured DeepFM, no BGE."""
    def __init__(self,root,*,_parity_test=False):
        import torch
        from scripts.overnight_bohb_neural import StructuredBid
        from pricing.neural_bid_blend_live import load as original_load
        if not _parity_test and not available(root,"bid"):
            raise FileNotFoundError("Bid BOHB release has not passed full parity")
        folder=Path(root)/FOLDERS["bid"]
        self.manifest=json.loads((folder/"manifest.json").read_text())
        self.original=original_load(root)
        pack=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        cfg=self.manifest["config"]
        if (pack.get("search_group")!="bid_refined"
            or pack.get("trial")!=self.manifest["origin_trial"]
            or pack.get("trial_configuration")!=cfg):
            raise ValueError("Bid BOHB checkpoint configuration/origin mismatch")
        state=pack["state_dict"]
        self.model=StructuredBid(self.original.cards,
                                  float(state["core.fm.base"]),cfg["structure"])
        self.model.load_state_dict(state,strict=True)
        self.model.eval()
        self.alpha=float(cfg["hyperparameters"]["blend_alpha"])
        if self.alpha!=float(self.manifest["neural_weight"]) or not .5<=self.alpha<=1.:
            raise ValueError("Registered wholesale blend is inconsistent")

    def predict(self,fields):
        import torch
        from pricing.neural_bid_blend_live import (
            CATS,_norm,checked_price,_auction_date)
        from pricing.hierarchy_sparse_live import base_predict
        n,c,raw,unknown=self.original.features(fields)
        with torch.inference_mode():
            log=float(self.model(torch.from_numpy(n),
                                 torch.from_numpy(c))[:1].numpy()[0])
        deepfm=checked_price(float(np.exp(np.clip(log,np.log(100.),np.log(1e6)))),
                              "BOHB structured DeepFM")
        original={**{k:_norm(fields.get(k)) for k in CATS[:8]},
                  "odometer":float(fields["odometer"]),
                  "auction_grade":float(raw[-1]) if np.isfinite(raw[-1]) else None}
        xgb,missing=base_predict(self.original.xgb,self.original.xgb_schema,None,
                                  original,float(raw[1]),"bid")
        xgb=checked_price(xgb,"Original Lemon XGBoost")
        price=checked_price(self.alpha*deepfm+(1.-self.alpha)*xgb,
                            "BOHB wholesale ensemble")
        return {"market":"bid","model":"BOHB DeepFM #116 + Lemon XGBoost",
                "price":price,"deepfm_price":deepfm,
                "xgboost_lemon_price":xgb,"neural_weight":self.alpha,
                "historical_price_basis":"2015-index-normalized USD",
                "original_auction_date":_auction_date(fields["auction_date"]).isoformat(),
                "is_completed_auction_sale_estimate":True,
                "live_2026_validated":False,"verified":True,
                "unrecognized_categories":sorted(set(unknown+missing))}


def _load(root,market):
    import streamlit as st
    folder=Path(root)/FOLDERS[market]
    fp=tuple((n,(folder/n).stat().st_size,(folder/n).stat().st_mtime_ns)
             for n in FILES)
    # Native BGE releases can switch to parity-certified INT8 without cloning
    # the frozen encoder. Bid never loads BGE.
    @st.cache_resource(max_entries=2,show_spinner=False)
    def factory(folder_root,side,fingerprint):
        return BOHBAsking(folder_root) if side=="ask" else BOHBWholesale(folder_root)
    return factory(str(root),market,fp)


def load_ask(root):
    if not available(root,"ask"):
        raise FileNotFoundError("BOHB Ask weights unverified")
    return _load(root,"ask")


def load_bid(root):
    if not available(root,"bid"):
        raise FileNotFoundError("BOHB Bid weights unverified")
    return _load(root,"bid")
