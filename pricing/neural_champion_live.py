"""Validation-selected CVaR95 challenger inference, isolated to Experimental tab.

The original parity-verified BGE encoder, train-only preprocessing and Lemon
XGBoost remain shared. No extra encoder copy, ONNX, pickle, fake prices,
automatic 2026 extrapolation, or change to the primary calculator.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ASK_FOLDER="models/neural-experimental-ask-cvar95"
BID_FOLDER="models/neural-experimental-bid-cvar95"
NAMES=("weights.pt","manifest.json","verification.json")


def _sha(path):
    with open(path,"rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def _available(root,market):
    folder=Path(root)/(ASK_FOLDER if market=="ask" else BID_FOLDER)
    if not all((folder/name).is_file() for name in NAMES):
        return False
    try:
        manifest=json.loads((folder/"manifest.json").read_text())
        check=json.loads((folder/"verification.json").read_text())
        return bool(
            manifest.get("market")==market
            and manifest.get("status")=="release_candidate_verified"
            and manifest.get("native_pytorch_only") is True
            and manifest.get("original_test_scored_after_validation_selection") is True
            and manifest.get("production_main_calculator_unchanged") is True
            and check.get("status")=="passed"
            and check.get("verified") is True
            and check.get("complete_original_validation_parity") is True
            and check.get("original_test_rows_scored")==(
                31682 if market=="ask" else 83603)
            and check.get("shared_preprocessor_parity") is True
            and check.get("live_prediction_checked") is True
            and check.get("weight_sha256")==manifest.get("checkpoint_sha256")
            and _sha(folder/"weights.pt")==manifest.get("checkpoint_sha256")
        )
    except (OSError,ValueError,KeyError,TypeError):
        return False


def ask_available(root):
    from pricing.neural_ask_embedding_live import available as original
    return original(root) and _available(root,"ask")


def bid_available(root):
    from pricing.neural_bid_blend_live import available as original
    return original(root) and _available(root,"bid")


def _read(root,market):
    folder=Path(root)/(ASK_FOLDER if market=="ask" else BID_FOLDER)
    return folder,json.loads((folder/"manifest.json").read_text())


class CVaR95Asking:
    """Reuse ONE original BGE encoder; only the cheap neural pricing head differs."""
    def __init__(self,root,*,_parity_test=False):
        import torch
        from scripts.ask25_joint_search import ResearchAskingNetwork,trial_config
        from pricing.neural_ask_embedding_live import load as original_load
        if not _parity_test and not ask_available(root):
            raise FileNotFoundError("CVaR95 Ask winner not installed with full parity")
        folder,self.manifest=_read(root,"ask")
        if (self.manifest.get("source_study"),self.manifest.get("trial"))!=(
                "ask25-joint",15):
            raise ValueError("Unexpected Ask CVaR winner identity")
        self.original=original_load(root)
        pack=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        expected=trial_config(15)
        if (pack.get("trial")!=15 or pack.get("parameters")!=expected
            or self.manifest.get("config")!=expected
            or pack.get("provenance",{}).get("cohort_sha256")!=
                 self.manifest.get("cohort_sha256")):
            raise ValueError("Ask checkpoint did not match validation-selected trial")
        self.model=ResearchAskingNetwork(
            self.original.cards,384,expected)
        self.model.load_state_dict(pack["state_dict"],strict=True)
        self.model.eval()

    def prepare_text(self,description):
        return self.original.prepare_text(description)

    def predict_prepared(self,fields,prepared):
        import torch
        numeric,category,text=self.original.encode_inputs(fields,prepared)
        with torch.inference_mode():
            log=self.model(torch.from_numpy(numeric),
                torch.from_numpy(category.astype(np.int64)),
                torch.from_numpy(text))[:,0].numpy()[0]
        value=float(np.exp(np.clip(log,np.log(100.),np.log(1e6))))
        if not np.isfinite(value) or value<=0:
            raise ValueError("ResMLP produced an invalid asking amount")
        return {"market":"ask","price":value,
            "model":"BGE-256 + ResMLP · CVaR95 winner #15",
            "historical_anchor":"2021-05-01",
            "text_price_heads":0,
            "price_predictions_from":"complete native ResMLP, not text-price Ridge",
            "current_market_accuracy_validated":False}

    def predict(self,description,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))


class CVaR95Wholesale:
    """Reuse the ORIGINAL Lemon XGBoost/feature dictionary; replace only FM weights."""
    def __init__(self,root,*,_parity_test=False):
        import torch
        from pricing.neural_bid_blend_live import load as original_load
        from scripts.bid_fm_scale_train import ScaledAuctionFM
        from scripts.neural_200_sweep import spec
        if not _parity_test and not bid_available(root):
            raise FileNotFoundError("CVaR95 wholesale winner not installed with full parity")
        folder,self.manifest=_read(root,"bid")
        if (self.manifest.get("source_study"),self.manifest.get("trial"))!=(
                "bid_hparams",18):
            raise ValueError("Unexpected Bid CVaR winner identity")
        self.original=original_load(root)
        pack=torch.load(folder/"weights.pt",map_location="cpu",weights_only=True)
        if (pack.get("search_group"),pack.get("trial"),
            pack.get("trial_configuration"))!=("bid_hparams",18,spec("bid_hparams",18)):
            raise ValueError("Bid checkpoint is not the validated DeepFM16 trial")
        if self.manifest.get("config")!=pack.get("trial_configuration"):
            raise ValueError("Bid checkpoint training settings mismatched")
        alpha=float(self.manifest["neural_weight"])
        if not .5<=alpha<=1.:
            raise ValueError("Unverified Bid ensemble blend weight")
        self.alpha=alpha
        self.model=ScaledAuctionFM(self.original.cards,"deepfm",16,
             initial_log_price=float(pack["state_dict"]["fm.base"]))
        self.model.load_state_dict(pack["state_dict"],strict=True)
        self.model.eval()

    def predict(self,fields):
        import torch
        from pricing.neural_bid_blend_live import (
            CATS,_norm,checked_price,_auction_date)
        from pricing.hierarchy_sparse_live import base_predict
        numeric,categories,raw,unknown=self.original.features(fields)
        with torch.inference_mode():
            log=self.model(torch.from_numpy(numeric),
                           torch.from_numpy(categories))[:1].numpy()[0]
        fm_price=checked_price(float(np.exp(np.clip(log,
                 np.log(100.),np.log(1e6)))),"CVaR95 DeepFM")
        original={**{k:_norm(fields.get(k)) for k in CATS[:8]},
                  "odometer":float(fields["odometer"]),
                  "auction_grade":float(raw[-1]) if np.isfinite(raw[-1]) else None}
        xgb,missing=base_predict(self.original.xgb,self.original.xgb_schema,None,
                    original,float(raw[1]),"bid")
        xgb=checked_price(xgb,"Original Lemon XGBoost")
        final=checked_price(self.alpha*fm_price+(1-self.alpha)*xgb,
                            "CVaR95 wholesale research ensemble")
        return {"market":"bid","model":"DeepFM16 CVaR95 winner #18 + Lemon XGBoost",
                "price":final,"deepfm_price":fm_price,
                "xgboost_lemon_price":xgb,"neural_weight":self.alpha,
                "historical_price_basis":"2015-index-normalized USD",
                "original_auction_date":_auction_date(fields["auction_date"]).isoformat(),
                "is_completed_auction_sale_estimate":True,
                "live_2026_validated":False,"verified":True,
                "unrecognized_categories":sorted(set(unknown+missing))}


def _load(root,market):
    import streamlit as st
    folder=Path(root)/(ASK_FOLDER if market=="ask" else BID_FOLDER)
    sig=tuple((name,(folder/name).stat().st_mtime_ns,
               (folder/name).stat().st_size) for name in NAMES)
    return _cached(str(root),market,sig)


def _cached(root,market,signature):
    import streamlit as st
    @st.cache_resource(max_entries=2,show_spinner=False)
    def factory(original,side,fingerprint):
        return CVaR95Asking(original) if side=="ask" else CVaR95Wholesale(original)
    return factory(root,market,signature)


def load_ask(root):
    if not ask_available(root):
        raise FileNotFoundError("CVaR95 Ask release not verified")
    return _load(root,"ask")


def load_bid(root):
    if not bid_available(root):
        raise FileNotFoundError("CVaR95 Bid release not verified")
    return _load(root,"bid")
