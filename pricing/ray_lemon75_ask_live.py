"""Fail-closed, unmodified FP32 PyTorch serving for Ray Lemon75 DeepCross Ask.

The original neural Ask price network is never quantized or retrained.
The separately verified shared BGE text encoder may use its independently
gated INT8 runtime; that does not change the FP32 price network.
The exact same native network is reused for on-click Captum explanations.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

FOLDER = "models/neural-ray-lemon75-ask"
REQUIRED = ("weights.pt", "manifest.json", "verification.json")
STUDY_RUN = 42485146978
STUDY_ARTIFACT = 13621540013
RELEASE_DAY = "2026-11-08"


def same_architecture(actual,expected):
    """Exact architecture keys and structure, with float serialization tolerance."""
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


def _hash(path):
    with open(path,"rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def manifest(root):
    folder=Path(root)/FOLDER
    if not all((folder/name).is_file() for name in REQUIRED):
        return None
    try:
        m=json.loads((folder/"manifest.json").read_text())
        v=json.loads((folder/"verification.json").read_text())
        if not (
            m.get("status")=="passed"
            and m.get("market")=="ask"
            and m.get("release_date")==RELEASE_DAY
            and m.get("study_run_id")==STUDY_RUN
            and m.get("artifact_id")==STUDY_ARTIFACT
            and m.get("architecture")=="deepcross"
            and m.get("training_cohort")=="frozen-lemon-oof-retained-75"
            and m.get("quantization")=="none-native-fp32"
            and m.get("inference_backend")=="pytorch-native-fp32"
            and m.get("native_checkpoint_for_predictions") is True
            and m.get("lemon_retained_train_rows")==110832
            and m.get("filtered_validation_rows")==23751
            and m.get("original_validation_rows")==31668
            and m.get("filtered_validation_is_conditional") is True
            and m.get("original_test_scored") is False
            and m.get("native_head_for_explanations") is True
            and v.get("status")=="passed"
            and v.get("verified") is True
            and v.get("complete_original_validation_replay") is True
            and v.get("filtered_validation_replay") is True
            and v.get("native_checkpoint_replay") is True
            and v.get("production_source_unchanged") is True
            and v.get("test_evaluated") is False
            and v.get("native_sha256")==m.get("native_sha256")
            and v.get("unmodified_checkpoint_sha256")==m.get("native_sha256")
            and _hash(folder/"weights.pt")==m.get("native_sha256")
        ):
            return None
        return m
    except (OSError,ValueError,TypeError,KeyError):
        return None


def available(root):
    from pricing.neural_ask_embedding_live import available as original
    return original(root) and manifest(root) is not None


class RayLemonDeepCrossAsk:
    """Original frozen BGE preprocessing with the original FP32 DeepCross head."""

    def __init__(self,root):
        from pricing.neural_ask_embedding_live import load as original_load
        self.root=Path(root)
        self.manifest=manifest(root)
        if self.manifest is None or not available(root):
            raise FileNotFoundError("Audited original Lemon75 FP32 Ask release not installed")
        self.original=original_load(root)
        self._native_model=None

    def prepare_text(self,description):
        return self.original.prepare_text(description)

    def predict_prepared(self,fields,prepared):
        import torch
        numeric,category,vector=self.original.encode_inputs(fields,prepared)
        with torch.inference_mode():
            output=self.model(
                torch.from_numpy(np.asarray(numeric,dtype=np.float32)),
                torch.from_numpy(np.asarray(category,dtype=np.int64)),
                torch.from_numpy(np.asarray(vector,dtype=np.float32)))
        raw=float(output[0,0].item())
        price=float(np.exp(np.clip(raw,math.log(100.),math.log(1_000_000.))))
        if not math.isfinite(price) or price<=0:
            raise ValueError("Native Lemon75 DeepCross returned an invalid price")
        return {
            "market":"ask","price":price,
            "model":"08 Nov 2026 · Lemon75 DeepCross · Original FP32",
            "historical_anchor":"2021-05-01",
            "text_price_heads":0,
            "price_predictions_from":"Full original FP32 PyTorch DeepCross price head",
            "current_market_accuracy_validated":False,
        }

    def predict(self,description,fields,explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))

    @property
    def model(self):
        """One original FP32 PyTorch model serves both pricing and Captum."""
        if self._native_model is None:
            import torch
            from scripts.ask25_joint_search import ResearchAskingNetwork
            from scripts.ray_pbt_ask import architecture_params
            pack=torch.load(self.root/FOLDER/"weights.pt",
                            map_location="cpu",weights_only=True)
            expected=architecture_params("deepcross")
            if (not same_architecture(pack.get("architecture"),expected)
                    or pack.get("training_cohort")!="frozen-lemon-oof-retained-75"
                    or pack.get("original_test_scored") is not False
                    or pack.get("cohort_sha256")!=self.manifest.get("cohort_sha256")
                    or pack.get("frozen_bge_sha256") is None):
                raise ValueError("Original Ask checkpoint does not match audited provenance")
            network=ResearchAskingNetwork(self.original.cards,384,expected)
            network.load_state_dict(pack["state_dict"],strict=True)
            network.eval()
            self._native_model=network
        return self._native_model


def load(root):
    import streamlit as st
    folder=Path(root)/FOLDER
    if not available(root):
        raise FileNotFoundError("Lemon75 DeepCross FP32 release has not passed audit")
    stamp=tuple((name,(folder/name).stat().st_size,
                 (folder/name).stat().st_mtime_ns) for name in REQUIRED)
    @st.cache_resource(show_spinner=False,max_entries=1)
    def _cache(origin,fingerprint):
        return RayLemonDeepCrossAsk(origin)
    return _cache(str(root),stamp)
