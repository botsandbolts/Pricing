"""Verified overnight BGE DeepCross Ask checkpoint, selected on training-only CVaR95.

Uses frozen 256-token BGE and exact original category/numeric preprocessing.
The English BGE encoder may run INT8 ONNX only if its *separate* parity
certificate passes; otherwise it remains native rather than misreporting INT8.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import numpy as np

FOLDER=Path("models/neural-bge-bohb-overnight-ask")
EXPECTED_SHA="d5e669d33fb34a379ca02837989beca757d43a1586cc99fa86e5c58d64a45c39"
COHORT_SHA="746bc9f8e02e1e708ea5e1d9479408effb131eb6b1d1ea7710380701e62dcf12"
EMBEDDING_SHA="d619d6b8b4bb17475b500dbfb7835aeb125609c3ed54dbd6113264a2a7569a33"
CLEAN_CVAR95=8740.063663572575


def _sha(file):
    with Path(file).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def manifest(root):
    folder=Path(root)/FOLDER
    try:
        weights=folder/"weights.pt"
        meta=json.loads((folder/"manifest.json").read_text())
        if not (weights.is_file() and meta.get("status")=="passed"
                and meta.get("source_run_id")==42504289617
                and meta.get("source_artifact_id")==13650299818
                and meta.get("source_slot")==5
                and meta.get("cohort_sha256")==COHORT_SHA
                and meta.get("embedding_sha256")==EMBEDDING_SHA
                and meta.get("checkpoint_sha256")==EXPECTED_SHA
                and meta.get("cleaned_validation",{}).get("CVaR95_absolute_USD")==CLEAN_CVAR95
                and meta.get("original_test_scored") is False
                and _sha(weights)==EXPECTED_SHA):
            return None
        return meta
    except (OSError,ValueError,TypeError):
        return None


def available(root):
    from pricing.neural_ask_embedding_live import available as original_available
    return original_available(root) and manifest(root) is not None


class BGEOvernightAsking:
    def __init__(self, root):
        import torch
        from pricing.neural_ask_embedding_live import load as original_load
        from scripts.ask25_joint_search import ResearchAskingNetwork
        from scripts.ray_pbt_ask import architecture_params
        root=Path(root)
        self.manifest=manifest(root)
        if self.manifest is None or not available(root):
            raise FileNotFoundError("Unverified BGE overnight Ask head")
        # Reuse the very same pretrained frozen BGE encoder. If a previously
        # parity-tested INT8 export is installed, the original loader selects it.
        self.original=original_load(root)
        pack=torch.load(root/FOLDER/"weights.pt",map_location="cpu",weights_only=True)
        expected=architecture_params("deepcross")
        shape_keys=("family","project_dim","project_layers","project_dropout",
                    "cross_layers","rank","experts","width","blocks","dropout","activation")
        if (any(pack.get("architecture",{}).get(k)!=expected.get(k) for k in shape_keys)
                or pack.get("cohort_sha256")!=COHORT_SHA
                or pack.get("embedding_sha256")!=EMBEDDING_SHA
                or pack.get("training_cohort")!="frozen-lemon-oof-retained-75"
                or pack.get("test_scored") is not False
                or pack.get("best_record",{}).get("inner_metrics",{}).get("CVaR95_absolute_USD")!=7941.740520305461):
            raise ValueError("BGE overnight checkpoint is not the verified stage-3 winner")
        self.price_network=ResearchAskingNetwork(self.original.cards,384,expected)
        self.price_network.load_state_dict(pack["state_dict"],strict=True)
        self.price_network.eval()
        self.encoder_backend=getattr(self.original,"encoder_backend","native-pytorch")

    def prepare_text(self, description):
        return self.original.prepare_text(description)

    def predict_prepared(self, fields, prepared):
        import torch
        num,cat,text=self.original.encode_inputs(fields,prepared)
        with torch.inference_mode():
            raw=float(self.price_network(
                torch.from_numpy(np.asarray(num,dtype=np.float32)),
                torch.from_numpy(np.asarray(cat,dtype=np.int64)),
                torch.from_numpy(np.asarray(text,dtype=np.float32)))[0,0].item())
        price=float(np.exp(np.clip(raw,math.log(100.),math.log(1000000.))))
        if not math.isfinite(price) or price<=0:
            raise ValueError("BGE overnight price network returned an invalid price")
        return {"market":"ask","price":price,
                "model":"BGE DeepCross BOHB · 9 Nov 2026 · "+self.encoder_backend,
                "historical_anchor":"2021-05-01","text_price_heads":0,
                "current_market_accuracy_validated":False}

    def predict(self, description, fields, explain=False):
        return self.predict_prepared(fields,self.prepare_text(description))


def load(root):
    import streamlit as st
    folder=Path(root)/FOLDER
    if not available(root):
        raise FileNotFoundError("BGE overnight release has not passed its gate")
    stamp=tuple((name,(folder/name).stat().st_size,
                 (folder/name).stat().st_mtime_ns)
                for name in ("manifest.json","weights.pt"))
    @st.cache_resource(show_spinner=False,max_entries=1)
    def _cache(path,signature):
        return BGEOvernightAsking(path)
    return _cache(str(root),stamp)
